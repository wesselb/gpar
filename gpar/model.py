# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging

from lab import B
from stheno import Delta, GP, Obs, SparseObs

__all__ = ['GPAR']
log = logging.getLogger(__name__)


def _merge(x, updates, to_update):
    # Stack them, which screws up the order.
    concat = B.concat([x[~to_update], updates], axis=0)

    # Generate an index mapping to fix the ordering.
    i_original = 0
    i_update = B.sum(~to_update)
    indices = []
    for i in range(len(to_update)):
        # Careful not to update the indices in-place! This generates trouble
        # with PyTorch.
        if to_update[i]:
            indices.append(i_update)
            i_update = i_update + 1
        else:
            indices.append(i_original)
            i_original = i_original + 1

    # Perform the fix.
    return B.take(concat, indices)


def _construct_model(f, noise):
    return lambda: (f, noise)


class GPAR(object):
    """Basic GPAR model.

    Args:
        replace (bool, optional): Condition on the predictive mean instead of
            the data. Defaults to `False`.
        impute (bool, optional): Impute missing data points with the predictive
            mean to make the data set closed downwards. Defaults to `False`.
        x_ind (tensor, optional): Locations of inducing points
            for a sparse approximation. Defaults to `None`.
    """

    def __init__(self, replace=False, impute=False, x_ind=None):
        self.replace = replace
        self.impute = impute
        self.layers = []

        # Parse inputs of inducing points.
        self.sparse = x_ind is not None
        self.x_ind = None if x_ind is None else x_ind

    def copy(self):
        """Create a new GPAR model with the same configuration.

        Returns:
            :class:`.gpar.GPAR`: New GPAR model with the same configuration.
        """
        gpar = GPAR(replace=self.replace,
                    impute=self.impute,
                    x_ind=self.x_ind)
        return gpar

    def add_layer(self, model_constructor):
        """Add a layer.

        Args:
            model_constructor (function): Constructor of the model, which should
                return a tuple containing the GP and the noise.

        Returns:
            :class:`.gpar.GPAR`: Updated GPAR model.
        """
        gpar = self.copy()
        gpar.layers = list(self.layers) + [model_constructor]
        return gpar

    def __or__(self, x_and_y):
        """Condition on data.

        Args:
            x (tensor): Inputs.
            y (tensor): Outputs.

        Returns:
            :class:`.gpar.GPAR`: Updated GPAR model.
        """
        x, y = x_and_y
        gpar = self.copy()
        state = GPARState(gpar, x, self.x_ind)

        for (y, mask), model in zip(per_output(y, self.impute), self.layers):
            state.next_layer(model, mask)
            state.observe(y)

            # Update with the posterior.
            gpar.layers.append(_construct_model(state.f_post, state.noise))

        return gpar

    def logpdf(self, x, y, only_last_layer=False, unbiased_sample=False):
        """Compute the logpdf.

        Args:
            x (tensor): Inputs.
            y (tensor): Outputs.
            only_last_layer (bool, optional): Compute the pdf of only the last
                layer. Defaults to `False`.
            unbiased_sample (bool, optional): Compute an unbiased sample of the
                pdf, _not_ logpdf. Defaults to `False`.

        Returns:
            :class:`.gpar.GPAR`: Updated GPAR model.
        """
        logpdf = B.cast(0, dtype=B.dtype(x))
        state = GPARState(self, x, self.x_ind)

        y_per_output = per_output(y, self.impute or unbiased_sample)
        for i, ((y, mask), model) in enumerate(zip(y_per_output, self.layers)):

            # Sample missing to yield an unbiased estimate.
            state.next_layer(model, mask, sample_missing=unbiased_sample)
            state.observe(y)

            # Accumulate logpdf.
            last_layer = i == len(self.layers) - 1
            if ~only_last_layer or (last_layer and only_last_layer):
                logpdf += state.compute_logpdf()

        return logpdf

    def sample(self, x, latent=False):
        """Sample.

        Args:
            x (tensor): Inputs to sample at.
            latent (bool, optional): Sample latent function. Defaults to
                `False`.

        Returns:
            tensor: Sample.
        """
        sample = B.zeros((B.shape(x)[0], 0), dtype=B.dtype(x))
        state = GPARState(self, x, self.x_ind)

        for model in self.layers:
            state.next_layer(model)

            # Sample the current layer.
            f, y = state.sample(state.x)
            sample = B.concat([sample, f if latent else y], axis=1)

            # Feed sample into the next layer.
            state.observe(y)

        return sample


class GPARState(object):
    """Constructing GPAR to do various things involves maintaining a fairly
    complicated state. This class implements that state.

    Args:
        gpar (:class:`.model.GPAR`): GPAR model.
        x (tensor): Inputs of the training data.
        x_ind (tensor): Inputs of the inducing points.
    """

    def __init__(self, gpar, x, x_ind):
        self.gpar = gpar
        self.x = x
        self.x_ind = x_ind
        self.first_layer = True

        # Model components:
        self.graph = None
        self.noise = None
        self.e = None
        self.f = None
        self.f_noisy = None

        # Observations of the current layer:
        self.y = None

        # The posterior of the latent and observed function will not be exposed
        # directly, to implement lazy computation of the posterior.
        self._f_post = None
        self._f_noisy_post = None
        self._obs = None
        self._obs_args = None

    def next_layer(self, model, mask=None, sample_missing=False):
        """Move to the next layer.

        Args:
            model (function): Model constructor.
            mask (tensor, optional): Boolean mask that determines which data
                points to keep with respect to the previous layer. If not
                specified, all data points are kept.
            sample_missing (bool, optional): Sample missing data. Defaults to
                `False`.
        """
        # If this is not the first layer, update concatenate the previous
        # outputs to the inputs.
        if not self.first_layer:
            self._update_inputs(sample_missing=sample_missing)

            # Clear observations:
            self.y = None

            # Clear posterior:
            self._f_post = None
            self._f_noisy_post = None
            self._obs = None
            self._obs_args = None
        else:
            self.first_layer = False

        # Filter inputs according to a mask if one is given.
        if mask is not None:
            self.x = self.x[mask]

        # Construct model.
        self.f, self.noise = model()
        self.graph = self.f.graph
        self.e = GP(Delta() * self.noise, graph=self.f.graph)
        self.f_noisy = self.f + self.e

    @property
    def f_post(self):
        """Latent process."""
        # If no observations are available, simply return the prior.
        if self.obs is None:
            return self.f

        # Construct posterior lazily.
        if self._f_post is None:
            self._f_post = self.f | self.obs
        return self._f_post

    @property
    def f_noisy_post(self):
        """Observed process."""
        # If no observations are available, simply return the prior.
        if self.obs is None:
            return self.f_noisy

        # Construct posterior lazily.
        if self._f_noisy_post is None:
            self._f_noisy_post = self.f_noisy | self.obs
        return self._f_noisy_post

    @property
    def obs(self):
        # If no observations are available, simply return `None` to indicate
        # that that is the case.
        if self._obs_args is None:
            return None

        # Lazily compute observations.
        if self._obs is None:
            x, y = self._obs_args  # Extract arguments from `self._obs_args`!
            if self.gpar.sparse:
                self._obs = SparseObs(self.f(self.x_ind), self.e, self.f(x), y)
            else:
                self._obs = Obs(self.f_noisy(x), y)
        return self._obs

    def _update_inputs(self, sample_missing):
        available = ~B.isnan(self.y[:, 0])

        # TODO: After imputation, should GPAR condition on the imputed data?
        # TODO: Should this happen before updating of sparse inputs?

        # Update inputs of inducing points.
        if self.gpar.sparse:
            self.x_ind = B.concat([self.x_ind,
                                   self.f_post.mean(self.x_ind)], axis=1)

        # Sample missing data
        if sample_missing and B.any(~available):
            self.y = _merge(self.y,
                            self.f_noisy_post(self.x[~available]).sample(),
                            ~available)
            available = ~B.isnan(self.y[:, 0])

        # Impute missing data and replace available data.
        if self.gpar.impute and self.gpar.replace:
            self.y = self.f_post.mean(self.x)
        else:
            # Just impute missing data.
            if self.gpar.impute and B.any(~available):
                self.y = _merge(self.y,
                                self.f_post.mean(self.x[~available]),
                                ~available)

            # Just replace available data.
            if self.gpar.replace and B.any(available):
                self.y = _merge(self.y,
                                self.f_post.mean(self.x[available]),
                                available)

        # Finally, actually update inputs.
        self.x = B.concat([self.x, self.y], axis=1)

    def observe(self, y):
        """Observe values for the current layer.

        Args:
            y (tensor): Observations.
        """
        # Save observations.
        self.y = y

        # If there are any observations, save them in `self._obs_args` for lazy
        # computation of the posterior.
        available = ~B.isnan(y[:, 0])
        if B.any(available):
            self._obs_args = (self.x[available], self.y[available])

    def compute_logpdf(self):
        """Compute the logpdf of the observations.

        Returns:
            float: Logpdf of observations.
        """
        if self.obs is None:
            return 0
        else:
            return self.graph.logpdf(self.obs)

    def sample(self, x):
        """Sample from the _prior_.

        Args:
            x (tensor): Inputs to sample at.

        Returns:
            tensor: Sample.
        """
        # Ancestral sampling is more efficient than joint sampling.
        f_sample = self.f(x).sample()
        e_sample = self.e(x).sample()
        return f_sample, f_sample + e_sample


def per_output(y, keep=False):
    """Return observations per output, respecting that the data must be
    closed downwards.

    Args:
        y (tensor): Outputs.
        keep (bool, optional): Also return missing observations that would
            make the data closed downwards.

    Returns:
        generator: Generator that generates tuples containing the
            observations per layer and a mask which observations are not
            missing relative to the previous layer.
    """
    p = B.shape_int(y)[1]  # Number of outputs

    for i in range(p):
        # Check current and future availability.
        available = ~B.isnan(y)
        future = B.any(available[:, i + 1:], axis=1)

        # Initialise the mask to current availability.
        mask = available[:, i]

        # Take into account future observations if necessary.
        if keep and i < p - 1:  # Check whether this is the last output.
            mask = mask | future

        # Give stuff back.
        yield y[mask, i:i + 1], mask

        # Filter observations.
        y = y[mask]
