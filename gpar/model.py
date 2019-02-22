# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging
from types import FunctionType

from lab import B
from plum import Dispatcher, Referentiable, Self
from stheno import Delta, GP, Obs, SparseObs

__all__ = ['GPAR']
log = logging.getLogger(__name__)


def _merge(x, updates, to_update):
    # Stack them, which screws up the order.
    concat = B.concat([x[~to_update], updates], axis=0)

    # Generate an index mapping to fix the ordering.
    original_i = 0
    update_i = B.sum(~to_update)
    indices = []
    for i in range(len(to_update)):
        # Careful not to update the indices in-place!
        if to_update[i]:
            indices.append(update_i)
            update_i = update_i + 1
        else:
            indices.append(original_i)
            original_i = original_i + 1

    # Perform the fix.
    return B.take(concat, indices)


def _construct_model(f, noise):
    return lambda: (f, noise)


class GPAR(Referentiable):
    """Basic GPAR model.

    Args:
        replace (bool, optional): Condition on the predictive mean instead of
            the data. Defaults to `False`.
        impute (bool, optional): Impute missing data points with the predictive
            mean to make the data set closed downwards. Defaults to `False`.
        x_ind (tensor, optional): Locations of inducing points
            for a sparse approximation. Defaults to `None`.
    """
    _dispatch = Dispatcher(in_class=Self)

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

    @_dispatch(FunctionType)
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
            gpar.layers.append(_construct_model(state.f, state.noise))

        return gpar

    def logpdf(self, x, y, only_last_layer=False, unbiased_sample=False):
        """Compute the logpdf.

        Args:
            x (tensor): Inputs.
            y (tensor): Outputs.
            only_last_layer (bool, optional): Compute the pdf of only the last
                layer. Defaults to `False`.
            unbiased_sample (bool, optional): Compute an unbiased sample of the
                logpdf. Defaults to `False`.

        Returns:
            :class:`.gpar.GPAR`: Updated GPAR model.
        """
        logpdf = 0
        state = GPARState(self, x, self.x_ind)

        for i, ((y, mask), model) in enumerate(zip(per_output(y, self.impute),
                                                   self.layers)):
            state.next_layer(model, mask)
            state.observe(y)

            # Accumulate logpdf.
            last_layer = i == len(self.layers) - 1
            if ~only_last_layer or (last_layer and only_last_layer):
                logpdf += state.compute_logpdf()

            # Sample missing data for an unbiased sample of the logpdf.
            if unbiased_sample:
                state.sample_missing()

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
            f_sample, y_sample = state.sample(state.x)
            sample = B.concat([sample, f_sample if latent else y_sample],
                              axis=1)

            # Feed sample into the next layer.
            state.observe(y_sample)

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
        # The latent and observed function will not be exposed directly. This
        # is to implement lazy computation the posterior.
        self._f = None
        self._f_noisy = None
        self._obs = None  # Observations on which is conditioned.
        self._obs_queue = None  # Observations on which need to be conditioned.

        # Observations of the current layer:
        self.y = None
        self.available = None

    def next_layer(self, model, mask=None):
        """Move to the next layer.

        Args:
            model (function): Model constructor.
            mask (tensor, optional): Boolean mask that determines which data
                points to keep with respect to the previous layer. If not
                specified, all data points are kept.
        """
        # If this is not the first layer, update concatenate the previous
        # outputs to the inputs.
        if not self.first_layer:
            self._update_inputs()
        else:
            self.first_layer = False

        # Filter inputs according to mask if one is given.
        if mask is not None:
            self.x = self.x[mask]

        # Empty observation queue.
        self._obs = []
        self._obs_queue = []

        # Construct model.
        self._f, self.noise = model()
        self.graph = self._f.graph
        self.e = GP(Delta() * self.noise, graph=self.f.graph)
        self._f_noisy = self.f + self.e

    @property
    def f(self):
        """Latent process."""
        self._process_obs_queue()
        return self._f

    @property
    def f_noisy(self):
        """Observed process."""
        self._process_obs_queue()
        return self._f_noisy

    def _process_obs_queue(self):
        for x, y in self._obs_queue:
            obs = self._generate_obs(x, y)
            self._f |= obs
            self._f_noisy |= obs
            self._obs.append(obs)
        self._obs_queue = []

    def _generate_obs(self, x, y):
        if self.gpar.sparse:
            return SparseObs(self._f(self.x_ind), self.e, self._f(x), y)
        else:
            return Obs(self._f_noisy(x), y)

    def _update_inputs(self):
        # Update inputs of inducing points.
        if self.gpar.sparse:
            self.x_ind = B.concat([self.x_ind,
                                   self.f.mean(self.x_ind)], axis=1)

        if self.gpar.impute and self.gpar.replace:
            # Impute missing data and replace available data.
            self.y = self.f.mean(self.x)

        elif self.gpar.impute and B.any(~self.available):
            # Just impute missing data.
            self.y = _merge(self.y,
                            self.f.mean(self.x[~self.available]),
                            ~self.available)

        elif self.gpar.replace and B.any(self.available):
            # Just replace available data.
            self.y = _merge(self.y,
                            self.f.mean(self.x[self.available]),
                            self.available)

        # Finally, actually update inputs.
        self.x = B.concat([self.x, self.y], axis=1)

    def observe(self, y):
        """Observe values for the current layer.

        Args:
            y (tensor): Observations.
        """
        # Save observations and add to observation queue for the posterior.
        self.y = y
        self.available = ~B.isnan(y[:, 0])
        self._obs_queue.append((self.x[self.available],
                                self.y[self.available]))

    def compute_logpdf(self):
        """Compute the logpdf of the observations.

        Returns:
            float: Logpdf of observations.
        """
        self._process_obs_queue()
        return sum([self.graph.logpdf(obs) for obs in self._obs])

    def sample(self, x):
        """Sample at particular inputs.

        Args:
            x (tensor): Inputs to sample at.

        Returns:
            tensor: Sample.
        """
        # Ancestral sampling is more efficient than joint sampling.
        f_sample = self.f(x).sample()
        e_sample = self.e(x).sample()
        return f_sample, f_sample + e_sample

    def sample_missing(self):
        """Sample missing observations."""
        if B.any(~self.available):
            # Sample missing data.
            _, y_missing = self.sample(self.x[~self.available])

            # Merge into the observations.
            self.y = _merge(self.y, y_missing, ~self.available)

            # Condition on the delta.
            self._obs_queue.append((self.x[~self.available], y_missing))


def per_output(y, impute=False):
    """Return observations per output, respecting that the data must be
    closed downwards.

    Args:
        y (tensor): Outputs.
        impute (bool, optional): Also return missing observations that would
            make the data closed downwards.

    Returns:
        generator: Generator that generates tuples containing the
            observations per layer and a mask which observations are not
            missing relative to the previous layer.
    """
    p = B.shape_int(y)[1]  # Number of outputs

    for i in range(p):
        # Check availability.
        available = ~B.isnan(y)
        mask = available[:, i]

        # Take into account future observations if necessary.
        if impute and i < p - 1:
            # Careful not to update the mask in-place!
            mask = mask | B.any(available[:, i + 1:], axis=1)

        # Give stuff back.
        yield y[mask, i:i + 1], mask

        # Filter observations.
        y = y[mask]
