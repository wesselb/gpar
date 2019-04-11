# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging

from lab import B
from stheno import Delta, GP, Obs, SparseObs, dense

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


def _construct_model(f, e):
    return lambda: (f, e)


def _last(xs, select=None):
    xs = list(xs)
    is_last = [False for _ in xs]
    is_last[-1] = True
    xs_with_last = list(zip(is_last, xs))
    if select is None:
        return xs_with_last
    else:
        return [xs_with_last[i] for i in select]


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
        gpar, x_ind = self.copy(), self.x_ind

        for last, ((y, mask), model) in \
                _last(zip(per_output(y, self.impute), self.layers)):
            x = x[mask]  # Filter according to mask.
            f, e = model()  # Construct model.
            obs = self._obs(x, x_ind, y, f, e)  # Construct observations.

            # Update with posterior.
            f_post = f | obs
            e_new = GP(e.kernel, graph=f.graph)
            gpar.layers.append(_construct_model(f_post, e_new))

            # Update inputs.
            if not last:
                x, x_ind = self._update_inputs(x, x_ind, y, f, obs)

        return gpar

    def logpdf(self,
               x,
               y,
               only_last_layer=False,
               sample_missing=False,
               return_inputs=False,
               x_ind=None,
               outputs=None):
        """Compute the logpdf.

        Args:
            x (tensor): Inputs.
            y (tensor): Outputs.
            only_last_layer (bool, optional): Compute the pdf of only the last
                layer. Defaults to `False`.
            sample_missing (bool, optional): Sample missing data to compute an
                unbiased estimate of the pdf, *not* logpdf. Defaults to `False`.

        Returns:
            :class:`.gpar.GPAR`: Updated GPAR model.
        """
        logpdf = B.cast(0, dtype=B.dtype(x))
        x_ind = self.x_ind if x_ind is None else x_ind

        y_per_output = per_output(y, self.impute or sample_missing)
        for last, ((y, mask), model) in \
                _last(zip(y_per_output, self.layers), select=outputs):
            x = x[mask]  # Filter according to mask.
            f, e = model()  # Construct model.
            obs = self._obs(x, x_ind, y, f, e)  # Construct observations.

            # Accumulate logpdf.
            if not only_last_layer or (last and only_last_layer):
                logpdf += f.graph.logpdf(obs)

            if not last:
                # Sample missing data for an unbiased sample of the pdf.
                missing = B.isnan(y[:, 0])
                if sample_missing and B.any(missing):
                    y = _merge(y, ((f + e) | obs)(x[missing]).sample(), missing)

                # Update inputs.
                x, x_ind = self._update_inputs(x, x_ind, y, f, obs)

        # Return inputs if asked for.
        return (x, x_ind) if return_inputs else logpdf

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
        x_ind = self.x_ind

        for last, model in _last(self.layers):
            f, e = model()  # Construct model.

            if latent:
                # Sample latent function: use ancestral sampling.
                f_sample = f(x).sample()
                y_sample = f_sample + e(x).sample()
                sample = B.concat([sample, f_sample], axis=1)
            else:
                # Sample observed function.
                y_sample = (f + e)(x).sample()
                sample = B.concat([sample, y_sample], axis=1)

            if not last:
                # Construct observations.
                obs = self._obs(x, x_ind, y_sample, f, e)

                # Update inputs.
                x, x_ind = self._update_inputs(x, x_ind, y_sample, f, obs)

        return sample

    def _obs(self, x, x_ind, y, f, e):
        available = ~B.isnan(y[:, 0])
        if self.sparse:
            return SparseObs(f(x_ind), e, f(x[available]), y[available])
        else:
            return Obs((f + e)(x[available]), y[available])

    def _update_inputs(self, x, x_ind, y, f, obs):
        available = ~B.isnan(y[:, 0])

        def estimate(x_):
            return (f | obs).mean(x_)

        # Update inputs of inducing points.
        if self.sparse:
            x_ind = B.concat([x_ind, estimate(x_ind)], axis=1)

        # Impute missing data and replace available data.
        if self.impute and self.replace:
            y = estimate(x)
        else:
            # Just impute missing data.
            if self.impute and B.any(~available):
                y = _merge(y, estimate(x[~available]), ~available)

            # Just replace available data.
            if self.replace and B.any(available):
                y = _merge(y, estimate(x[available]), available)

        # Finally, actually update inputs.
        x = B.concat([x, y], axis=1)

        return x, x_ind


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
