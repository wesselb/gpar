# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging
from types import FunctionType

from lab import B
from plum import Dispatcher, Referentiable, Self
from stheno import Delta, GP, Obs, SparseObs
import numpy as np

from .data import Data

__all__ = ['GPAR']
log = logging.getLogger(__name__)


def _merge(x, updates, to_update):
    # Stack them, which screws up the order.
    concat = B.concat([x[~to_update], updates], axis=0)

    # Generate an index mapping to fix the ordering.
    original_i = 0
    update_i = sum(~to_update)
    indices = []
    for i in range(len(to_update)):
        if to_update[i]:
            indices.append(update_i)
            update_i += 1
        else:
            indices.append(original_i)
            original_i += 1

    # Perform the fix.
    return B.take(concat, indices)


def _construct_model_generator(f, noise):
    return lambda: (f, noise)


class GPAR(Referentiable):
    """Basic GPAR model.

    Args:
        replace (bool, optional): Condition on the predictive mean instead of
            the data. Defaults to `False`.
        impute (bool, optional): Impute missing data points with the predictive
            mean to make the data set closed downwards. Defaults to `False`.
        x_sparse (tensor, optional): Locations of inducing points for a sparse
            approximation. Defaults to `None`.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, replace=False, impute=False, x_ind=None):
        self.replace = replace
        self.impute = impute
        self.layers = []

        # Parse inputs of inducing points.
        if x_ind is None:
            self.sparse = False
            self.x_ind = None
        else:
            self.sparse = True
            if B.rank(x_ind) == 1:
                self.x_ind = B.expand_dims(x_ind, 1)
            elif B.rank(x_ind) == 2:
                self.x_ind = x_ind
            else:
                raise ValueError('Invalid rank {} for locations of the '
                                 'inducing points.'.format(B.rank(x_ind)))

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

    @_dispatch(Data)
    def __or__(self, data):
        """Condition.

        Args:
            data (:class:`.data.Data`): Data to condition on.

        Returns:
            :class:`.gpar.GPAR`: Updated GPAR model.
        """
        gpar, xs, xs_ind = self.copy(), data.x, self.x_ind

        for (y, mask), model in zip(data.per_output(self.impute), self.layers):
            # Filter inputs according to the mask.
            xs = xs[mask]

            # Construct model.
            f, noise = model()
            e = GP(noise * Delta(), graph=f.graph)
            f_noisy = f + e

            # Condition model.
            avail = ~np.isnan(y[:, 0])  # Filter for available data.
            if self.sparse:
                obs = SparseObs(f(xs_ind), e, f(xs[avail]), y[avail])
            else:
                obs = Obs(f_noisy(xs[avail]), y[avail])
            f_post = f | obs

            # Update model.
            gpar.layers.append(_construct_model_generator(f_post, noise))

            # Update inputs.
            xs, xs_ind = self._update_inputs(xs, xs_ind, y, f_post)

        return gpar

    @_dispatch(Data)
    def logpdf(self, data, only_last_layer=False):
        """Compute the logpdf.

        Args:
            data (:class:`.data.Data`): Data to compute logpdf of.
            only_last_layer (bool, optiona): Compute the pdf of only the last
                layer. Defaults to `False`.

        Returns:
            :class:`.gpar.GPAR`: Updated GPAR model.
        """
        logpdf, xs, xs_ind = 0, data.x, self.x_ind

        for i, ((y, mask), model) in enumerate(zip(data.per_output(self.impute),
                                                   self.layers)):
            # Filter inputs according to the mask.
            xs = xs[mask]

            # Construct model.
            f, noise = model()
            e = GP(noise * Delta(), graph=f.graph)
            f_noisy = f + e

            # Check whether this is the last layer.
            last_layer = i == len(self.layers) - 1
            # Check whether a posterior is needed.
            need_posterior = self.impute or self.replace or self.sparse
            # Check whether the logpdf should be accumulated.
            accumulate = (last_layer and only_last_layer) or ~only_last_layer

            # Compute observations if needed.
            if need_posterior or accumulate:
                avail = ~np.isnan(y[:, 0])  # Filter for available data.
                if self.sparse:
                    obs = SparseObs(f(xs_ind), e, f(xs[avail]), y[avail])
                else:
                    obs = Obs(f_noisy(xs[avail]), y[avail])

            # Accumate logpdf if needed.
            if accumulate:
                if self.sparse:
                    logpdf += obs.elbo
                else:
                    logpdf += obs.x.logpdf(obs.y)

            # Compute posterior if needed.
            f_post = (f | obs) if need_posterior else None

            # Update inputs.
            xs, xs_ind = self._update_inputs(xs, xs_ind, y, f_post)

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
        sample, xs = None, x

        for model in self.layers:
            # Construct model.
            f, noise = model()
            e = GP(noise * Delta(), graph=f.graph)

            # Sample current output.
            f_sample = f(xs).sample()
            y_sample = f_sample + e(xs).sample()

            # Update sample.
            selected_sample = f_sample if latent else y_sample
            if sample is None:
                sample = selected_sample
            else:
                sample = B.concat([sample, selected_sample], axis=1)

            # Replace data.
            if self.replace:
                y_sample = f.mean(xs)

            # Update inputs.
            xs = B.concat([xs, y_sample], axis=1)

        return sample

    def _update_inputs(self, xs, xs_ind, y, f_post):
        available = ~np.isnan(y[:, 0])

        # Update inputs of inducing points.
        if xs_ind is not None:
            xs_ind = B.concat([xs_ind, f_post.mean(xs_ind)], axis=1)

        if self.impute and self.replace:
            # Impute missing data and replace available data:
            y = f_post.mean(xs)
        elif self.impute:
            # Just impute missing data.
            y = _merge(y, f_post.mean(xs[~available]), ~available)
        elif self.replace:
            # Just replace available data.
            y = _merge(y, f_post.mean(xs[available]), available)

        # Update inputs.
        xs = B.concat([xs, y], axis=1)

        return xs, xs_ind
