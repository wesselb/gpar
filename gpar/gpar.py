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

    def __init__(self, replace=False, impute=False, x_sparse=None):
        self.replace = replace
        self.impute = impute
        self.layers = []

        # Parse inputs of inducing points.
        if x_sparse is None:
            self.x_sparse = None
        else:
            if B.rank(x_sparse) == 1:
                self.x_sparse = B.expand_dims(x_sparse, 1)
            elif B.rank(x_sparse) == 2:
                self.x_sparse = x_sparse
            else:
                raise ValueError('Invalid rank {} for locations of the '
                                 'inducing points.'.format(B.rank(x_sparse)))

    def copy(self):
        """Create a new GPAR model with the same configuration.

        Returns:
            :class:`.gpar.GPAR`: New GPAR model with the same configuration.
        """
        gpar = GPAR(replace=self.replace,
                    impute=self.impute,
                    x_sparse=self.x_sparse)
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
        gpar, xs, xs_sparse = self.copy(), data.x, self.x_sparse

        for (y, mask), model in zip(data.per_output(self.impute), self.layers):
            # Filter inputs according to the mask.
            xs = xs[mask]

            # Construct model.
            f, noise = model()
            e = GP(noise * Delta(), graph=f.graph)
            f_noisy = f + e

            # Condition on available data.
            available = ~np.isnan(y[:, 0])
            if xs_sparse is not None:
                obs = SparseObs(f(xs_sparse),
                                e,
                                f(xs[available]),
                                y[available])
            else:
                obs = Obs(f_noisy(xs[available]), y[available])
            f_post = f | obs

            # Update inputs of inducing points.
            if xs_sparse is not None:
                xs_sparse = B.concat([xs_sparse,
                                      f_post.mean(xs_sparse)], axis=1)

            # Update model.
            gpar.layers.append(_construct_model_generator(f_post, noise))

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

        return gpar

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
