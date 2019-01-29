# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging
from types import FunctionType

from lab import B
from plum import Dispatcher, Referentiable, Self
from stheno import Delta, GP, Obs

from .data import Data

__all__ = ['GPAR']
log = logging.getLogger(__name__)


def _construct_model_generator(f, noise):
    return lambda: (f, noise)


class GPAR(Referentiable):
    """Basic GPAR model.

    Args:
        replace (bool): Condition on the predictive mean instead of the data.
        impute (bool): Impute missing data points with the predictive mean to
            make the data set closed downwards.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, replace=False, impute=False):
        self.replace = replace
        self.impute = impute
        self.layers = []

    def copy(self):
        """Create a new GPAR model with the same configuration.

        Returns:
            :class:`.gpar.GPAR`: New GPAR model with the same configuration.
        """
        gpar = GPAR(replace=self.replace,
                    impute=self.impute)
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
        gpar, xs = self.copy(), data.x

        for (y, mask), model in zip(data.per_output(), self.layers):
            # Construct model.
            f, noise = model()
            f_noisy = f + GP(noise * Delta(), graph=f.graph)

            # Condition and update new model.
            obs = Obs(f_noisy(xs), y)
            gpar.layers.append(_construct_model_generator(f | obs, noise))

            # Update inputs.
            xs = B.concat([xs[mask], y], axis=1)

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
            sample_f = f(xs).sample()
            sample_y = sample_f + e(xs).sample()

            # Update sample.
            this_sample = sample_f if latent else sample_y
            if sample is None:
                sample = this_sample
            else:
                sample = B.concat([sample, this_sample], axis=1)

            # Update inputs.
            xs = B.concat([xs, sample_y], axis=1)

        return sample
