# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging

import numpy as np

__all__ = ['Data']
log = logging.getLogger(__name__)


class Data(object):
    """Data set.

    Args:
        x (tensor): Inputs with rows corresponding to data points and columns to
            features.
        y (tensor): Multiple outputs with rows corresponding to data points and
            columns to outputs.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

        # Set some handy properties.
        self.n = x.shape[0]  # Number of data points
        self.m = x.shape[1]  # Number of features
        self.p = y.shape[1]  # Number of outputs

    def per_output(self, impute=False):
        """Return observations per output, respecting that the data must be
        closed downwards.

        Args:
            impute (bool, optional): Also return missing observations that would
                make the data closed downwards.

        Returns:
            generator: Generator that generates tuples containing the
                observations per layer and a mask which observations are not
                missing relative to the previous layer.
        """
        y = self.y

        for i in range(self.p):
            # Check availability.
            mask = ~np.isnan(y[:, i])

            # Take into account future observations.
            if impute:
                future = np.any(~np.isnan(y[:, i + 1:]), axis=1)
                mask |= future

            # Give stuff back.
            yield y[mask, i:i + 1], mask

            # Filter observations.
            y = y[mask]
