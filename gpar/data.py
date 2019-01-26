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

    def per_output(self):
        """Return observations per output, respecting that the data must be
        closed downwards.

        Returns:
            generator: Generator that generates tuples containing the
                observations per layer and a mask which observations are not
                missing relative to the previous layer.
        """
        mask = np.ones(self.n).astype(np.bool)
        y = self.y

        for i in range(self.p):
            # Update mask and outputs.
            mask = ~np.isnan(y[mask, i])
            y = y[mask]

            # Give them.
            yield y, mask
