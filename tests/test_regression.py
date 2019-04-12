# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
from gpar.regression import _uprank, _vector_from_init, log_transform, \
    squishing_transform

# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, lam, allclose, \
    approx, array


def test_transforms():
    f, f_inv = log_transform
    yield allclose, f(f_inv(array([1, 2, 3, 4]))), array([1, 2, 3, 4])
    f, f_inv = squishing_transform
    yield allclose, f(f_inv(array([-2, -1, 3, 4]))), array([-2, -1, 3, 4])


def test_uprank():
    yield allclose, _uprank(1), np.ones((1, 1))
    yield allclose, _uprank(np.ones(1)), np.ones((1, 1))
    yield allclose, _uprank(np.ones((1, 1))), np.ones((1, 1))
    yield raises, ValueError, lambda: _uprank(np.ones((1, 1, 1)))


def test_vector_from_init():
    yield allclose, _vector_from_init(2, 2), np.array([2, 2])
    yield allclose, \
          _vector_from_init(np.array([1, 2, 3]), 2), \
          np.array([1, 2])
    yield raises, \
          ValueError, \
          lambda: _vector_from_init(np.random.randn(2, 2), 1)
    yield raises, \
          ValueError, \
          lambda: _vector_from_init(np.array([1, 2]), 3)
