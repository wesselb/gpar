# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from lab.torch import B
from gpar.model import merge, construct_model, last, per_output
import numpy as np

# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, lam, allclose


def test_merge():
    original = B.array([1, 2, 3, 4])
    updates = B.array([5, 6])

    result = merge(original, updates, B.array([True, True, False, False]))
    yield allclose, result, [5, 6, 3, 4]

    result = merge(original, updates, B.array([True, False, True, False]))
    yield allclose, result, [5, 2, 6, 4]


def test_construct_model():
    model = construct_model(1, 2)
    yield eq, model(), (1, 2)


def test_last():
    xs = [1, 2, 3, 4]
    yield eq, last(xs), [(False, 1), (False, 2), (False, 3), (True, 4)]
    yield eq, last(xs, [1, 2]), [(False, 2), (False, 3)]
    yield eq, last(xs, [0, 3]), [(False, 1), (True, 4)]


def test_per_output():
    y = B.array([[1, 2, np.nan, np.nan],
                 [3, np.nan, 4, np.nan],
                 [5, 6, 7, np.nan],
                 [8, np.nan, np.nan, np.nan],
                 [9, 10, np.nan, np.nan],
                 [11, np.nan, np.nan, 12]])

    expected = [([1, 3, 5, 8, 9, 11], [True, True, True, True, True, True]),
                ([2, 6, 10], [True, False, True, False, True, False]),
                ([7], [False, True, False]),
                ([], [False])]
    result = [(x.numpy()[:, 0].tolist(), y.numpy().tolist())
              for x, y in per_output(y, keep=False)]
    yield eq, result, expected

    expected = [([1, 3, 5, 8, 9, 11], [True, True, True, True, True, True]),
                ([2, -1, 6, 10, -1], [True, True, True, False, True, True]),
                ([4, 7, -1], [False, True, True, False, True]),
                ([12], [False, False, True])]
    result = [([-1 if np.isnan(z) else z for z in x.numpy()[:, 0].tolist()],
               y.numpy().tolist())
              for x, y in per_output(y, keep=True)]
    yield eq, result, expected
