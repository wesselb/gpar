# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import torch
from numpy.testing import assert_allclose, assert_array_almost_equal
from plum import Dispatcher

_dispatch = Dispatcher()


@_dispatch(object)
def convert(a):
    return a


@_dispatch(torch.Tensor)
def convert(a):
    return a.numpy()


@_dispatch(object, object)
def allclose(a, b):
    assert_allclose(convert(a), convert(b))


@_dispatch(tuple, tuple)
def allclose(a, b):
    if len(a) != len(b):
        raise AssertionError('Inputs "{}" and "{}" are not of the same length.'
                             ''.format(a, b))
    for x, y in zip(a, b):
        assert_allclose(convert(x), convert(y))


@_dispatch(object, object, [object])
def approx(a, b, digits=4):
    assert_array_almost_equal(convert(a), convert(b), decimal=digits)


@_dispatch(tuple, tuple, [object])
def approx(a, b, digits=4):
    if len(a) != len(b):
        raise AssertionError('Inputs "{}" and "{}" are not of the same length.'
                             ''.format(a, b))
    for x, y in zip(a, b):
        approx(x, y, digits=digits)


def tensor(x):
    """Construct a PyTorch tensor of type `torch.float64`.

    Args:
        x (obj): Object to construct array from.

    Returns:
        tensor: PyTorch array of type `torch.float64`.

    """
    return torch.tensor(x, dtype=torch.float64)
