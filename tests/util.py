# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import sys

import torch
from lab.torch import B
from nose.tools import assert_raises, assert_equal, assert_less, \
    assert_less_equal, assert_not_equal, assert_greater, \
    assert_greater_equal, ok_
from numpy.testing import assert_allclose, assert_array_almost_equal
from plum import Dispatcher

_dispatch = Dispatcher()

le = assert_less_equal
lt = assert_less
eq = assert_equal
neq = assert_not_equal
ge = assert_greater_equal
gt = assert_greater
raises = assert_raises
ok = ok_


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


def call(f, method, args=(), res=True):
    assert_equal(getattr(f, method)(*args), res)


def lam(f, args=()):
    ok_(f(*args), 'Lambda returned False.')


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def array(x):
    """Construct a PyTorch array of type `torch.float64`.

    Args:
        x (obj): Object to construct array from.

    Returns:
        tensor: PyTorch array of type `torch.float64`.

    """
    return B.array(x, dtype=torch.float64)
