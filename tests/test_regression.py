# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
from gpar.regression import _uprank, _vector_from_init, log_transform, \
    squishing_transform, GPARRegressor, _construct_gpar
from lab.torch import B
from stheno import GP

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


def test_get_variables():
    gpar = GPARRegressor()
    gpar.vs.get(init=1.0, name='variable')
    yield eq, list(gpar.get_variables().items()), [('variable', 1.0)]


def test_cases():
    pass


def test_logpdf():
    # Sample some data from a "sensitive" GPAR.
    reg = GPARRegressor(replace=False, impute=False, normalise_y=False,
                        nonlinear=True, nonlinear_scale=0.1,
                        linear=True, linear_scale=10.,
                        noise=1e-4)
    x = np.linspace(0, 5, 10)
    y = reg.sample(x, p=2, latent=True)

    # Extract models.
    gpar = _construct_gpar(reg, reg.vs, 1, 2)
    f1, e1 = gpar.layers[0]()
    f2, e2 = gpar.layers[1]()

    # Test computation under prior.
    logpdf1 = (f1 + e1)(B.array(x)).logpdf(B.array(y[:, 0]))
    x_stack = np.concatenate([x[:, None], y[:, 0:1]], axis=1)
    logpdf2 = (f2 + e2)(B.array(x_stack)).logpdf(B.array(y[:, 1]))
    yield approx, reg.logpdf(x, y), logpdf1 + logpdf2, 6

    # Test computation under posterior.
    e1_post = GP(e1.kernel, e1.mean, graph=e1.graph)
    e2_post = GP(e2.kernel, e2.mean, graph=e2.graph)
    f1_post = f1 | ((f1 + e1)(B.array(x)), B.array(y[:, 0]))
    f2_post = f2 | ((f2 + e2)(B.array(x_stack)), B.array(y[:, 1]))
    logpdf1 = (f1_post + e1_post)(B.array(x)).logpdf(B.array(y[:, 0]))
    logpdf2 = (f2_post + e2_post)(B.array(x_stack)).logpdf(B.array(y[:, 1]))
    reg.is_fit = True
    reg.x = B.array(x[:, None])
    reg.y = B.array(y)
    yield approx, reg.logpdf(x, y, posterior=True), logpdf1 + logpdf2, 6

    # Test that sampling missing gives a stochastic estimate.
    y[::2, 0] = np.nan
    yield neq, \
          reg.logpdf(x, y, sample_missing=True), \
          reg.logpdf(x, y, sample_missing=True)
    