# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
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


def test_logpdf():
    # Sample some data from a "sensitive" GPAR.
    reg = GPARRegressor(replace=False, impute=False,
                        nonlinear=True, nonlinear_scale=0.1,
                        linear=True, linear_scale=10.,
                        noise=1e-4, normalise_y=False)
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
    reg.fit(x, y, iters=0)
    yield approx, reg.logpdf(x, y, posterior=True), logpdf1 + logpdf2, 6

    # Test that sampling missing gives a stochastic estimate.
    y[::2, 0] = np.nan
    yield ge, \
          np.abs(reg.logpdf(x, y, sample_missing=True) -
                 reg.logpdf(x, y, sample_missing=True)), \
          1e-3


def test_sample_and_predict():
    reg = GPARRegressor(replace=False, impute=False,
                        linear=True, linear_scale=1., nonlinear=False,
                        noise=1e-8, normalise_y=False)
    x = np.linspace(0, 5, 10)

    # Test checks.
    yield raises, ValueError, lambda: reg.sample(x)
    yield raises, RuntimeError, lambda: reg.sample(x, posterior=True)

    # Test that output is simplified correctly.
    yield isinstance, reg.sample(x, p=2), np.ndarray
    yield isinstance, reg.sample(x, p=2, num_samples=2), list

    # Test that it produces random samples. Not sure how to test correctness.
    yield ge, np.sum(np.abs(reg.sample(x, p=2) - reg.sample(x, p=2))), 1e-2
    yield ge, np.sum(np.abs(reg.sample(x, p=2, latent=True) -
                            reg.sample(x, p=2, latent=True))), 1e-3

    # Test that mean of posterior samples are around the data.
    y = reg.sample(x, p=2)
    reg.fit(x, y, iters=0)
    yield approx, y, np.mean(reg.sample(x,
                                        posterior=True,
                                        num_samples=20), axis=0), 4
    yield approx, y, np.mean(reg.sample(x,
                                        latent=True,
                                        posterior=True,
                                        num_samples=20), axis=0), 4

    # Test that prediction is around the data.
    yield approx, y, reg.predict(x, num_samples=20), 4
    yield approx, y, reg.predict(x, latent=True, num_samples=20), 4

    # Test that prediction is confident.
    _, lowers, uppers = reg.predict(x, num_samples=10, credible_bounds=True)
    yield ok, np.less_equal(uppers - lowers, 1e-3).all()


def test_fit():
    reg = GPARRegressor(replace=False, impute=False,
                        normalise_y=True, transform_y=squishing_transform)
    x = np.linspace(0, 5, 10)
    y = reg.sample(x, p=2)

    # TODO: Remove this once greedy search is implemented.
    yield raises, NotImplementedError, lambda: reg.fit(x, y, greedy=True)

    # Test that data is correctly transformed if it has an output with zero
    # variance.
    reg.fit(x, y, iters=0)
    yield ok, (~B.isnan(reg.y)).numpy().all()
    y_pathological = y.copy()
    y_pathological[:, 0] = 1
    reg.fit(x, y_pathological, iters=0)
    yield ok, (~B.isnan(reg.y)).numpy().all()

    # Test transformation of outputs.
    z = B.linspace(-1, 1, 10, dtype=torch.float64)
    z = B.stack([z, 2 * z], axis=1)
    yield allclose, reg._untransform_y(reg._transform_y(z)), z

    # Test that fitting runs without issues.
    vs = reg.vs.detach()
    yield lam, lambda: reg.fit(x, y, fix=False) is None
    reg.vs = vs
    yield lam, lambda: reg.fit(x, y, fix=True) is None


def test_cases():
    pass
