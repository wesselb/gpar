# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
import pytest
import torch
from lab.torch import B
from stheno import GP

from gpar.regression import (
    _vector_from_init,
    log_transform,
    squishing_transform,
    GPARRegressor,
    _construct_gpar,
    _determine_indices
)
from .util import allclose, approx, tensor


def test_transforms():
    f, f_inv = log_transform
    allclose(f(f_inv(tensor([1, 2, 3, 4]))), tensor([1, 2, 3, 4]))
    f, f_inv = squishing_transform
    allclose(f(f_inv(tensor([-2, -1, 3, 4]))), tensor([-2, -1, 3, 4]))


def test_vector_from_init():
    allclose(_vector_from_init(2, 2), np.array([2, 2]))
    allclose(_vector_from_init(np.array([1, 2, 3]), 2), np.array([1, 2]))
    with pytest.raises(ValueError):
        _vector_from_init(np.random.randn(2, 2), 1)
    with pytest.raises(ValueError):
        _vector_from_init(np.array([1, 2]), 3)


def test_determine_indices():
    # No Markov structure.
    assert _determine_indices(1, 0, None) == ([0], [], 0)
    assert _determine_indices(1, 1, None) == ([0], [1], 1)
    assert _determine_indices(1, 2, None) == ([0], [1, 2], 2)
    assert _determine_indices(2, 0, None) == ([0, 1], [], 0)
    assert _determine_indices(2, 1, None) == ([0, 1], [2], 1)
    assert _determine_indices(2, 2, None) == ([0, 1], [2, 3], 2)

    # Markov order: 0.
    assert _determine_indices(1, 0, 0) == ([0], [], 0)
    assert _determine_indices(1, 1, 0) == ([0], [], 0)
    assert _determine_indices(1, 2, 0) == ([0], [], 0)
    assert _determine_indices(2, 0, 0) == ([0, 1], [], 0)
    assert _determine_indices(2, 1, 0) == ([0, 1], [], 0)
    assert _determine_indices(2, 2, 0) == ([0, 1], [], 0)

    # Markov order: 1.
    assert _determine_indices(1, 0, 1) == ([0], [], 0)
    assert _determine_indices(1, 1, 1) == ([0], [1], 1)
    assert _determine_indices(1, 2, 1) == ([0], [2], 1)
    assert _determine_indices(2, 0, 1) == ([0, 1], [], 0)
    assert _determine_indices(2, 1, 1) == ([0, 1], [2], 1)
    assert _determine_indices(2, 2, 1) == ([0, 1], [3], 1)

    # Markov order: 2.
    assert _determine_indices(1, 0, 2) == ([0], [], 0)
    assert _determine_indices(1, 1, 2) == ([0], [1], 1)
    assert _determine_indices(1, 2, 2) == ([0], [1, 2], 2)
    assert _determine_indices(2, 0, 2) == ([0, 1], [], 0)
    assert _determine_indices(2, 1, 2) == ([0, 1], [2], 1)
    assert _determine_indices(2, 2, 2) == ([0, 1], [2, 3], 2)


def test_get_variables():
    gpar = GPARRegressor()
    gpar.vs.get(init=1.0, name='variable')
    assert list(gpar.get_variables().items()) == [('variable', 1.0)]


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
    logpdf1 = (f1 + e1)(tensor(x)).logpdf(tensor(y[:, 0]))
    x_stack = np.concatenate([x[:, None], y[:, 0:1]], axis=1)
    logpdf2 = (f2 + e2)(tensor(x_stack)).logpdf(tensor(y[:, 1]))
    approx(reg.logpdf(x, y), logpdf1 + logpdf2, digits=6)

    # Test computation under posterior.
    e1_post = GP(e1.kernel, e1.mean, graph=e1.graph)
    e2_post = GP(e2.kernel, e2.mean, graph=e2.graph)
    f1_post = f1 | ((f1 + e1)(tensor(x)), tensor(y[:, 0]))
    f2_post = f2 | ((f2 + e2)(tensor(x_stack)), tensor(y[:, 1]))
    logpdf1 = (f1_post + e1_post)(tensor(x)).logpdf(tensor(y[:, 0]))
    logpdf2 = (f2_post + e2_post)(tensor(x_stack)).logpdf(tensor(y[:, 1]))
    with pytest.raises(RuntimeError):
        reg.logpdf(x, y, posterior=True)
    reg.fit(x, y, iters=0)
    approx(reg.logpdf(x, y, posterior=True), logpdf1 + logpdf2, digits=6)

    # Test that sampling missing gives a stochastic estimate.
    y[::2, 0] = np.nan
    assert np.abs(reg.logpdf(x, y, sample_missing=True) -
                  reg.logpdf(x, y, sample_missing=True)) >= 1e-3


def test_sample_and_predict():
    reg = GPARRegressor(replace=False, impute=False,
                        linear=True, linear_scale=1., nonlinear=False,
                        noise=1e-8, normalise_y=False)
    x = np.linspace(0, 5, 10)

    # Test checks.
    with pytest.raises(ValueError):
        reg.sample(x)
    with pytest.raises(RuntimeError):
        reg.sample(x, posterior=True)

    # Test that output is simplified correctly.
    assert isinstance(reg.sample(x, p=2), np.ndarray)
    assert isinstance(reg.sample(x, p=2, num_samples=2), list)

    # Test that it produces random samples. Not sure how to test correctness.
    assert np.sum(np.abs(reg.sample(x, p=2) - reg.sample(x, p=2))) >= 1e-2
    assert np.sum(np.abs(reg.sample(x, p=2, latent=True) -
                         reg.sample(x, p=2, latent=True))) >= 1e-3

    # Test that mean of posterior samples are around the data.
    y = reg.sample(x, p=2)
    reg.fit(x, y, iters=0)
    approx(y, np.mean(reg.sample(x,
                                 posterior=True,
                                 num_samples=20), axis=0), digits=4)
    approx(y, np.mean(reg.sample(x,
                                 latent=True,
                                 posterior=True,
                                 num_samples=20), axis=0), digits=4)

    # Test that prediction is around the data.
    approx(y, reg.predict(x, num_samples=20), digits=4)
    approx(y, reg.predict(x, latent=True, num_samples=20), digits=4)

    # Test that prediction is confident.
    _, lowers, uppers = reg.predict(x, num_samples=10, credible_bounds=True)
    assert np.less_equal(uppers - lowers, 1e-3).all()


def test_fit():
    reg = GPARRegressor(replace=False, impute=False,
                        normalise_y=True, transform_y=squishing_transform)
    x = np.linspace(0, 5, 10)
    y = reg.sample(x, p=2)

    # TODO: Remove this once greedy search is implemented.
    with pytest.raises(NotImplementedError):
        reg.fit(x, y, greedy=True)

    # Test that data is correctly transformed if it has an output with zero
    # variance.
    reg.fit(x, y, iters=0)
    assert (~B.isnan(reg.y)).numpy().all()
    y_pathological = y.copy()
    y_pathological[:, 0] = 1
    reg.fit(x, y_pathological, iters=0)
    assert (~B.isnan(reg.y)).numpy().all()

    # Test transformation and normalisation of outputs.
    z = torch.linspace(-1, 1, 10, dtype=torch.float64)
    z = B.stack(z, 2 * z, axis=1)
    allclose(reg._untransform_y(reg._transform_y(z)), z)
    allclose(reg._unnormalise_y(reg._normalise_y(z)), z)

    # Test that fitting runs without issues.
    vs = reg.vs.copy(detach=True)
    reg.fit(x, y, fix=False)
    reg.vs = vs
    reg.fit(x, y, fix=True)


def test_features():
    # Test that optimisation runs for a full-fledged GPAR.
    reg = GPARRegressor(replace=True, scale=1.0,
                        per=True, per_period=1.0, per_decay=10.0,
                        input_linear=True, input_linear_scale=0.1,
                        linear=True, linear_scale=1.0,
                        nonlinear=True, nonlinear_scale=1.0,
                        rq=True, noise=0.1)
    x = np.stack([np.linspace(0, 10, 20),
                  np.linspace(10, 20, 20)], axis=1)
    y = reg.sample(x, p=2)
    reg.fit(x, y, iters=10)


def test_scale_tying():
    reg = GPARRegressor(scale_tie=True)
    reg.sample(np.linspace(0, 10, 20), p=2)  # Instantiate variables.
    vs = reg.get_variables()
    assert '0/input/scales' in vs
    assert '1/input/scales' not in vs


def test_inducing_points_uprank():
    reg = GPARRegressor(x_ind=np.linspace(0, 10, 20))
    assert reg.x_ind is not None
    assert B.rank(reg.x_ind) == 2
