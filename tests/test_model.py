# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
from lab.torch import B
from stheno import GP, EQ, Delta, Graph, Obs, SparseObs, ZeroKernel, Linear

from gpar.model import merge, construct_model, last, per_output, GPAR
from .util import allclose, approx, tensor


def test_merge():
    original = torch.tensor([1, 2, 3, 4])
    updates = torch.tensor([5, 6])

    result = merge(original, updates, torch.tensor([True, True, False, False]))
    allclose(result, [5, 6, 3, 4])

    result = merge(original, updates, torch.tensor([True, False, True, False]))
    allclose(result, [5, 2, 6, 4])


def test_construct_model():
    model = construct_model(1, 2)
    assert model() == (1, 2)


def test_last():
    xs = [1, 2, 3, 4]
    assert list(last(xs)) == [(False, 1), (False, 2), (False, 3), (True, 4)]
    assert list(last(xs, [1, 2])) == [(False, 2), (False, 3)]
    assert list(last(xs, [0, 3])) == [(False, 1), (True, 4)]
    assert list(last([])) == []
    assert list(last([], [0, 1])) == []


def test_per_output():
    y = torch.tensor([[1, 2, np.nan, np.nan],
                      [3, np.nan, 4, np.nan],
                      [5, 6, 7, np.nan],
                      [8, np.nan, np.nan, np.nan],
                      [9, 10, np.nan, np.nan],
                      [11, np.nan, np.nan, 12]])

    expected = [([1, 3, 5, 8, 9, 11], [True, True, True, True, True, True]),
                ([2, 6, 10], [True, False, True, False, True, False]),
                ([7], [False, True, False]),
                ([], [False])]
    result = [(a.numpy()[:, 0].tolist(), b.numpy().tolist())
              for a, b in per_output(y, keep=False)]
    assert result == expected

    expected = [([1, 3, 5, 8, 9, 11], [True, True, True, True, True, True]),
                ([2, -1, 6, 10, -1], [True, True, True, False, True, True]),
                ([4, 7, -1], [False, True, True, False, True]),
                ([12], [False, False, True])]
    result = [([-1 if np.isnan(c) else c for c in a.numpy()[:, 0].tolist()],
               b.numpy().tolist())
              for a, b in per_output(y, keep=True)]
    assert result == expected

    # Test caching.
    assert list(per_output({True: [2, 3], False: [3, 4]}, keep=True)) == [2, 3]
    assert list(per_output({True: [2, 3], False: [4]}, keep=False)) == [4]


def test_misc():
    gpar = GPAR(x_ind=None)
    assert gpar.sparse == False
    assert gpar.x_ind is None

    gpar = GPAR(x_ind=1)
    assert gpar.sparse == True
    assert gpar.x_ind == 1


def test_obs():
    graph = Graph()
    f = GP(EQ(), graph=graph)
    e = GP(1e-8 * Delta(), graph=graph)

    # Check that it produces the correct observations.
    x = torch.linspace(0, 0.1, 10, dtype=torch.float64)
    y = f(x).sample()

    # Set some observations to be missing.
    y_missing = y.clone()
    y_missing[::2] = np.nan

    # Check dense case.
    gpar = GPAR()
    obs = gpar._obs(x, None, y_missing, f, e)
    assert type(obs) == Obs
    approx(y, (f | obs).mean(x))

    # Check sparse case.
    gpar = GPAR(x_ind=x)
    obs = gpar._obs(x, x, y_missing, f, e)
    assert type(obs) == SparseObs
    approx(y, (f | obs).mean(x))


def test_update_inputs():
    graph = Graph()
    f = GP(EQ(), graph=graph)

    x = tensor([[1], [2], [3]])
    y = tensor([[4], [5], [6]])
    res = B.concat(x, y, axis=1)
    x_ind = tensor([[6], [7]])
    res_ind = tensor([[6, 0], [7, 0]])

    # Check vanilla case.
    gpar = GPAR(x_ind=x_ind)
    allclose(gpar._update_inputs(x, x_ind, y, f, None), (res, res_ind))

    # Check imputation with prior.
    gpar = GPAR(impute=True, x_ind=x_ind)
    this_y = y.clone()
    this_y[1] = np.nan
    this_res = res.clone()
    this_res[1, 1] = 0
    allclose(gpar._update_inputs(x, x_ind, this_y, f, None),
             (this_res, res_ind))

    # Check replacing with prior.
    gpar = GPAR(replace=True, x_ind=x_ind)
    this_y = y.clone()
    this_y[1] = np.nan
    this_res = res.clone()
    this_res[0, 1] = 0
    this_res[1, 1] = np.nan
    this_res[2, 1] = 0
    allclose(gpar._update_inputs(x, x_ind, this_y, f, None),
             (this_res, res_ind))

    # Check imputation and replacing with prior.
    gpar = GPAR(impute=True, replace=True, x_ind=x_ind)
    this_res = res.clone()
    this_res[:, 1] = 0
    allclose(gpar._update_inputs(x, x_ind, y, f, None), (this_res, res_ind))

    # Construct observations and update result for inducing points.
    obs = Obs(f(tensor([1, 2, 3, 6, 7])), tensor([9, 10, 11, 12, 13]))
    res_ind = tensor([[6, 12], [7, 13]])

    # Check imputation with posterior.
    gpar = GPAR(impute=True, x_ind=x_ind)
    this_y = y.clone()
    this_y[1] = np.nan
    this_res = res.clone()
    this_res[1, 1] = 10
    allclose(gpar._update_inputs(x, x_ind, this_y, f, obs), (this_res, res_ind))

    # Check replacing with posterior.
    gpar = GPAR(replace=True, x_ind=x_ind)
    this_y = y.clone()
    this_y[1] = np.nan
    this_res = res.clone()
    this_res[0, 1] = 9
    this_res[1, 1] = np.nan
    this_res[2, 1] = 11
    allclose(gpar._update_inputs(x, x_ind, this_y, f, obs), (this_res, res_ind))

    # Check imputation and replacing with posterior.
    gpar = GPAR(impute=True, replace=True, x_ind=x_ind)
    this_res = res.clone()
    this_res[0, 1] = 9
    this_res[1, 1] = 10
    this_res[2, 1] = 11
    allclose(gpar._update_inputs(x, x_ind, y, f, obs), (this_res, res_ind))


def test_conditioning():
    graph = Graph()
    f1, e1 = GP(EQ(), graph=graph), GP(1e-8 * Delta(), graph=graph)
    f2, e2 = GP(EQ(), graph=graph), GP(2e-8 * Delta(), graph=graph)
    gpar = GPAR().add_layer(lambda: (f1, e1)).add_layer(lambda: (f2, e2))

    x = tensor([[1], [2], [3]])
    y = tensor([[4, 5],
                [6, 7],
                [8, 9]])
    gpar = gpar | (x, y)

    # Extract posterior processes.
    f1_post, e1_post = gpar.layers[0]()
    f2_post, e2_post = gpar.layers[1]()

    # Test independence of noises.
    assert graph.kernels[f1_post, e1_post] == ZeroKernel()
    assert graph.kernels[f2_post, e2_post] == ZeroKernel()

    # Test form of noises.
    assert e1.mean == e1_post.mean
    assert e1.kernel == e1_post.kernel
    assert e2.mean == e2_post.mean
    assert e2.kernel == e2_post.kernel

    # Test posteriors.
    approx(f1_post.mean(x), y[:, 0:1])
    approx(f2_post.mean(B.concat(x, y[:, 0:1], axis=1)), y[:, 1:2])


def test_logpdf():
    graph = Graph()
    f1, e1 = GP(EQ(), graph=graph), GP(2e-1 * Delta(), graph=graph)
    f2, e2 = GP(Linear(), graph=graph), GP(1e-1 * Delta(), graph=graph)
    gpar = GPAR().add_layer(lambda: (f1, e1)).add_layer(lambda: (f2, e2))

    # Sample some data from GPAR.
    x = torch.linspace(0, 2, 10, dtype=torch.float64)[:, None]
    y = gpar.sample(x, latent=True)

    # Compute logpdf.
    logpdf1 = (f1 + e1)(x).logpdf(y[:, 0])
    logpdf2 = (f2 + e2)(B.concat(x, y[:, 0:1], axis=1)).logpdf(y[:, 1])

    # Test computation of GPAR.
    assert gpar.logpdf(x, y) == logpdf1 + logpdf2
    assert gpar.logpdf(x, y, only_last_layer=True) == logpdf2

    # Test resuming computation.
    x_int, x_ind_int = gpar.logpdf(x, y, return_inputs=True, outputs=[0])
    assert gpar.logpdf(x_int, y, x_ind=x_ind_int, outputs=[1]) == logpdf2

    # Test that sampling missing gives a stochastic estimate.
    y[1, 0] = np.nan
    assert B.abs(gpar.logpdf(x, y, sample_missing=True) -
                 gpar.logpdf(x, y, sample_missing=True)).numpy() >= 1e-3


def test_sample():
    graph = Graph()
    x = tensor([1, 2, 3])[:, None]

    # Test that it produces random samples. Not sure how to test for
    # correctness.
    f1, e1 = GP(EQ(), graph=graph), GP(1e-1 * Delta(), graph=graph)
    f2, e2 = GP(EQ(), graph=graph), GP(1e-1 * Delta(), graph=graph)
    gpar = GPAR().add_layer(lambda: (f1, e1)).add_layer(lambda: (f2, e2))
    assert B.sum(B.abs(gpar.sample(x) - gpar.sample(x))) >= 1e-3
    assert B.sum(B.abs(gpar.sample(x, latent=True) -
                       gpar.sample(x, latent=True))) >= 1e-3

    # Test that posterior latent samples are around the data that is
    # conditioned on.
    graph = Graph()
    f1, e1 = GP(EQ(), graph=graph), GP(1e-8 * Delta(), graph=graph)
    f2, e2 = GP(EQ(), graph=graph), GP(1e-8 * Delta(), graph=graph)
    gpar = GPAR().add_layer(lambda: (f1, e1)).add_layer(lambda: (f2, e2))
    y = gpar.sample(x, latent=True)
    gpar = gpar | (x, y)
    approx(gpar.sample(x), y, digits=3)
    approx(gpar.sample(x, latent=True), y, digits=3)
