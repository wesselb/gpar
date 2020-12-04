import numpy as np
import pytest
from lab.torch import B
from stheno import (
    GP,
    EQ,
    Delta,
    Measure,
    Obs,
    SparseObs,
    ZeroKernel,
    Linear,
    WeightedUnique,
)

from gpar.model import merge, construct_model, last, per_output, GPAR
from .util import approx, all_different


@pytest.fixture(params=[1, 2])
def x(request):
    # In this module, weights are always rank-two tensors.
    d = request.param
    return B.randn(10, d)


@pytest.fixture()
def w():
    # In this module, weights are always required.
    return B.rand(10, 2) + 1e-2


def test_merge():
    original = np.array([1, 2, 3, 4])
    updates = np.array([5, 6])

    result = merge(original, updates, np.array([True, True, False, False]))
    approx(result, [5, 6, 3, 4])

    result = merge(original, updates, np.array([True, False, True, False]))
    approx(result, [5, 2, 6, 4])


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


@pytest.mark.parametrize("i", [0, 1])
def test_per_output(i):
    def per_output_i(y_, **kw_args):
        for yi, wi, mask in per_output(y_, y_, **kw_args):
            if i == 0:
                # We are testing `yi`. Squeeze it to make testing consistent with `wi`.
                assert B.rank(yi) == 2
                yield yi[:, 0], mask
            elif i == 1:
                # We are testing `wi`.
                assert B.rank(wi) == 1
                yield wi, mask
            else:
                raise RuntimeError(f'Invalid value {i} for "i".')

    y = np.array(
        [
            [1, 2, np.nan, np.nan],
            [3, np.nan, 4, np.nan],
            [5, 6, 7, np.nan],
            [8, np.nan, np.nan, np.nan],
            [9, 10, np.nan, np.nan],
            [11, np.nan, np.nan, 12],
        ]
    )

    expected = [
        ([1, 3, 5, 8, 9, 11], [True, True, True, True, True, True]),
        ([2, 6, 10], [True, False, True, False, True, False]),
        ([7], [False, True, False]),
        ([], [False]),
    ]
    result = [(a.tolist(), b.tolist()) for a, b in per_output_i(y, keep=False)]
    assert result == expected

    expected = [
        ([1, 3, 5, 8, 9, 11], [True, True, True, True, True, True]),
        ([2, None, 6, 10, None], [True, True, True, False, True, True]),
        ([4, 7, None], [False, True, True, False, True]),
        ([12], [False, False, True]),
    ]
    result = [
        ([None if np.isnan(c) else c for c in a], B.to_numpy(b).tolist())
        for a, b in per_output_i(y, keep=True)
    ]
    assert result == expected


def test_per_output_caching():
    assert list(per_output({True: [2, 3], False: [3, 4]}, None, keep=True)) == [2, 3]
    assert list(per_output({True: [2, 3], False: [4]}, None, keep=False)) == [4]


def test_misc():
    gpar = GPAR(x_ind=None)
    assert not gpar.sparse
    assert gpar.x_ind is None

    gpar = GPAR(x_ind=1)
    assert gpar.sparse
    assert gpar.x_ind == 1


def test_obs(x):
    prior = Measure()
    f = GP(EQ(), measure=prior)
    e = GP(1e-1 * Delta(), measure=prior)

    # Generate some data.
    w = B.rand(B.shape(x)[0]) + 1e-2
    y = f(x).sample()

    # Set some observations to be missing.
    y_missing = y.copy()
    y_missing[::2] = np.nan

    # Check dense case.
    gpar = GPAR()
    obs = gpar._obs(x, None, y_missing, w, f, e)
    assert isinstance(obs, Obs)
    approx(
        prior.logpdf(obs),
        (f + e)(WeightedUnique(x[1::2], w[1::2])).logpdf(y[1::2]),
        atol=1e-6,
    )

    # Check sparse case.
    gpar = GPAR(x_ind=x)
    obs = gpar._obs(x, x, y_missing, w, f, e)
    assert isinstance(obs, SparseObs)
    approx(
        prior.logpdf(obs),
        (f + e)(WeightedUnique(x[1::2], w[1::2])).logpdf(y[1::2]),
        atol=1e-6,
    )


def test_update_inputs():
    prior = Measure()
    f = GP(EQ(), measure=prior)

    x = np.array([[1], [2], [3]])
    y = np.array([[4], [5], [6]], dtype=float)
    res = B.concat(x, y, axis=1)
    x_ind = np.array([[6], [7]])
    res_ind = np.array([[6, 0], [7, 0]])

    # Check vanilla case.
    gpar = GPAR(x_ind=x_ind)
    approx(gpar._update_inputs(x, x_ind, y, f, None), (res, res_ind))

    # Check imputation with prior.
    gpar = GPAR(impute=True, x_ind=x_ind)
    this_y = y.copy()
    this_y[1] = np.nan
    this_res = res.copy()
    this_res[1, 1] = 0
    approx(gpar._update_inputs(x, x_ind, this_y, f, None), (this_res, res_ind))

    # Check replacing with prior.
    gpar = GPAR(replace=True, x_ind=x_ind)
    this_y = y.copy()
    this_y[1] = np.nan
    this_res = res.copy()
    this_res[0, 1] = 0
    this_res[1, 1] = np.nan
    this_res[2, 1] = 0
    approx(gpar._update_inputs(x, x_ind, this_y, f, None), (this_res, res_ind))

    # Check imputation and replacing with prior.
    gpar = GPAR(impute=True, replace=True, x_ind=x_ind)
    this_res = res.copy()
    this_res[:, 1] = 0
    approx(gpar._update_inputs(x, x_ind, y, f, None), (this_res, res_ind))

    # Construct observations and update result for inducing points.
    obs = Obs(f(np.array([1, 2, 3, 6, 7])), np.array([9, 10, 11, 12, 13]))
    res_ind = np.array([[6, 12], [7, 13]])

    # Check imputation with posterior.
    gpar = GPAR(impute=True, x_ind=x_ind)
    this_y = y.copy()
    this_y[1] = np.nan
    this_res = res.copy()
    this_res[1, 1] = 10
    approx(gpar._update_inputs(x, x_ind, this_y, f, obs), (this_res, res_ind))

    # Check replacing with posterior.
    gpar = GPAR(replace=True, x_ind=x_ind)
    this_y = y.copy()
    this_y[1] = np.nan
    this_res = res.copy()
    this_res[0, 1] = 9
    this_res[1, 1] = np.nan
    this_res[2, 1] = 11
    approx(gpar._update_inputs(x, x_ind, this_y, f, obs), (this_res, res_ind))

    # Check imputation and replacing with posterior.
    gpar = GPAR(impute=True, replace=True, x_ind=x_ind)
    this_res = res.copy()
    this_res[0, 1] = 9
    this_res[1, 1] = 10
    this_res[2, 1] = 11
    approx(gpar._update_inputs(x, x_ind, y, f, obs), (this_res, res_ind))


def test_conditioning(x, w):
    prior = Measure()
    f1, e1 = GP(EQ(), measure=prior), GP(1e-10 * Delta(), measure=prior)
    f2, e2 = GP(EQ(), measure=prior), GP(2e-10 * Delta(), measure=prior)
    gpar = GPAR().add_layer(lambda: (f1, e1)).add_layer(lambda: (f2, e2))

    # Generate some data.
    y = B.concat((f1 + e1)(x).sample(), (f2 + e2)(x).sample(), axis=1)

    # Extract posterior processes.
    gpar = gpar | (x, y, w)
    f1_post, e1_post = gpar.layers[0]()
    f2_post, e2_post = gpar.layers[1]()

    # Test independence of noises.
    assert f1_post.measure.kernels[f1_post, e1_post] == ZeroKernel()
    assert f2_post.measure.kernels[f2_post, e2_post] == ZeroKernel()

    # Test form of noises.
    assert e1.mean == e1_post.mean
    assert e1.kernel == e1_post.kernel
    assert e2.mean == e2_post.mean
    assert e2.kernel == e2_post.kernel

    # Test posteriors.
    approx(f1_post.mean(x), y[:, 0:1], atol=1e-3)
    approx(f2_post.mean(B.concat(x, y[:, 0:1], axis=1)), y[:, 1:2], atol=1e-3)


def test_logpdf(x, w):
    prior = Measure()
    f1, e1 = GP(EQ(), measure=prior), GP(2e-1 * Delta(), measure=prior)
    f2, e2 = GP(Linear(), measure=prior), GP(1e-1 * Delta(), measure=prior)
    gpar = GPAR().add_layer(lambda: (f1, e1)).add_layer(lambda: (f2, e2))

    # Generate some data.
    y = gpar.sample(x, w, latent=True)

    # Compute logpdf.
    x1 = WeightedUnique(x, w[:, 0])
    x2 = WeightedUnique(B.concat(x, y[:, 0:1], axis=1), w[:, 1])
    logpdf1 = (f1 + e1)(x1).logpdf(y[:, 0])
    logpdf2 = (f2 + e2)(x2).logpdf(y[:, 1])

    # Test computation of GPAR.
    assert gpar.logpdf(x, y, w) == logpdf1 + logpdf2
    assert gpar.logpdf(x, y, w, only_last_layer=True) == logpdf2

    # Test resuming computation.
    x_partial, x_ind_partial = gpar.logpdf(x, y, w, return_inputs=True, outputs=[0])
    assert gpar.logpdf(x_partial, y, w, x_ind=x_ind_partial, outputs=[1]) == logpdf2

    # Test that sampling missing gives a stochastic estimate.
    y[1, 0] = np.nan
    all_different(
        gpar.logpdf(x, y, w, sample_missing=True),
        gpar.logpdf(x, y, w, sample_missing=True),
    )


def test_sample(x, w):
    prior = Measure()

    # Test that it produces random samples.
    f1, e1 = GP(EQ(), measure=prior), GP(1e-1 * Delta(), measure=prior)
    f2, e2 = GP(EQ(), measure=prior), GP(2e-1 * Delta(), measure=prior)
    gpar = GPAR().add_layer(lambda: (f1, e1)).add_layer(lambda: (f2, e2))
    all_different(gpar.sample(x, w), gpar.sample(x, w))
    all_different(gpar.sample(x, w, latent=True), gpar.sample(x, w, latent=True))

    # Test that posterior latent samples are around the data that is conditioned on.
    prior = Measure()
    f1, e1 = GP(EQ(), measure=prior), GP(1e-10 * Delta(), measure=prior)
    f2, e2 = GP(EQ(), measure=prior), GP(2e-10 * Delta(), measure=prior)
    gpar = GPAR().add_layer(lambda: (f1, e1)).add_layer(lambda: (f2, e2))
    y = gpar.sample(x, w, latent=True)
    gpar = gpar | (x, y, w)
    approx(gpar.sample(x, w), y, atol=1e-3)
    approx(gpar.sample(x, w, latent=True), y, atol=1e-3)
