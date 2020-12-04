import numpy as np
import pytest
from lab.torch import B
from stheno import GP, WeightedUnique

from gpar.regression import (
    _vector_from_init,
    log_transform,
    squishing_transform,
    GPARRegressor,
    _construct_gpar,
    _determine_indices,
)
from .util import approx, tensor, all_different


@pytest.fixture(params=((10,), (10, 1), (10, 2)))
def x(request):
    shape = request.param
    return B.randn(*shape)


@pytest.fixture(params=([True, False]))
def w(request):
    use_w = request.param
    if use_w:
        return B.rand(10, 2) + 1
    else:
        return None


def test_log_transform():
    x = B.rand(5)
    f, f_inv = log_transform
    approx(f(f_inv(x)), x)


def test_squishing_transform():
    x = B.randn(5)
    f, f_inv = squishing_transform
    approx(f(f_inv(x)), x)


def test_vector_from_init():
    approx(_vector_from_init(2, 2), np.array([2, 2]))
    approx(_vector_from_init(np.array([1, 2, 3]), 2), np.array([1, 2]))
    with pytest.raises(ValueError):
        _vector_from_init(B.randn(2, 2), 1)
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
    gpar.vs.get(init=1.0, name="variable")
    assert list(gpar.get_variables().items()) == [("variable", 1.0)]


def test_logpdf(x, w):
    # Sample some data from a "sensitive" GPAR.
    reg = GPARRegressor(
        replace=False,
        impute=False,
        nonlinear=True,
        nonlinear_scale=0.1,
        linear=True,
        linear_scale=10.0,
        noise=1e-2,
        normalise_y=False,
    )
    y = reg.sample(x, w, p=2, latent=True)

    # Extract models.
    gpar = _construct_gpar(reg, reg.vs, B.shape(B.uprank(x))[1], 2)
    f1, e1 = gpar.layers[0]()
    f2, e2 = gpar.layers[1]()

    # Test computation under prior.
    x1 = x
    x2 = B.concat(B.uprank(x), y[:, 0:1], axis=1)
    if w is not None:
        x1 = WeightedUnique(x1, w[:, 0])
        x2 = WeightedUnique(x2, w[:, 1])
    logpdf1 = (f1 + e1)(x1).logpdf(y[:, 0])
    logpdf2 = (f2 + e2)(x2).logpdf(y[:, 1])
    approx(reg.logpdf(x, y, w), logpdf1 + logpdf2, atol=1e-6)

    # Test computation under posterior.
    post1 = f1.measure | ((f1 + e1)(x1), y[:, 0])
    post2 = f2.measure | ((f2 + e2)(x2), y[:, 1])
    e1_post = GP(e1.mean, e1.kernel, measure=post1)
    e2_post = GP(e2.mean, e2.kernel, measure=post2)
    logpdf1 = (post1(f1) + e1_post)(x1).logpdf(y[:, 0])
    logpdf2 = (post2(f2) + e2_post)(x2).logpdf(y[:, 1])
    with pytest.raises(RuntimeError):
        reg.logpdf(x, y, w, posterior=True)
    reg.condition(x, y, w)
    approx(reg.logpdf(x, y, w, posterior=True), logpdf1 + logpdf2, atol=1e-6)

    # Test that sampling missing gives a stochastic estimate.
    y[::2, 0] = np.nan
    all_different(
        reg.logpdf(x, y, w, sample_missing=True),
        reg.logpdf(x, y, w, sample_missing=True),
    )


def test_logpdf_differentiable(x, w):
    reg = GPARRegressor(
        replace=False,
        impute=False,
        linear=True,
        linear_scale=1.0,
        nonlinear=False,
        noise=1e-8,
        normalise_y=False,
    )
    y = reg.sample(x, w, p=2, latent=True)

    # Test that gradient calculation works.
    reg.vs.requires_grad(True)
    for var in reg.vs.get_vars():
        assert var.grad is None
    reg.logpdf(tensor(x), tensor(y)).backward()
    for var in reg.vs.get_vars():
        assert var.grad is not None


def test_sample_and_predict(x, w):
    # Use output transform to ensure that is handled correctly.
    reg = GPARRegressor(
        replace=False,
        impute=False,
        linear=True,
        linear_scale=1.0,
        nonlinear=False,
        noise=1e-8,
        normalise_y=False,
        transform_y=squishing_transform,
    )

    # Test checks.
    with pytest.raises(ValueError):
        reg.sample(x, w)
    with pytest.raises(RuntimeError):
        reg.sample(x, w, posterior=True)

    # Test that output is simplified correctly.
    assert isinstance(reg.sample(x, w, p=2), np.ndarray)
    assert isinstance(reg.sample(x, w, p=2, num_samples=2), list)

    # Test that it produces random samples. Not sure how to test correctness.
    all_different(reg.sample(x, w, p=2), reg.sample(x, w, p=2))
    all_different(
        reg.sample(x, w, p=2, latent=True), reg.sample(x, w, p=2, latent=True)
    )

    # Test that mean of posterior samples are around the data.
    y = reg.sample(x, w, p=2)
    reg.condition(x, y, w)
    approx(
        y, np.mean(reg.sample(x, w, posterior=True, num_samples=100), axis=0), atol=5e-2
    )
    approx(
        y,
        np.mean(reg.sample(x, w, latent=True, posterior=True, num_samples=100), axis=0),
        atol=5e-2,
    )

    # Test that prediction is around the data.
    approx(y, reg.predict(x, w, num_samples=100), atol=5e-2)
    approx(y, reg.predict(x, w, latent=True, num_samples=100), atol=5e-2)

    # Test that prediction is confident.
    _, lowers, uppers = reg.predict(x, w, num_samples=100, credible_bounds=True)
    approx(uppers, lowers, atol=5e-2)


def test_condition_and_fit(x, w):
    reg = GPARRegressor(
        replace=False, impute=False, normalise_y=True, transform_y=squishing_transform
    )
    y = reg.sample(x, w, p=2)

    # Test that data is correctly normalised.
    reg.condition(x, y, w)
    approx(B.mean(reg.y, axis=0), B.zeros(reg.p))
    approx(B.std(reg.y, axis=0), B.ones(reg.p))

    # Test that data is correctly normalised if it has an output with zero
    # variance.
    y_pathological = y.copy()
    y_pathological[:, 0] = 1
    reg.condition(x, y_pathological, w)
    assert B.all(~B.isnan(reg.y))

    # Test transformation and normalisation of outputs.
    z = B.linspace(-1, 1, 10)
    z = B.stack(z, 2 * z, axis=1)
    approx(reg._untransform_y(reg._transform_y(z)), z)
    approx(reg._unnormalise_y(reg._normalise_y(z)), z)

    # Test that fitting runs without issues.
    vs = reg.vs.copy(detach=True)
    reg.fit(x, y, w, fix=False)
    reg.vs = vs
    reg.fit(x, y, w, fix=True)

    # TODO: Remove this once greedy search is implemented.
    with pytest.raises(NotImplementedError):
        reg.fit(x, y, w, greedy=True)


def test_features():
    # Test that optimisation runs for a full-fledged GPAR.
    reg = GPARRegressor(
        replace=True,
        scale=1.0,
        per=True,
        per_period=1.0,
        per_decay=10.0,
        input_linear=True,
        input_linear_scale=0.1,
        linear=True,
        linear_scale=1.0,
        nonlinear=True,
        nonlinear_scale=1.0,
        rq=True,
        noise=0.1,
    )
    x = B.stack(B.linspace(0, 10, 20), B.linspace(10, 20, 20), axis=1)
    y = reg.sample(x, p=2)
    reg.fit(x, y, iters=10)


def test_scale_tying(x, w):
    reg = GPARRegressor(scale_tie=True)
    reg.sample(x, w, p=2)  # Instantiate variables.
    vs = reg.get_variables()
    assert "0/input/scales" in vs
    assert "1/input/scales" not in vs


def test_inducing_points_uprank():
    reg = GPARRegressor(x_ind=B.linspace(0, 10, 20))
    assert reg.x_ind is not None
    assert B.rank(reg.x_ind) == 2
