import lab as B
import torch
from numpy.testing import assert_allclose
from plum import Dispatcher

__all__ = ["approx", "all_different", "tensor"]

_dispatch = Dispatcher()


@_dispatch
def approx(a, b, rtol=1e-7, atol=1e-12):
    """Assert that two objects are approximately equal.

    Args:
        a (object): First object.
        b (object): Second object.
        rtol (:obj:`float`, optional): Relative tolerance. Defaults to `1e-7`.
        atol (:obj:`float`, optional): Absolute tolerance. Defaults to `1e-12`.
    """
    assert_allclose(B.to_numpy(a), B.to_numpy(b), rtol=rtol, atol=atol)


@_dispatch
def approx(a: tuple, b: tuple, **kw_args):
    if len(a) != len(b):
        raise AssertionError(f'Inputs "{a}" and "{b}" are not of the same length.')
    for x, y in zip(a, b):
        approx(x, y, **kw_args)


def all_different(x, y):
    """Assert that two matrices have all different columns.

    Args:
        x (matrix): First matrix.
        y (matrix): Second matrix.
    """
    assert B.all(B.pw_dists(B.transpose(x), B.transpose(y)) > 1e-2)


def tensor(x):
    """Construct a PyTorch tensor of data type `torch.float64`.

    Args:
        x (object): Object to construct array from.

    Returns:
        tensor: PyTorch array of data type `torch.float64`.
    """
    return torch.tensor(x, dtype=torch.float64)
