from typing import Tuple, Union

import pytest
import torch

from src.manifolds import Euclidean, Hyperboloid, ManifoldParameter, PoincareBall
from src.optim import RiemannianAdam, RiemannianSGD

from .fixtures import manifold, seed








@pytest.fixture(scope="module")
def tolerance(dtype: torch.dtype) -> Tuple[float, float]:
    """Set the absolute and relative tolerance(s) for the tests."""
    if dtype == torch.float32:
        atol = 1e-5
        rtol = 1e-5
    elif dtype == torch.float64:
        # TODO: Can be updated?
        atol = 1e-5
        rtol = 1e-5
    return atol, rtol


@pytest.mark.parametrize("expmap_update", [True, False])
def test_riemannian_adam(
    seed: int,
    dtype: torch.dtype,
    manifold: Union[Euclidean, PoincareBall, Hyperboloid],
    c: torch.Tensor,
    tolerance: Tuple[float, float],
    expmap_update: bool,
):
    """Optimizer test: Fit a random starting point towards (0.5, 0.5)."""
    atol, rtol = tolerance
    ideal = torch.tensor([[0.5, 0.5]], dtype=dtype)
    start = torch.randn(2, dtype=dtype) / 2
    start = manifold.expmap_0(start, c=c)
    start = ManifoldParameter(start, manifold=manifold, requires_grad=True, c=c)

    def closure():
        optim.zero_grad()
        loss = manifold.dist(x=start, y=ideal, c=c) ** 2
        loss.backward()
        return loss.item()

    optim = RiemannianAdam([start], lr=1e-2, c=c, expmap_update=expmap_update, eps=1e-5)

    for _ in range(1100):
        optim.step(closure)
    torch.testing.assert_close(start.data, ideal, atol=atol, rtol=rtol)


@pytest.mark.parametrize("expmap_update", [True, False])
def test_riemannian_sgd(
    seed: int,
    dtype: torch.dtype,
    manifold: Union[Euclidean, PoincareBall, Hyperboloid],
    c: torch.Tensor,
    tolerance: Tuple[float, float],
    expmap_update: bool,
):
    """Optimizer test: Fit a random starting point towards (0.5, 0.5)."""
    atol, rtol = tolerance
    ideal = torch.tensor([0.5, 0.5], dtype=dtype)
    start = torch.randn(2, dtype=dtype) / 2
    start = manifold.expmap_0(start, c=c)
    start = ManifoldParameter(start, manifold=manifold, requires_grad=True, c=c)

    def closure():
        optim.zero_grad()
        loss = manifold.dist(x=start, y=ideal, c=c) ** 2
        loss.backward()
        return loss.item()

    optim = RiemannianSGD([start], lr=1e-2, c=c, expmap_update=expmap_update, momentum=0.9)

    for _ in range(1000):
        optim.step(closure)
    torch.testing.assert_close(start.data, ideal, atol=atol, rtol=rtol)
