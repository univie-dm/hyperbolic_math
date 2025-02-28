import pytest
import torch

from typing import Tuple, Type
from src.manifolds import Manifold, ManifoldParameter
from src.optim import RiemannianAdam, RiemannianSGD


@pytest.mark.parametrize("expmap_update", [True, False])
def test_riemannian_adam(manifold: Type[Manifold], tolerance: Tuple[float, float],
                         uniform_points: torch.Tensor, expmap_update: bool) -> None:
    """Test the RiemannianAdam for convergence."""
    atol, _ = tolerance
    target = uniform_points[0, :]
    start = manifold.scalar_mul(0.9, target)
    start = ManifoldParameter(start, requires_grad=True, manifold=manifold)

    optim = RiemannianAdam([start], lr=1e-3, eps=1e-5, expmap_update=expmap_update)
    for _ in range(300_000):
        optim.zero_grad()
        loss = manifold.dist(start, target).pow(2).mean()
        if (start-target).norm(p=2) < atol:
            break
        loss.backward()
        optim.step()
    else:
        assert False, "RiemannianAdam did not converge!"

@pytest.mark.parametrize("expmap_update", [True, False])
def test_riemannian_sgd(manifold: Type[Manifold], tolerance: Tuple[float, float],
                        uniform_points: torch.Tensor, expmap_update: bool) -> None:
    """Test the RiemannianSGD for convergence."""
    atol, rtol = tolerance
    target = uniform_points[0, :]
    start = manifold.scalar_mul(0.3, target)
    start = ManifoldParameter(start, requires_grad=True, manifold=manifold)

    optim = RiemannianSGD([start], lr=1e-3, momentum=0.9, expmap_update=expmap_update)
    for _ in range(1000):
        optim.zero_grad()
        loss = manifold.dist(start, target).pow(2).mean()
        loss.backward()
        optim.step()
    torch.testing.assert_close(start.data, target, atol=atol, rtol=rtol)
