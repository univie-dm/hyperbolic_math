"""This file contains global fixtures that are used across all our tests."""

import pytest
import torch

from typing import Tuple
from src.manifolds import Manifold, Euclidean, Hyperboloid, PoincareBall


@pytest.fixture(scope="package", params=[*range(10, 13)])
def seed(request: pytest.FixtureRequest) -> None:
    """Global seed for reproducibility."""
    torch.manual_seed(request.param)

#@pytest.fixture(scope="package", params=[torch.float32, torch.float64], ids=["float32", "float64"])
@pytest.fixture(scope="package", params=[torch.float64], ids=["float64"])
def dtype(request: pytest.FixtureRequest) -> torch.dtype:
    """Test different data types."""
    return request.param

@pytest.fixture(scope="package")
def tolerance(dtype: torch.dtype) -> Tuple[float, float]:
    """Set numerical tolerances for floating point comparisons."""
    if dtype == torch.float32:
        atol = torch.finfo(dtype).eps
        rtol = torch.finfo(dtype).eps
    else:   # float64
        atol = 1e-13
        rtol = 1e-09
    return atol, rtol

#@pytest.fixture(scope="package", params=[Euclidean, Hyperboloid, PoincareBall], ids=["Euclidean", "Hyperboloid", "PoincareBall"])
@pytest.fixture(scope="package", params=[Euclidean, PoincareBall], ids=["Euclidean", "PoincareBall"])
def manifold(seed: None, dtype: torch.dtype, request: pytest.FixtureRequest) -> Manifold:
    """Test different manifolds and curvatures."""
    c = torch.empty(1, dtype=dtype).exponential_(0.5)
    return request.param(c=c, dtype=dtype)

@pytest.fixture(scope="package", params=[2, 5, 10, 15])
def uniform_points(seed: None, dtype: torch.dtype, manifold: Manifold,
                   request: pytest.FixtureRequest) -> torch.Tensor:
    """Helper to generate uniformly distributed points for each manifold type."""
    dim = request.param
    num_pts = 2_500 * 6
    if isinstance(manifold, Euclidean):
        bound = 100
        points = torch.empty((num_pts, dim), dtype=dtype).uniform_(-bound, bound)
    elif isinstance(manifold, Hyperboloid):
        assert False, "Not implemented yet"
    else:   # PoincareBall
        random_dirs = torch.normal(0, 1, size=(num_pts, dim), dtype=dtype)
        random_dirs /= random_dirs.norm(p=2, dim=-1, keepdim=True)
        random_radii = torch.rand((num_pts, 1), dtype=dtype).pow(1 / dim)
        points = manifold.c**-0.5 * (random_dirs * random_radii)
    # Check if the points are in the manifold
    assert manifold.is_in_manifold(points), "Points are not in manifold!"
    return points
