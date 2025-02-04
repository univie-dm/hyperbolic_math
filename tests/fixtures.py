"""This file contains global fixtures that are used across all our tests."""

import pytest
import random
import torch
import numpy as np

from typing import Union
from src.manifolds import Euclidean, Hyperboloid, PoincareBall

#@pytest.fixture(scope="package", params=[range(10, 20)])
@pytest.fixture(scope="package", params=[14], ids=["seed=14"])
def seed(request: pytest.FixtureRequest) -> int:
    """Set the seed(s) for all tests."""
    seed = request.param
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed

#@pytest.fixture(scope="package", params=[torch.float32, torch.float64], ids=["float32", "float64"])
@pytest.fixture(scope="package", params=[torch.float64], ids=["float64"])
def dtype(request: pytest.FixtureRequest) -> torch.dtype:
    """Test different data types."""
    return request.param

@pytest.fixture(scope="package")
def c(dtype: torch.dtype) -> torch.Tensor:
    """Generate random curvature magnitude(s)."""
    return torch.empty(1, dtype=dtype).uniform_(torch.finfo(dtype).eps, 5)

#@pytest.fixture(scope="module", params=[Euclidean, Hyperboloid, PoincareBall], ids=["Euclidean", "Hyperboloid", "PoincareBall"])
@pytest.fixture(scope="module", params=[Euclidean, PoincareBall], ids=["Euclidean", "PoincareBall"])
def manifold(c: torch.Tensor, request: pytest.FixtureRequest) -> Union[Euclidean, PoincareBall, Hyperboloid]:
    """Instantiate the manifold(s)."""
    if request.param == Euclidean:
        manifold = Euclidean(c)
    elif request.param == PoincareBall:
        manifold = PoincareBall(c)
    elif request.param == Hyperboloid:
        manifold = Hyperboloid(c)
    return manifold
