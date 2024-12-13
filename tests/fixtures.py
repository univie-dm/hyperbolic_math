"""This file contains global fixtures that are used across all our tests."""

import random
from typing import Union

import numpy as np
import pytest
import torch

from src.manifolds import Euclidean, Hyperboloid, PoincareBall


@pytest.fixture(scope="package")
def seed() -> int:
    """Set the seed(s) for all tests."""
    # Make it for one seed (14) only now
    # seed = request.param
    seed = 14
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed


@pytest.fixture(scope="package", params=[Euclidean, PoincareBall], ids=["Euclidean", "PoincareBall"])
def manifold(request: pytest.FixtureRequest) -> Union[Euclidean, PoincareBall, Hyperboloid]:
    """Instantiate the manifold(s)."""
    if request.param == Euclidean:
        manifold = Euclidean()
    elif request.param == PoincareBall:
        manifold = PoincareBall()
    elif request.param == Hyperboloid:
        manifold = Hyperboloid()
    return manifold
