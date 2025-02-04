"""This file contains global fixtures that are used across all our tests."""

import pytest
import random
import torch
import numpy as np

from typing import Union
from src.manifolds import Euclidean, Hyperboloid, PoincareBall

def get_test_configs(dtype):
    curvatures = [torch.tensor([0.5], dtype=dtype), 
                  torch.tensor([1.0], dtype=dtype),
                  torch.tensor([2.0], dtype=dtype)]
    
    configs = [(Euclidean, c) for c in curvatures]
    configs.extend([(PoincareBall, c) for c in curvatures])
    return configs

@pytest.fixture(scope="package", params=[14])
def seed(request):
    """Global seed for reproducibility"""
    torch.manual_seed(request.param)
    return request.param

@pytest.fixture(scope="package", params=[torch.float64])
def dtype(request):
    return request.param

@pytest.fixture(scope="package")
def tolerance(dtype):
    """Set numerical tolerances for floating point comparisons"""
    if dtype == torch.float32:
        atol = torch.finfo(dtype).eps
        rtol = torch.finfo(dtype).eps
    else:  # float64
        atol = torch.finfo(dtype).eps  
        rtol = 1e-10
    return atol, rtol

