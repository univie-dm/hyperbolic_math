"""Math utils functions for hyperbolic operations with numerically stable limits."""

import torch
import math


def cosh(x):
    eps = torch.finfo(x.dtype).eps
    clamp = float(math.log(2/eps))  # Calculate limit based on precision
    return x.clamp(-clamp, clamp).cosh()


def sinh(x):
    eps = torch.finfo(x.dtype).eps
    clamp = float(math.log(2/eps))
    return x.clamp(-clamp, clamp).sinh()

def tanh(x):
    eps = torch.finfo(x.dtype).eps
    clamp = float(-math.log(eps/2)/2)
    return x.clamp(-clamp, clamp).tanh()


def arcosh(x):
    """Inverse hyperbolic cosine."""
    # Domain starts at 1, upper bound depends on dtype max
    min_val = 1.0 + torch.finfo(x.dtype).eps
    return (x.clamp(min=min_val).double()
             .add(torch.sqrt(x.clamp(min=min_val).double().pow(2) - 1))
             .log()
             .to(x.dtype))


def arsinh(x):
    """Inverse hyperbolic sine."""
    # No domain restrictions, just dtype limits
    return (x.double()
             .add(torch.sqrt(1 + x.double().pow(2)))
             .log()
             .to(x.dtype))


def artanh(x):
    """Inverse hyperbolic tangent."""
    # Domain is strictly (-1, 1)
    eps = torch.finfo(x.dtype).eps
    x = x.clamp(-1 + eps, 1 - eps)
    return (torch.log1p(x.double()) - torch.log1p(-x.double())).mul(0.5).to(x.dtype)