"""Math utils functions for hyperbolic operations with numerically stable limits."""

import math
import torch


def cosh(x: torch.Tensor) -> torch.Tensor:
    """Hyperbolic cosine. Domain=(-inf, inf)."""
    eps = torch.finfo(x.dtype).eps
    clamp = float(math.log(2 / eps))
    return x.clamp(-clamp, clamp).cosh()

def sinh(x: torch.Tensor) -> torch.Tensor:
    """Hyperbolic sine. Domain=(-inf, inf)."""
    eps = torch.finfo(x.dtype).eps
    clamp = float(math.log(2 / eps))
    return x.clamp(-clamp, clamp).sinh()

def tanh(x: torch.Tensor) -> torch.Tensor:
    """Hyperbolic tangent. Domain=(-inf, inf)."""
    eps = torch.finfo(x.dtype).eps
    clamp = float(-math.log(eps / 2) / 2)
    return x.clamp(-clamp, clamp).tanh()

def arcosh(x: torch.Tensor) -> torch.Tensor:
    """Inverse hyperbolic cosine. Domain=[1, inf)."""
    min_val = 1.0 + torch.finfo(x.dtype).eps
    return x.clamp(min=min_val).double().add(torch.sqrt(x.clamp(min=min_val).double().pow(2) - 1)).log().to(x.dtype)

def arsinh(x: torch.Tensor) -> torch.Tensor:
    """Inverse hyperbolic sine. Domain=(-inf, inf)."""
    return x.double().add(torch.sqrt(1 + x.double().pow(2))).log().to(x.dtype)

def artanh(x: torch.Tensor) -> torch.Tensor:
    """Inverse hyperbolic tangent. Domain=(-1, 1)."""
    eps = torch.finfo(x.dtype).eps
    x = x.clamp(-1 + eps, 1 - eps)
    return (torch.log1p(x.double()) - torch.log1p(-x.double())).mul(0.5).to(x.dtype)
