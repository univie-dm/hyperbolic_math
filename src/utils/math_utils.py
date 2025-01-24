"""Math utils functions for hyperbolic operations with numerically stable limits."""

import math
import torch

@torch.jit.script
def _get_tensor_eps(
    x: torch.Tensor,
    eps16: float = torch.finfo(torch.float16).eps,
    eps32: float = torch.finfo(torch.float32).eps,
    eps64: float = torch.finfo(torch.float64).eps,
) -> float:
    if x.dtype == torch.float16:
        return eps16
    elif x.dtype == torch.float32:
        return eps32
    elif x.dtype == torch.float64:
        return eps64
    else:
        raise RuntimeError(f"Expected x to be floating-point, got {x.dtype}")
    
@torch.jit.script
def cosh(x: torch.Tensor) -> torch.Tensor:
    """Hyperbolic cosine. Domain=(-inf, inf)."""
    eps = _get_tensor_eps(x)
    clamp = float(math.log(2 / eps))
    return x.clamp(-clamp, clamp).cosh()

@torch.jit.script
def sinh(x: torch.Tensor) -> torch.Tensor:
    """Hyperbolic sine. Domain=(-inf, inf)."""
    eps = _get_tensor_eps(x)
    clamp = float(math.log(2 / eps))
    return x.clamp(-clamp, clamp).sinh()

@torch.jit.script
def tanh(x: torch.Tensor) -> torch.Tensor:
    """Hyperbolic tangent. Domain=(-inf, inf)."""
    eps = _get_tensor_eps(x)
    clamp = float(-math.log(eps / 2) / 2)
    return x.clamp(-clamp, clamp).tanh()

@torch.jit.script
def arcosh(x: torch.Tensor) -> torch.Tensor:
    """Inverse hyperbolic cosine. Domain=[1, inf)."""
    min_val = 1.0 + _get_tensor_eps(x)
    return x.clamp(min=min_val).double().add(torch.sqrt(x.clamp(min=min_val).double().pow(2) - 1)).log().to(x.dtype)

@torch.jit.script
def arsinh(x: torch.Tensor) -> torch.Tensor:
    """Inverse hyperbolic sine. Domain=(-inf, inf)."""
    return x.double().add(torch.sqrt(1 + x.double().pow(2))).log().to(x.dtype)

@torch.jit.script
def artanh(x: torch.Tensor) -> torch.Tensor:
    """Inverse hyperbolic tangent. Domain=(-1, 1)."""
    eps = _get_tensor_eps(x)
    x = x.clamp(-1 + eps, 1 - eps)
    return (torch.log1p(x.double()) - torch.log1p(-x.double())).mul(0.5).to(x.dtype)
