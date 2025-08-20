"""Math utils functions for hyperbolic operations with numerically stable limits."""

import torch


@torch.jit.script
def _get_tensor_eps(
    x: torch.Tensor,
    eps32: float = torch.finfo(torch.float32).eps,
    eps64: float = torch.finfo(torch.float64).eps,
) -> float:
    if x.dtype == torch.float32:
        return eps32
    elif x.dtype == torch.float64:
        return eps64
    else:
        raise RuntimeError(f"Expected x to be floating-point, got {x.dtype}")

@torch.jit.script
def smooth_clamp_min(x: torch.Tensor, min_value: float) -> torch.Tensor:
    """Smoothly clamp tensor values to a minimum."""
    eps = _get_tensor_eps(x)
    shift = min_value + eps
    x_clamped = shift + torch.nn.functional.softplus(x - shift, beta=100.0)
    return torch.where(x < shift, x_clamped, x)

@torch.jit.script
def smooth_clamp_max(x: torch.Tensor, max_value: float) -> torch.Tensor:
    """Smoothly clamp tensor values to a maximum."""
    eps = _get_tensor_eps(x)
    shift = max_value - eps
    x_clamped = shift - torch.nn.functional.softplus(shift - x, beta=100.0)
    return torch.where(x > shift, x_clamped, x)

@torch.jit.script
def smooth_clamp(x: torch.Tensor, min_value: float, max_value: float) -> torch.Tensor:
    """Smoothly clamp tensor values to a range [min_value, max_value]."""
    return smooth_clamp_min(smooth_clamp_max(x, max_value), min_value)

@torch.jit.script
def cosh(x: torch.Tensor) -> torch.Tensor:
    """Hyperbolic cosine. Domain=(-inf, inf)."""
    # Practical limits: float32 ~88.0, float64 ~709.0
    clamp = 88.0 if x.dtype == torch.float32 else 709.0
    return torch.cosh(x.clamp(-clamp, clamp))

@torch.jit.script
def sinh(x: torch.Tensor) -> torch.Tensor:
    """Hyperbolic sine. Domain=(-inf, inf)."""
    # Practical limits: float32 ~88.0, float64 ~709.0
    clamp = 88.0 if x.dtype == torch.float32 else 709.0
    return torch.sinh(x.clamp(-clamp, clamp))

@torch.jit.script
def tanh(x: torch.Tensor) -> torch.Tensor:
    """Hyperbolic tangent. Domain=(-inf, inf)."""
    return torch.tanh(x)

@torch.jit.script
def arcosh(x: torch.Tensor) -> torch.Tensor:
    """Inverse hyperbolic cosine. Domain=[1, inf)."""
    return torch.acosh(x.clamp_min(1.0))

@torch.jit.script
def arsinh(x: torch.Tensor) -> torch.Tensor:
    """Inverse hyperbolic sine. Domain=(-inf, inf)."""
    return torch.asinh(x)

@torch.jit.script
def artanh(x: torch.Tensor) -> torch.Tensor:
    """Inverse hyperbolic tangent. Domain=(-1, 1)."""
    eps = _get_tensor_eps(x)
    return torch.atanh(x.clamp(-1 + eps, 1 - eps))
