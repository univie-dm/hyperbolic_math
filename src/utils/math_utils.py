"""Math utils functions for hyperbolic operations with numerically stable limits."""

import math
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
def smooth_clamp_min(x: torch.Tensor, min_value: float, smoothing_factor: float=50) -> torch.Tensor:
    """Smoothly clamp tensor values to a minimum."""
    eps = _get_tensor_eps(x)
    shift = min_value + eps
    x_clamped = shift + torch.nn.functional.softplus(x - shift, beta=smoothing_factor)
    return torch.where(x < shift, x_clamped, x)

@torch.jit.script
def smooth_clamp_max(x: torch.Tensor, max_value: float, smoothing_factor: float=50) -> torch.Tensor:
    """Smoothly clamp tensor values to a maximum."""
    eps = _get_tensor_eps(x)
    shift = max_value - eps
    x_clamped = shift - torch.nn.functional.softplus(shift - x, beta=smoothing_factor)
    return torch.where(x > shift, x_clamped, x)

@torch.jit.script
def smooth_clamp(x: torch.Tensor, min_value: float, max_value: float, smoothing_factor: float=50) -> torch.Tensor:
    """Smoothly clamp tensor values to a range [min_value, max_value]."""
    x = smooth_clamp_max(x, max_value, smoothing_factor=smoothing_factor)
    return smooth_clamp_min(x, min_value, smoothing_factor=smoothing_factor)

@torch.jit.script
#TODO: test if this even makes a diff for clustering
def cosh(x: torch.Tensor) -> torch.Tensor:
    """Hyperbolic cosine. Domain=(-inf, inf)."""
    eps = _get_tensor_eps(x)
    clamp = float(math.log(2 / eps))
    #clamp = 88.0 if x.dtype == torch.float32 else 709.0
    x = smooth_clamp(x, -clamp, clamp)
    #x = x.clamp(-clamp, clamp)
    return torch.cosh(x)

@torch.jit.script
#TODO: test if this even makes a diff for clustering
def sinh(x: torch.Tensor) -> torch.Tensor:
    """Hyperbolic sine. Domain=(-inf, inf)."""
    eps = _get_tensor_eps(x)
    clamp = float(math.log(2 / eps))
    #clamp = 88.0 if x.dtype == torch.float32 else 709.0
    x = smooth_clamp(x, -clamp, clamp)
    #x = x.clamp(-clamp, clamp)
    return torch.sinh(x)

#@torch.jit.script
#TODO: test if clamping happens in the best run
def tanh(x: torch.Tensor) -> torch.Tensor:
    """Hyperbolic tangent. Domain=(-inf, inf)."""
    eps = _get_tensor_eps(x)
    clamp = float(-math.log(eps / 2) / 2)
    x_temp = smooth_clamp(x, -clamp, clamp)
    num_mismatches = torch.sum(x != x_temp).item()
    if num_mismatches > 0:
        pass
        #print(f"tanh: {num_mismatches} mismatches", flush=True)
    return torch.tanh(x)

@torch.jit.script
def acosh(x: torch.Tensor) -> torch.Tensor:
    """Inverse hyperbolic cosine. Domain=[1, inf)."""
    x = x.clamp_min(1.0)
    return torch.acosh(x)

@torch.jit.script
def asinh(x: torch.Tensor) -> torch.Tensor:
    """Inverse hyperbolic sine. Domain=(-inf, inf)."""
    eps = _get_tensor_eps(x)
    clamp = float(math.log(2 / eps))
    x = smooth_clamp(x, -clamp, clamp)
    return torch.asinh(x)

#@torch.jit.script
#TODO: check if smooth clamping has an impact on performance against best run
def atanh(x: torch.Tensor) -> torch.Tensor:
    """Inverse hyperbolic tangent. Domain=(-1, 1)."""
    eps = _get_tensor_eps(x)
    #x = smooth_clamp(x, -1 + eps, 1 - eps)
    x = x.clamp(-1 + eps, 1 - eps)
    return torch.atanh(x)
