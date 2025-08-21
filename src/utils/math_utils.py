"""Math utils functions for hyperbolic operations with numerically stable limits."""

import math
import torch


VERSION_SMOOTH = "classic"
# "classic":         small range,    sharp clamps
# "classic_smooth":  small range,    original clamps
# "extended":        extended range, sharp clamps
# "extended_smooth": extended range, smooth clamps

#@torch.jit.script
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

#@torch.jit.script
def smooth_clamp_min(x: torch.Tensor, min_value: float) -> torch.Tensor:
    """Smoothly clamp tensor values to a minimum."""
    eps = _get_tensor_eps(x)
    shift = min_value + eps
    x_clamped = shift + torch.nn.functional.softplus(x - shift, beta=100.0)
    return torch.where(x < shift, x_clamped, x)

#@torch.jit.script
def smooth_clamp_max(x: torch.Tensor, max_value: float) -> torch.Tensor:
    """Smoothly clamp tensor values to a maximum."""
    eps = _get_tensor_eps(x)
    shift = max_value - eps
    x_clamped = shift - torch.nn.functional.softplus(shift - x, beta=100.0)
    return torch.where(x > shift, x_clamped, x)

#@torch.jit.script
def smooth_clamp(x: torch.Tensor, min_value: float, max_value: float) -> torch.Tensor:
    """Smoothly clamp tensor values to a range [min_value, max_value]."""
    return smooth_clamp_min(smooth_clamp_max(x, max_value), min_value)

#@torch.jit.script
def cosh(x: torch.Tensor) -> torch.Tensor:
    """Hyperbolic cosine. Domain=(-inf, inf)."""
    if VERSION_SMOOTH == "classic":
        eps = _get_tensor_eps(x)
        clamp = float(math.log(2 / eps))
        return torch.cosh(x.clamp(-clamp, clamp))
    elif VERSION_SMOOTH == "classic_smooth":
        eps = _get_tensor_eps(x)
        clamp = float(math.log(2 / eps))
        return torch.cosh(smooth_clamp(x, -clamp, clamp))
    elif VERSION_SMOOTH == "extended":
        # Practical limits: float32 ~88.0, float64 ~709.0
        clamp = 88.0 if x.dtype == torch.float32 else 709.0
        return torch.cosh(x.clamp(-clamp, clamp))
    elif VERSION_SMOOTH == "extended_smooth":
        # Practical limits: float32 ~88.0, float64 ~709.0
        clamp = 88.0 if x.dtype == torch.float32 else 709.0
        return torch.cosh(smooth_clamp(x, -clamp, clamp))

#@torch.jit.script
def sinh(x: torch.Tensor) -> torch.Tensor:
    """Hyperbolic sine. Domain=(-inf, inf)."""
    if VERSION_SMOOTH == "classic":
        eps = _get_tensor_eps(x)
        clamp = float(math.log(2 / eps))
        return torch.sinh(x.clamp(-clamp, clamp))
    elif VERSION_SMOOTH == "classic_smooth":
        eps = _get_tensor_eps(x)
        clamp = float(math.log(2 / eps))
        return torch.sinh(smooth_clamp(x, -clamp, clamp))
    elif VERSION_SMOOTH == "extended":
        # Practical limits: float32 ~88.0, float64 ~709.0
        clamp = 88.0 if x.dtype == torch.float32 else 709.0
        return torch.sinh(x.clamp(-clamp, clamp))
    elif VERSION_SMOOTH == "extended_smooth":
        # Practical limits: float32 ~88.0, float64 ~709.0
        clamp = 88.0 if x.dtype == torch.float32 else 709.0
        return torch.sinh(smooth_clamp(x, -clamp, clamp))

#@torch.jit.script
def tanh(x: torch.Tensor) -> torch.Tensor:
    """Hyperbolic tangent. Domain=(-inf, inf)."""
    if VERSION_SMOOTH == "classic":
        eps = _get_tensor_eps(x)
        clamp = float(-math.log(eps / 2) / 2)
        return torch.tanh(x.clamp(-clamp, clamp))
    elif VERSION_SMOOTH == "classic_smooth":
        eps = _get_tensor_eps(x)
        clamp = float(-math.log(eps / 2) / 2)
        return torch.tanh(smooth_clamp(x, -clamp, clamp))
    elif VERSION_SMOOTH == "extended":
        # Can possibly be extended if nan?
        return torch.tanh(x)
    elif VERSION_SMOOTH == "extended_smooth":
        # Can possibly be extended if nan?
        return torch.tanh(x)

#@torch.jit.script
def acosh(x: torch.Tensor) -> torch.Tensor:
    """Inverse hyperbolic cosine. Domain=[1, inf)."""
    if VERSION_SMOOTH == "classic":
        print("classic", flush=True)
        eps = _get_tensor_eps(x)
        clamp = float(math.log(2 / eps))
        return torch.acosh(x.clamp(1.0, clamp))
    elif VERSION_SMOOTH == "classic_smooth":
        print("classic_smooth", flush=True)
        eps = _get_tensor_eps(x)
        clamp = float(math.log(2 / eps))
        return torch.acosh(smooth_clamp(x, 1.0, clamp))
    elif VERSION_SMOOTH == "extended":
        print("extended", flush=True)
        # Can possibly be extended if nan?
        return torch.acosh(x.clamp_min(1.0))
    elif VERSION_SMOOTH == "extended_smooth":
        print("extended_smooth", flush=True)
        # Can possibly be extended if nan?
        return torch.acosh(smooth_clamp_min(x, 1.0))

#@torch.jit.script
def asinh(x: torch.Tensor) -> torch.Tensor:
    """Inverse hyperbolic sine. Domain=(-inf, inf)."""
    if VERSION_SMOOTH == "classic":
        eps = _get_tensor_eps(x)
        clamp = float(math.log(2 / eps))
        return torch.asinh(x.clamp(-clamp, clamp))
    elif VERSION_SMOOTH == "classic_smooth":
        eps = _get_tensor_eps(x)
        clamp = float(math.log(2 / eps))
        return torch.asinh(smooth_clamp(x, -clamp, clamp))
    elif VERSION_SMOOTH == "extended":
        # Can possibly be extended if nan?
        return torch.asinh(x)
    elif VERSION_SMOOTH == "extended_smooth":
        # Can possibly be extended if nan?
        return torch.asinh(x)

#@torch.jit.script
def atanh(x: torch.Tensor) -> torch.Tensor:
    """Inverse hyperbolic tangent. Domain=(-1, 1)."""
    eps = _get_tensor_eps(x)
    if VERSION_SMOOTH == "classic":
        return torch.atanh(x.clamp(-1 + eps, 1 - eps))
    elif VERSION_SMOOTH == "classic_smooth":
        return torch.atanh(smooth_clamp(x, -1 + eps, 1 - eps))
    elif VERSION_SMOOTH == "extended":
        return torch.atanh(x.clamp(-1 + eps, 1 - eps))
    elif VERSION_SMOOTH == "extended_smooth":
        return torch.atanh(smooth_clamp(x, -1 + eps, 1 - eps))
