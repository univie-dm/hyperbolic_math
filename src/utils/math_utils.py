"""Math utils functions for hyperbolic operations with numerically stable limits."""

import math

import torch


def cosh(x):
    eps = torch.finfo(x.dtype).eps
    clamp = float(math.log(2 / eps))  # Calculate limit based on precision
    return x.clamp(-clamp, clamp).cosh()


def sinh(x):
    eps = torch.finfo(x.dtype).eps
    clamp = float(math.log(2 / eps))
    return x.clamp(-clamp, clamp).sinh()


def tanh(x):
    eps = torch.finfo(x.dtype).eps
    clamp = float(-math.log(eps / 2) / 2)
    return x.clamp(-clamp, clamp).tanh()


def arcosh(x):
    """Inverse hyperbolic cosine."""
    # Domain starts at 1, upper bound depends on dtype max
    min_val = 1.0 + torch.finfo(x.dtype).eps
    return x.clamp(min=min_val).double().add(torch.sqrt(x.clamp(min=min_val).double().pow(2) - 1)).log().to(x.dtype)


def arsinh(x):
    """Inverse hyperbolic sine."""
    # No domain restrictions, just dtype limits
    return x.double().add(torch.sqrt(1 + x.double().pow(2))).log().to(x.dtype)


def artanh(x):
    """Inverse hyperbolic tangent."""
    # Domain is strictly (-1, 1)
    eps = torch.finfo(x.dtype).eps
    x = x.clamp(-1 + eps, 1 - eps)
    return (torch.log1p(x.double()) - torch.log1p(-x.double())).mul(0.5).to(x.dtype)


def sign(x):
    return torch.sign(x.sign() + 0.5)


def sabs(x, eps: float = 1e-15):
    return x.abs().add_(eps)


def clamp_abs(x, eps: float = 1e-15):
    s = sign(x)
    return s * sabs(x, eps=eps)


def abs_zero_grad(x):
    # this op has derivative equal to 1 at zero
    return x * sign(x)


# TODO: Use ours to replace
def arsin_k_zero_taylor(x: torch.Tensor, c: torch.Tensor, order: int = -1):
    k = -c
    if order == 0:
        return x
    k = abs_zero_grad(k)
    if order == -1 or order == 5:
        return (
            x
            + k * x**3 / 6
            + 3 / 40 * k**2 * x**5
            + 5 / 112 * k**3 * x**7
            + 35 / 1152 * k**4 * x**9
            + 63 / 2816 * k**5 * x**11
            # + o(k**6)
        )
    elif order == 1:
        return x + k * x**3 / 6
    elif order == 2:
        return x + k * x**3 / 6 + 3 / 40 * k**2 * x**5
    elif order == 3:
        return x + k * x**3 / 6 + 3 / 40 * k**2 * x**5 + 5 / 112 * k**3 * x**7
    elif order == 4:
        return x + k * x**3 / 6 + 3 / 40 * k**2 * x**5 + 5 / 112 * k**3 * x**7 + 35 / 1152 * k**4 * x**9
    else:
        raise RuntimeError("order not in [-1, 5]")


def arsin_k(x: torch.Tensor, c: torch.Tensor):
    k = -c
    k_sign = k.sign()
    zero = torch.zeros((), device=k.device, dtype=k.dtype)
    k_zero = k.isclose(zero)
    # shrink sign
    k_sign = torch.masked_fill(k_sign, k_zero, zero.to(k_sign.dtype))
    if torch.all(k_zero):
        return arsin_k_zero_taylor(x, k)
    k_sqrt = sabs(k).sqrt()
    scaled_x = x * k_sqrt

    if torch.all(k_sign.lt(0)):
        return k_sqrt.reciprocal() * arsinh(scaled_x)
    elif torch.all(k_sign.gt(0)):
        return k_sqrt.reciprocal() * scaled_x.asin()
    else:
        arsin_k_nonzero = (
            torch.where(
                k_sign.gt(0),
                scaled_x.clamp(-1 + 1e-7, 1 - 1e-7).asin(),
                arsinh(scaled_x),
            )
            * k_sqrt.reciprocal()
        )
        return torch.where(k_zero, arsin_k_zero_taylor(x, k, order=1), arsin_k_nonzero)
