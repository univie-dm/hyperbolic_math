import math
import torch

from typing import Dict
from ..manifolds import PoincareBall
from ..utils.math_utils import smooth_clamp, asinh, cosh, sinh


# Dictionary mapping of dtype strings to torch dtypes
DTYPE_MAP: Dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
}

def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype representation to torch dtype.

    Parameters
    ----------
    dtype_str : str
        String representation of dtype ('float32', or 'float64')

    Returns
    -------
    torch.dtype
        Corresponding torch dtype

    Raises
    ------
    ValueError
        If dtype_str is not supported
    """
    if dtype_str not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {dtype_str}. "
                         f"Supported dtypes are: {', '.join(DTYPE_MAP.keys())}")
    return DTYPE_MAP[dtype_str]

def compute_mlr_PP(manifold: PoincareBall, x: torch.Tensor, z: torch.Tensor, r: torch.Tensor,
                   hyperbolic_axis: int, clamping_factor: float, smoothing_factor: float) -> torch.Tensor:
        """
        Internal method for computing the 'Hyperbolic Neural Networks ++' multinomial linear regression.

        Parameters
        ----------
        manifold : PoincareBall
            The PoincareBall manifold
        x : torch.Tensor (B, in_dim)
            PoincareBall point(s)
        z : torch.Tensor (out_dim, in_dim)
            Hyperplane tangent normal(s) in the tangent space at the origin
        r : torch.Tensor (out_dim, 1)
            Hyperplane PoincareBall translation(s) defined by the scalar r and z
        clamping: float
            Clamping value for the output
        smoothing: float
            Smoothing factor for the output

        Returns
        -------
        res : torch.Tensor (B, out_dim)
            The multinomial linear regression score(s) of x with respect to the linear model(s) defined by a and z.

        References
        ----------
        Shimizu Ryohei, Yusuke Mukuta, and Tatsuya Harada. "Hyperbolic neural networks++."
            arXiv preprint arXiv:2006.08210 (2020).
        """
        sqrt_c = manifold.c.sqrt()
        sqrt_c2r = 2 * sqrt_c * r.T # (out_dim, 1)
        z_norm = z.norm(p=2, dim=hyperbolic_axis, keepdim=True).clamp_min(manifold.min_enorm) # (out_dim, 1)
        lambda_x = manifold._lambda(x, axis=hyperbolic_axis) # (B, 1)
        z_unitx = (x.unsqueeze(-1) * (z / z_norm).T).sum(dim=1) # (B, out_dim)
        asinh_arg = (1-lambda_x) * sinh(sqrt_c2r) + sqrt_c * lambda_x * cosh(sqrt_c2r) * z_unitx # (B, out_dim)
        # Improve the performance by smoothly clamping the input of asinh() to approximately the range of ...
        # ... [-16*clamping_factor, 16*clamping_factor] for float32
        # ... [-36*clamping_factor, 36*clamping_factor] for float64
        eps = torch.finfo(torch.float32).eps if manifold.dtype == torch.float32 else torch.finfo(torch.float64).eps
        clamp = clamping_factor * float(math.log(2 / eps))
        asinh_arg = smooth_clamp(asinh_arg, -clamp, clamp, smoothing_factor) # (B, out_dim)
        signed_dist2hyp = asinh(asinh_arg) / sqrt_c # (B, out_dim)
        res = 2 * z_norm.T * signed_dist2hyp # (B, out_dim)
        return res
