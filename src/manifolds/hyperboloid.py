import torch

from typing import List
from .manifold import Manifold
from ..utils.math_utils import arcosh, cosh, sinh, smooth_clamp_min


class Hyperboloid(Manifold):
    """
    Hyperboloid manifold class.
    Convention: -x0^2 + x1^2 + ... + xd^2 = -1/c, x0 > 0, with c > 0 and sectional curvature -c.
    """
    def __init__(
        #TODO
        self,
        c: torch.Tensor = torch.tensor([1.]),
        trainable_c: bool = False,
        dtype: str | torch.dtype = "float32",
    ):
        super().__init__(c, trainable_c)
        self.name = "Hyperboloid"

        # The following parameters are derived from the unittests
        if dtype == "float32" or dtype == torch.float32:
            self.dtype = torch.float32
            self.min_enorm = 1e-15
            #self.max_enorm_eps = 5e-06
        elif dtype == "float64" or dtype == torch.float64:
            self.dtype = torch.float64
            self.min_enorm = 1e-15
            #self.max_enorm_eps = 1e-08
        else:
            raise ValueError(f"Unsupported dtype: {dtype}. Supported dtypes are float32 and float64.")

        if torch.finfo(c.dtype).eps < torch.finfo(self.dtype).eps:
            print(f"Warning: self.c.dtype is {c.dtype}, but self.dtype is {self.dtype}."
                  f"All manifold operations will be performed in precision {c.dtype}!")
            self.dtype = c.dtype

    def _2manifold_dtype(self, xs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Convert the list of tensor(s) xs to the Hyperboloid's dtype.

        Parameters
        ----------
        xs : List[torch.Tensor]
            List of tensor(s)

        Returns
        -------
        res : List[torch.Tensor]
            The list of tensor(s) converted to the Hyperboloid's dtype
        """
        res = []
        for x in xs:
            res.append(x.to(self.dtype))
        return res

    def _minkowski_inner(self, x: torch.Tensor, y: torch.Tensor, axis: int=-1) -> torch.Tensor:
        """
        Computes the Minkowski inner product(s) of x and y with metric signature (-, +, ..., +).

        Parameters
        ----------
        x : torch.Tensor
            Ambient space/Hyperboloid point(s)
        y : torch.Tensor
            Ambient space/Hyperboloid point(s)
        axis : int
            Axis along which to compute the Minkowski inner product (default: -1)

        Returns
        -------
        res : torch.Tensor (dtype=self.dtype)
            The Minkowski inner product of x and y

        References
        ----------
        #TODO: ...
        """
        x, y = self._2manifold_dtype([x, y])
        xy_prod = x * y
        xy0 = xy_prod.narrow(axis, 0, 1)
        xy_rem = xy_prod.narrow(axis, 1, x.shape[axis]-1).sum(dim=axis, keepdim=True)
        res = xy_rem - xy0
        return res

    def _minkowski_norm(self, x: torch.Tensor, axis: int=-1) -> torch.Tensor:
        """
        Computes the Minkowski norm(s) of x with metric signature (-, +, ..., +).

        Parameters
        ----------
        x : torch.Tensor
            Ambient space/Hyperboloid point(s)
        axis : int
            Axis along which to compute the Minkowski norm (default: -1)

        Returns
        -------
        res : torch.Tensor (dtype=self.dtype)
            The Minkowski norm(s) of x

        References
        ----------
        #TODO: ...
        """
        x, = self._2manifold_dtype([x])
        res = (self._minkowski_inner(x, x, axis=axis)).clamp_min(0.).sqrt()
        return res

    def addition(self, x: torch.Tensor, y: torch.Tensor, axis: int=-1, backproject: bool=True) -> torch.Tensor:
        #TODO
        x, y = self._2manifold_dtype([x, y])
        x2 = x.pow(2).sum(dim=axis, keepdim=True)
        y2 = y.pow(2).sum(dim=axis, keepdim=True)
        xy = (x * y).sum(dim=axis, keepdim=True)
        num = (1 + 2 * self.c * xy + self.c * y2) * x + (1 - self.c * x2) * y
        denom = (1 + 2 * self.c * xy + self.c**2 * x2 * y2).clamp_min(self.min_enorm)
        res = num / denom
        if backproject:
            res = self.proj(res, axis=axis)
        # Some code:
        # u = self.logmap0(y, c)
        # v = self.ptransp0(x, u, c)
        # return self.expmap(v, x, c)
        return res

    def scalar_mul(self, r: torch.Tensor, x: torch.Tensor, axis: int=-1, backproject: bool=True) -> torch.Tensor:
        #TODO
        r, x = self._2manifold_dtype([r, x])
        x_norm = x.norm(p=2, dim=axis, keepdim=True).clamp_min(self.min_enorm)
        c_norm_prod = self.c.sqrt() * x_norm
        res = torch.tanh(r * torch.atanh(c_norm_prod)) / c_norm_prod * x
        if backproject:
            res = self.proj(res, axis=axis)
        return res

    def dist(self, x: torch.Tensor, y: torch.Tensor, axis: int=-1) -> torch.Tensor:
        """
        Compute the geodesic distance(s) between Hyperboloid point(s) x and y.

        Parameters
        ----------
        x : torch.Tensor
            Hyperboloid point(s)
        y : torch.Tensor
            Hyperboloid point(s)
        axis : int
            Axis along which to compute the geodesic distance (default: -1)

        Returns
        -------
        res : torch.Tensor (dtype=self.dtype)
            The geodesic distance(s) between x and y

        References
        ----------
        #TODO: ...
        """
        x, y = self._2manifold_dtype([x, y])
        res = arcosh(smooth_clamp_min(-self.c * self._minkowski_inner(x, y, axis=axis), 1.)) / self.c.sqrt()
        return res

    def dist_0(self, x: torch.Tensor, axis: int=-1) -> torch.Tensor:
        """
        Compute the geodesic distance(s) of Hyperboloid point(s) x from/to the Hyperboloid origin.

        Parameters
        ----------
        x : torch.Tensor
            Hyperboloid point(s)
        axis : int
            Axis along which to compute the geodesic distance (default: -1)

        Returns
        -------
        res : torch.Tensor (dtype=self.dtype)
            The geodesic distance(s) of x from/to the Hyperboloid origin

        References
        ----------
        #TODO: ...
        """
        x, = self._2manifold_dtype([x])
        if axis < 0:
            axis = x.dim() + axis
        slicing = [slice(None)] * x.dim()
        slicing[axis] = slice(0, 1)
        res = torch.acosh((self.c.sqrt() * x[tuple(slicing)]).clamp_min(1.)) / self.c.sqrt()
        return res

    def expmap(self, v: torch.Tensor, x: torch.Tensor, axis: int=-1, backproject: bool=True) -> torch.Tensor:
        """
        Map tangent vector(s) v at Hyperboloid point(s) x to the Hyperboloid.
        [Exponential map]

        Parameters
        ----------
        v : torch.Tensor
            Vector(s) in the tangent space(s) of x
        x : torch.Tensor
            Hyperboloid point(s)
        axis : int
            Axis along which to compute the exponential map (default: -1)
        backproject : bool
            Whether to project results back to the Hyperboloid (default: True)

        Returns
        -------
        res : torch.Tensor (dtype=self.dtype)
            The resulting Hyperboloid point(s) after mapping v to the Hyperboloid

        References
        ----------
        #TODO: ...

        Stability
        ---------
        expmap converges towards the addition x+v as the minkowski norm of v and/or c approaches zero,
        since torch.cosh(z) ~ 1, and sinh(z) ~ 0 for small z.
        """
        v, x = self._2manifold_dtype([v, x])
        v_norm = self._minkowski_norm(v, axis=axis)
        c_norm_prod = (self.c.sqrt() * v_norm).clamp_min(self.min_enorm)
        res = cosh(c_norm_prod) * x + sinh(c_norm_prod) / c_norm_prod * v
        if backproject:
            res = self.proj(res, axis=axis)
        return res

    def expmap_0(self, v: torch.Tensor, axis: int=-1, backproject: bool=True) -> torch.Tensor:
        v, = self._2manifold_dtype([v])
        v_norm = v.norm(p=2, dim=axis, keepdim=True)
        c_norm_prod = (self.c.sqrt() * v_norm).clamp_min(self.min_enorm)
        res = torch.tanh(c_norm_prod) / c_norm_prod * v
        if backproject:
            res = self.proj(res, axis=axis)
        return res

    def retraction(self, v: torch.Tensor, x: torch.Tensor, axis: int=-1, backproject: bool=True) -> torch.Tensor:
        v, x = self._2manifold_dtype([v, x])
        res = x + v
        if backproject:
            res = self.proj(res, axis=axis)
        return res

    def logmap(self, y: torch.Tensor, x: torch.Tensor, axis: int=-1) -> torch.Tensor:
        """
        Map Hyperboloid point(s) y to the tangent space(s) of Hyperboloid point(s) x.
        [Logarithmic map]

        Parameters
        ----------
        y : torch.Tensor
            PoincareBall point(s)
        x : torch.Tensor
            PoincareBall point(s)
        axis : int
            Axis along which to compute the logarithmic map (default: -1)

        Returns
        -------
        res : torch.Tensor (dtype=self.dtype)
            The resulting tangent vector(s) after mapping y to the tangent space(s) of x

        References
        ----------
        #TODO: ...

        Stability
        ---------
        #TODO: ...
        logmap converges towards the identity map as the norm of vector(s) y-x approaches zero,
        since artanh(z) ~ z for small z.
        """
        y, x = self._2manifold_dtype([y, x])
        dist = self.dist(x, y, axis=axis)
        num = y + self.c * self._minkowski_inner(x, y, axis=axis) * x
        denom = self._minkowski_norm(num, axis=axis)#.clamp_min(self.min_enorm)
        res = num * dist / denom
        return res

    def logmap_0(self, y: torch.Tensor, axis: int=-1) -> torch.Tensor:
        y, = self._2manifold_dtype([y])
        y_norm = y.norm(p=2, dim=axis, keepdim=True)
        c_norm_prod = (self.c.sqrt() * y_norm).clamp_min(self.min_enorm)
        res = torch.atanh(c_norm_prod) / c_norm_prod * y
        return res

    def ptransp(self, v: torch.Tensor, x: torch.Tensor, y: torch.Tensor, axis: int=-1) -> torch.Tensor:
        v, x, y = self._2manifold_dtype([v, x, y])
        conformal_frac = self._lambda(x, axis=axis) / self._lambda(y, axis=axis)
        res = conformal_frac * self._gyration(y, -x, v, axis=axis)
        return res

    def ptransp_0(self, v: torch.Tensor, y: torch.Tensor, axis: int=-1) -> torch.Tensor:
        v, y = self._2manifold_dtype([v, y])
        conformal_frac = 2 / self._lambda(y, axis=axis)
        res = conformal_frac * v
        return res

    def tangent_inner(self, u: torch.Tensor, v: torch.Tensor, x: torch.Tensor, axis: int=-1) -> torch.Tensor:
        u, v, x = self._2manifold_dtype([u, v, x])
        res = (u * v).sum(dim=axis, keepdim=True) * self._lambda(x, axis=axis) ** 2
        return res

    def tangent_norm(self, v: torch.Tensor, x: torch.Tensor, axis: int=-1) -> torch.Tensor:
        v, x = self._2manifold_dtype([v, x])
        res = self._lambda(x, axis=axis) * v.norm(p=2, dim=axis, keepdim=True)
        return res

    def egrad2rgrad(self, grad: torch.Tensor, x: torch.Tensor, axis: int=-1) -> torch.Tensor:
        # Compute the conformal factor in the manifold's precision and cast it to the gradient's precision
        x, = self._2manifold_dtype([x])
        conformal_scale = (self._lambda(x, axis=axis) ** 2).to(grad.dtype)
        res = grad / conformal_scale
        return res

    def proj(self, x: torch.Tensor, axis: int=-1) -> torch.Tensor:
        x, = self._2manifold_dtype([x])
        x_rem = x.narrow(axis, 1, x.shape[axis]-1)
        x_rem_norm_sq = x_rem.pow(2).sum(dim=axis, keepdim=True)
        x0 = (x_rem_norm_sq + 1 / self.c).sqrt()
        res = torch.cat((x0, x_rem), dim=axis)
        return res

    def is_in_manifold(self, x: torch.Tensor, axis: int=-1) -> bool:
        x, = self._2manifold_dtype([x])
        xBx = self._minkowski_inner(x, x, axis=axis)
        res = torch.allclose(-1 / self.c, xBx, atol=1e-07)
        return res

    def is_in_tangent_space(self, v: torch.Tensor, x: torch.Tensor, axis: int=-1) -> bool:
        res = True
        return res

    def to_poincare(self, x: torch.Tensor, ideal: bool=False, axis: int=-1) -> torch.Tensor:
        """
        Map Hyperboloid point(s) x to the PoincareBall.

        Parameters
        ----------
        x : torch.Tensor
            Hyperboloid point(s)
        ideal : bool
            Whether to convert x to ideal (=boundary) PoincareBall point(s) (default: False)
        axis : int
            Axis along which to compute the mapping (default: -1)

        Returns
        -------
        res : torch.Tensor (dtype=self.dtype)
            The PoincareBall point(s)
        """
        x, = self._2manifold_dtype([x])
        x0 = x.narrow(axis, 0, 1)
        x_rem = x.narrow(axis, 1, x.shape[axis]-1)
        if ideal:
            res = x_rem / (x0 * self.c.sqrt())
        else:
            res = x_rem / (1. + self.c.sqrt() * x0)
        return res
