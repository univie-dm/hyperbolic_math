import math
import torch

from typing import List
from .manifold import Manifold
from ..utils.math_utils import arcosh, cosh, sinh, smooth_clamp_min, smooth_clamp_max, smooth_clamp


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
        elif dtype == "float64" or dtype == torch.float64:
            self.dtype = torch.float64
            self.min_enorm = 1e-15
        else:
            raise ValueError(f"Unsupported dtype: {dtype}. Supported dtypes are float32 and float64.")

        if torch.finfo(c.dtype).eps < torch.finfo(self.dtype).eps:
            print(f"Warning: self.c.dtype is {c.dtype}, but self.dtype is {self.dtype}. "
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
            Point(s)
        y : torch.Tensor
            Point(s)
        axis : int
            Axis along which to compute the Minkowski inner product (default: -1)

        Returns
        -------
        res : torch.Tensor (dtype=self.dtype)
            The Minkowski inner product of x and y

        References
        ----------
        Ines Chami, et al. "Hyperbolic graph convolutional neural networks."
            Advances in neural information processing systems 32 (2019).
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
            Point(s)
        axis : int
            Axis along which to compute the Minkowski norm (default: -1)

        Returns
        -------
        res : torch.Tensor (dtype=self.dtype)
            The Minkowski norm(s) of x

        References
        ----------
        Ines Chami, et al. "Hyperbolic graph convolutional neural networks."
            Advances in neural information processing systems 32 (2019).
        """
        x, = self._2manifold_dtype([x])
        res = self._minkowski_inner(x, x, axis=axis).clamp_min(0.).sqrt() # TODO: smooth clamp??
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
        Ines Chami, et al. "Hyperbolic graph convolutional neural networks."
            Advances in neural information processing systems 32 (2019).
        """
        x, y = self._2manifold_dtype([x, y])
        res = arcosh(-self.c * self._minkowski_inner(x, y, axis=axis)) / self.c.sqrt()
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
        Ines Chami, et al. "Hyperbolic graph convolutional neural networks."
            Advances in neural information processing systems 32 (2019).
        """
        x, = self._2manifold_dtype([x])
        x0 = x.narrow(axis, 0, 1)
        res = arcosh(self.c.sqrt() * x0) / self.c.sqrt()
        return res

    def expmap(self, v: torch.Tensor, x: torch.Tensor, axis: int=-1, backproject: bool=True,
               version: str="normal") -> torch.Tensor:
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
        Ines Chami, et al. "Hyperbolic graph convolutional neural networks."
            Advances in neural information processing systems 32 (2019).
        """
        v, x = self._2manifold_dtype([v, x])
        # TODO: check if Taylor expansion has any advantages in real world applications
        if version[:-2] == "Taylor": # e.g. "Taylor 4" for degree 4 approx.
            degree = int(version[-1])
            scale = self.c * self._minkowski_inner(v, v, axis=axis)
            res = torch.zeros_like(x)
            for n in range(degree):
                scale_n = scale ** n / math.factorial(2*n)
                term = x + v / (2*n+1)
                res = res + scale_n * term
        else:
            # Normal expmap
            v_norm = self._minkowski_norm(v, axis=axis)
            c_norm_prod = (self.c.sqrt() * v_norm)#.clamp_min(self.min_enorm) # TODO
            res = cosh(c_norm_prod) * x + sinh(c_norm_prod) / c_norm_prod * v
        if backproject:
            res = self.proj(res, axis=axis)
        return res

    def expmap_0(self, v: torch.Tensor, axis: int=-1, backproject: bool=True,
                 version: str="normal") -> torch.Tensor:
        """
        Map tangent vector(s) v at the Hyperboloid origin to the Hyperboloid.
        [Exponential map]

        Parameters
        ----------
        v : torch.Tensor
            Vector(s) in the tangent space of the Hyperboloid origin
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
        Ines Chami, et al. "Hyperbolic graph convolutional neural networks."
            Advances in neural information processing systems 32 (2019).
        """
        v, = self._2manifold_dtype([v])
        # TODO: check if Taylor expansion has any advantages in real world applications
        if version[:-2] == "Taylor": # e.g. "Taylor 4" for degree 4 approx.
            degree = int(version[-1])
            origin = torch.zeros_like(v, dtype=self.dtype)
            if axis < 0:
                axis = v.dim() + axis
            slicing = [slice(None)] * v.dim()
            slicing[axis] = slice(0, 1)
            origin[tuple(slicing)] = 1 / self.c.sqrt()
            scale = self.c * self._minkowski_inner(v, v, axis=axis)
            res = torch.zeros_like(origin)
            for n in range(degree):
                scale_n = scale ** n / math.factorial(2*n)
                term = origin + v / (2*n+1)
                res = res + scale_n * term
        else:
            v_norm = self._minkowski_norm(v, axis=axis)
            c_norm_prod = self.c.sqrt() * v_norm
            res0 = cosh(c_norm_prod) + sinh(c_norm_prod) * v.narrow(axis, 0, 1) / v_norm
            res_rem = sinh(c_norm_prod) * v.narrow(axis, 1, v.shape[axis]-1) / v_norm
            res = torch.cat((res0, res_rem), dim=axis) / self.c.sqrt()
        if backproject:
            res = self.proj(res, axis=axis)
        return res

    def retraction(self, v: torch.Tensor, x: torch.Tensor, axis: int=-1, backproject: bool=True) -> torch.Tensor:
        """
        First-order approximation of the exponential map for vector(s) v at Hyperboloid point(s) x.
        [Retraction map]

        Parameters
        ----------
        v : torch.Tensor
            Vector(s) in the tangent space(s) of x
        x : torch.Tensor
            Hyperboloid point(s)
        axis : int
            Axis along which to compute the backprojection of the retraction (default: -1)
        backproject : bool
            Whether to project results back to the Hyperboloid (default: True)

        Returns
        -------
        res : torch.Tensor (dtype=self.dtype)
            The resulting Hyperboloid point(s) after approximate mapping v to the Hyperboloid

        References
        ----------
        Ines Chami, et al. "Hyperbolic graph convolutional neural networks."
            Advances in neural information processing systems 32 (2019).
        """
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
            Hyperboloid point(s)
        x : torch.Tensor
            Hyperboloid point(s)
        axis : int
            Axis along which to compute the logarithmic map (default: -1)

        Returns
        -------
        res : torch.Tensor (dtype=self.dtype)
            The resulting tangent vector(s) after mapping y to the tangent space(s) of x

        References
        ----------
        Ines Chami, et al. "Hyperbolic graph convolutional neural networks."
            Advances in neural information processing systems 32 (2019).
        """
        # Prob. not neceassary as this is only possible iff <x,y>=1/c,
        # which yields the evaluation of cosh(-1)=i*pi
        y, x = self._2manifold_dtype([y, x])
        dist = self.dist(x, y, axis=axis)
        num = y + self.c * self._minkowski_inner(x, y, axis=axis) * x
        denom = self._minkowski_norm(num, axis=axis)#.clamp_min(self.min_enorm) # TODO
        res = dist * num / denom
        if True:
            res = self.tangent_proj(res, x, axis=axis)
        return res

    def logmap_0(self, y: torch.Tensor, axis: int=-1) -> torch.Tensor:
        """
        Map Hyperboloid point(s) y to the tangent space of the Hyperboloid origin.
        [Logarithmic map]

        Parameters
        ----------
        y : torch.Tensor
            Hyperboloid point(s)
        axis : int
            Axis along which to compute the logarithmic map (default: -1)

        Returns
        -------
        res : torch.Tensor (dtype=self.dtype)
            The resulting tangent vector(s) after mapping y to the tangent space of the origin

        References
        ----------
        Ines Chami, et al. "Hyperbolic graph convolutional neural networks."
            Advances in neural information processing systems 32 (2019).
        """
        # TODO: should we just use logmap at the origin for speed here?
        y, = self._2manifold_dtype([y])
        y_rem = y.narrow(axis, 1, y.shape[axis]-1)
        y_rem_norm = y_rem.norm(p=2, dim=axis, keepdim=True)
        scale = self.dist_0(y, axis=axis) / y_rem_norm.clamp_min(self.min_enorm)
        res = torch.cat((torch.zeros_like(y.narrow(axis, 0, 1)), scale * y_rem), dim=axis)
        if True:
            origin = torch.zeros_like(res, dtype=self.dtype)
            if axis < 0:
                axis = res.dim() + axis
            slicing = [slice(None)] * res.dim()
            slicing[axis] = slice(0, 1)
            origin[tuple(slicing)] = 1 / self.c.sqrt()
            res = self.tangent_proj(res, origin, axis=axis)
        return res

    def ptransp(self, v: torch.Tensor, x: torch.Tensor, y: torch.Tensor, axis: int=-1) -> torch.Tensor:
        """
        Parallel transport tangent vector(s) v from the tangent space(s) of
        Hyperboloid point(s) x to the tangent space(s) of Hyperboloid point(s) y.

        Parameters
        ----------
        v : torch.Tensor
            Vector(s) in the tangent space(s) of x
        x : torch.Tensor
            Hyperboloid point(s)
        y : torch.Tensor
            Hyperboloid point(s)
        axis : int
            Axis along which to compute the parallel transport (default: -1)

        Returns
        -------
        res : torch.Tensor (dtype=self.dtype)
            The parallel transported tangent vector(s)

        References
        ----------
        Aaron Lou, et al. "Differentiating through the fréchet mean."
            International conference on machine learning (2020).
        """
        v, x, y = self._2manifold_dtype([v, x, y])
        vy = self._minkowski_inner(v, y, axis=axis)
        xy = self._minkowski_inner(x, y, axis=axis)
        denom = 1 / self.c - xy
        scale = vy / denom.clamp_min(self.min_enorm)
        res = v + scale * (x + y)
        if True:
            res = self.tangent_proj(res, y, axis=axis)
        return res

    def ptransp_0(self, v: torch.Tensor, y: torch.Tensor, axis: int=-1) -> torch.Tensor:
        """
        Parallel transport tangent vector(s) v from the tangent space of the
        Hyperboloid origin to the tangent space(s) of Hyperboloid point(s) y.

        Parameters
        ----------
        v : torch.Tensor
            Vector(s) in the tangent space of the Hyperboloid origin
        y : torch.Tensor
            Hyperboloid point(s)
        axis : int
            Axis along which to compute the parallel transport (default: -1)

        Returns
        -------
        res : torch.Tensor (dtype=self.dtype)
            The parallel transported tangent vector(s)

        References
        ----------
        Aaron Lou, et al. "Differentiating through the fréchet mean."
            International conference on machine learning (2020).
        """
        v, y = self._2manifold_dtype([v, y])
        origin = torch.zeros_like(y, dtype=self.dtype)
        if axis < 0:
            axis = y.dim() + axis
        slicing = [slice(None)] * y.dim()
        slicing[axis] = slice(0, 1)
        origin[tuple(slicing)] = 1 / self.c.sqrt()
        vy = self._minkowski_inner(v, y, axis=axis)
        y0 = y.narrow(axis, 0, 1)
        denom = 1 / self.c + y0 / self.c.sqrt()
        scale = vy / denom
        res = v + scale * (y + origin)
        if True:
            res = self.tangent_proj(res, y, axis=axis)
        return res

    def tangent_inner(self, u: torch.Tensor, v: torch.Tensor, x: torch.Tensor, axis: int=-1) -> torch.Tensor:
        """
        Compute the inner product(s) between tangent vectors u and v of the tangent space(s)
        at Hyperboloid point(s) x with respect to the Riemannian metric of the Hyperboloid.

        Parameters
        ----------
        u : torch.Tensor
            Vector(s) in the tangent space(s) of x
        v : torch.Tensor
            Vector(s) in the tangent space(s) of x
        x : torch.Tensor (ignored)
            Hyperboloid point(s)
        axis : int
            Axis along which to compute the tangent inner product (default: -1)
        Note: x is not used since the tangent inner product is just the restriction of the
              minkowski inner product to the tangent space with respect to x, but included
              for consistency with other manifolds.

        Returns
        -------
        res : torch.Tensor (dtype=self.dtype)
            The tangent inner product(s) of u and v

        References
        ----------
        Ines Chami, et al. "Hyperbolic graph convolutional neural networks."
            Advances in neural information processing systems 32 (2019).
        """
        u, v = self._2manifold_dtype([u, v])
        res = self._minkowski_inner(u, v, axis=axis)
        return res

    def tangent_norm(self, v: torch.Tensor, x: torch.Tensor, axis: int=-1) -> torch.Tensor:
        """
        Compute the norm(s) of tangent vector(s) v of the tangent space(s) at Hyperboloid
        point(s) x with respect to the Riemannian metric of the Hyperboloid.

        Parameters
        ----------
        v : torch.Tensor
            Vector(s) in the tangent space(s) of x
        x : torch.Tensor
            Hyperboloid point(s)
        axis : int
            Axis along which to compute the tangent norm (default: -1)
        Note: x is not used since the tangent norm is just the restriction of the
              minkowski norm to the tangent space with respect to x, but included
              for consistency with other manifolds.

        Returns
        -------
        res : torch.Tensor (dtype=self.dtype)
            The tangent norm(s) of v

        References
        ----------
        Ines Chami, et al. "Hyperbolic graph convolutional neural networks."
            Advances in neural information processing systems 32 (2019).
        """
        v, = self._2manifold_dtype([v])
        res = self._minkowski_norm(v, axis=axis)
        return res

    def egrad2rgrad(self, grad: torch.Tensor, x: torch.Tensor, axis: int=-1) -> torch.Tensor:
        """
        Compute the Riemannian gradient(s) at Hyperboloid point(s) x from the Euclidean gradient(s).

        Parameters
        ----------
        grad : torch.Tensor
            Euclidean gradient(s)
        x : torch.Tensor
            Hyperboloid point(s)
        axis : int
            Axis along which to compute the Riemannian gradient (default: -1)

        Returns
        -------
        res : torch.Tensor
            The Riemannian gradient(s) at x

        References
        ----------
        Maximillian Nickel, Douwe Kiela. "Learning continuous hierarchies in the lorentz model of hyperbolic geometry."
            International conference on machine learning. PMLR, 2018.
        """
        # Compute the orthogonal projection of the gradient onto the tangent space of x
        # in the manifold's precision and cast it to the gradient's precision
        x, = self._2manifold_dtype([x])
        normal = (-self.c * self._minkowski_inner(x, grad, axis=axis) * x).to(grad.dtype)
        res = grad - normal
        return res

    def proj(self, x: torch.Tensor, axis: int=-1) -> torch.Tensor:
        """
        Project point(s) x onto the Hyperboloid by scaling the time component(s) x0 of
        x such that the minkowski inner product of x with itself is equal to -1/c.

        Parameters
        ----------
        x : torch.Tensor
            Point(s)
        axis : int
            Axis along which to compute the projection (default: -1)

        Returns
        -------
        res : torch.Tensor (dtype=self.dtype)
            The projected Hyperboloid point(s)

        References
        ----------
        Ines Chami, et al. "Hyperbolic graph convolutional neural networks."
            Advances in neural information processing systems 32 (2019).
        """
        x, = self._2manifold_dtype([x])
        x_rem = x.narrow(axis, 1, x.shape[axis]-1)
        x_rem_norm_sq = x_rem.pow(2).sum(dim=axis, keepdim=True)
        x0 = (x_rem_norm_sq + 1 / self.c).sqrt()
        res = torch.cat((x0, x_rem), dim=axis)
        return res

    def tangent_proj(self, v: torch.Tensor, x: torch.Tensor, axis: int=-1) -> torch.Tensor:
        """
        Project point(s) v onto the tangent space of Hyperboloid point(s) x
        v such that the minkowski inner product of v and x vanishes.

        Parameters
        ----------
        v : torch.Tensor
            Point(s)
        x : torch.Tensor
            Hyperboloid point(s)
        axis : int
            Axis along which to compute the projection (default: -1)

        Returns
        -------
        res : torch.Tensor (dtype=self.dtype)
            The tangent vector(s)

        References
        ----------
        Ines Chami, et al. "Hyperbolic graph convolutional neural networks."
            Advances in neural information processing systems 32 (2019).
        """
        x, = self._2manifold_dtype([x])
        x_normalized = x / (-self.c * self._minkowski_inner(x, x, axis=axis)).sqrt()
        coeff = self._minkowski_inner(x_normalized, v, axis=axis) / self._minkowski_inner(x_normalized, x_normalized, axis=axis)
        res = v - coeff * x_normalized
        return res

    def is_in_manifold(self, x: torch.Tensor, axis: int=-1) -> bool:
        """
        Check if point(s) x lie on the Hyperboloid.

        Parameters
        ----------
        x : torch.Tensor
            Hyperboloid point(s)
        axis : int
            Axis along which to check if x lies in the Hyperboloid (default: -1)

        Returns
        -------
        res : bool
            True if all points x lie in the Hyperboloid, False otherwise
        """
        x, = self._2manifold_dtype([x])
        res = torch.allclose(self._minkowski_inner(x, x, axis=axis), -1 / self.c, atol=8e-05)
        return res

    def is_in_tangent_space(self, v: torch.Tensor, x: torch.Tensor, axis: int=-1) -> bool:
        """
        Check if vector(s) v belong to the tangent space(s) at Hyperboloid point(s) x.

        Parameters
        ----------
        v : torch.Tensor
            Vector(s)
        x : torch.Tensor
            Hyperboloid point(s)
        axis : int
            Axis along which to check if v belong to the tangent space (default: -1)

        Returns
        -------
        res : bool
            True if all vectors v belong to their tangent spaces, False otherwise
        """
        v, x = self._2manifold_dtype([v, x])
        res = torch.all(torch.abs(self._minkowski_inner(v, x, axis=axis)) < 5e-04)
        return res

    def to_poincare(self, x: torch.Tensor, axis: int=-1) -> torch.Tensor:
        """
        Map Hyperboloid point(s) x to the PoincareBall.

        Parameters
        ----------
        x : torch.Tensor
            Hyperboloid point(s)
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
        res = x_rem / (1. + self.c.sqrt() * x0)
        return res
