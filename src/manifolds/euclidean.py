import torch

from typing import List
from .manifold import Manifold


class Euclidean(Manifold):
    """
    Euclidean manifold class.
    """

    def __init__(
        self,
        c: torch.Tensor=torch.tensor([0.]),
        trainable_c: bool=False,
        dtype: str="float32",
    ):
        super().__init__(torch.tensor([0.]), trainable_c=False)
        self.name = "Euclidean"
        self.dtype = dtype
        if trainable_c:
            print("Warning: trainable_c is not supported for Euclidean manifold. Setting it to False.")
        elif not torch.allclose(c, torch.zeros_like(c)):
            print("Warning: c!=0 is not supported for Euclidean manifold. Setting it to 0.")
        if dtype == "float16":
            self.dtype = torch.float16
        elif dtype == "float32":
            self.dtype = torch.float32
        elif dtype == "float64":
            self.dtype = torch.float64
        else:
            raise ValueError(f"Unsupported dtype: {dtype}. Supported dtypes are float16, float32, and float64.")

    def _2manifold_dtype(self, xs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Convert the list of tensor(s) xs to the Euclidean manifold's dtype.

        Parameters
        ----------
        xs : List[torch.Tensor]
            List of tensor(s)

        Returns
        -------
        res : List[torch.Tensor]
            The list of tensor(s) converted to the Euclidean manifold's dtype
        """
        res = []
        for x in xs:
            res.append(x.to(self.dtype))
        return res

    def addition(self, x: torch.Tensor, y: torch.Tensor, dim: int=-1) -> torch.Tensor:
        """
        Add Euclidean manifold point(s) y to Euclidean manifold point(s) x.

        Parameters
        ----------
        x : torch.Tensor
            Euclidean manifold point(s)
        y : torch.Tensor
            Euclidean manifold point(s)
        dim : int
            Dimension along which to compute the addition (default: -1)
        Note: The dimension is not used in the Euclidean manifold, but it is included for consistency with other manifolds.

        Returns
        -------
        res : torch.Tensor
            The sum(s) of x and y
        """
        x, y = self._2manifold_dtype([x, y])
        res = x + y
        return res

    def scalar_mul(self, r: torch.Tensor, x: torch.Tensor, dim: int=-1) -> torch.Tensor:
        """
        Multiply Euclidean manifold point(s) x with scalar(s) r.

        Parameters
        ----------
        r : torch.Tensor
            Scalar factor(s)
        x : torch.Tensor
            Euclidean manifold point(s)
        dim : int
            Dimension along which to compute the multiplication (default: -1)
        Note: The dimension is not used in the Euclidean manifold, but it is included for consistency with other manifolds.

        Returns
        -------
        res : torch.Tensor
            The product(s) of r and x
        """
        x, r = self._2manifold_dtype([x, r])
        res = r * x
        return res

    def dist2hyperplane(self, x: torch.Tensor, a: torch.Tensor, p: torch.Tensor,
                        dim: int=-1, backproject: bool=True) -> torch.Tensor:
        """
        #TODO
        """
        pass

    def FC_forward(self, x: torch.Tensor, a: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Perform a fully connected forward pass with matrix a and bias p.

        Parameters
        ----------
        x : torch.Tensor (B, in_dim)
            Euclidean manifold point(s)
        a : torch.Tensor (out_dim, in_dim)
            (Euclidean) matrix
        p : torch.Tensor (1, out_dim)
            Euclidean manifold bias

        Returns
        -------
        res : torch.Tensor
            The forward propagated input(s) ax+b.
        """
        x, a, p = self._2manifold_dtype([x, a, p])
        res = (x.unsqueeze(-1) * a.T.unsqueeze(0)).sum(dim=1) # (B, out_dim)
        res = res + p # (B, out_dim)
        return res

    def MLR_forward(self, x: torch.Tensor, a: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Multinomial linear regressions score function.

        Parameters
        ----------
        x : torch.Tensor (B, in_dim)
            Euclidean manifold point(s)
        a : torch.Tensor (out_dim, in_dim)
            (Euclidean) matrix
        p : torch.Tensor (1, out_dim)
            Euclidean manifold bias

        Returns
        -------
        res : torch.Tensor (B, out_dim)
            The multinomial linear regression score(s) of x with respect to the linear model(s) defined by a and p.
        """
        # TODO: I presume this is equivalent to the MLR_forwards of HNN and HNN++
        res = self.FC_forward(x, a, p) # (B, out_dim)
        res = res / res.norm(p=2, dim=-1, keepdim=True) # (B, out_dim)
        return res

    def dist(self, x: torch.Tensor, y: torch.Tensor, dim: int=-1, backproject: bool=True) -> torch.Tensor:
        """
        Compute the geodesic distance(s) between Euclidean manifold points x and y.

        Parameters
        ----------
        x : torch.Tensor
            Euclidean manifold point(s)
        y : torch.Tensor
            Euclidean manifold point(s)
        dim : int
            Dimension along which to compute the distance (default: -1)
        backproject : bool
            Whether to project results back to the Euclidean manifold (default: True)
        Note: The backproject is not used in the Euclidean manifold, but it is included for consistency with other manifolds.

        Returns
        -------
        res : torch.Tensor
            The geodesic distance(s) between x and y
        """
        x, y = self._2manifold_dtype([x, y])
        res = (x - y).norm(p=2, dim=dim, keepdim=True)
        return res

    def dist_0(self, x: torch.Tensor, dim: int=-1) -> torch.Tensor:
        """
        Compute the geodesic distance(s) of Euclidean manifold point(s) x from/to the Euclidean origin.

        Parameters
        ----------
        x : torch.Tensor
            Euclidean manifold point(s)
        dim : int
            Dimension along which to compute the distance (default: -1)

        Returns
        -------
        res : torch.Tensor
            The geodesic distance(s) of x from/to the Euclidean origin
        """
        x, = self._2manifold_dtype([x])
        res = x.norm(p=2, dim=dim, keepdim=True)
        return res

    def expmap(self, v: torch.Tensor, x: torch.Tensor, dim: int=-1, backproject: bool=True) -> torch.Tensor:
        """
        Map tangent vector(s) v at Euclidean manifold point(s) x to the Euclidean manifold.
        [Exponential map]

        Parameters
        ----------
        v : torch.Tensor
            Vector(s) in the tangent space(s) of x
        x : torch.Tensor
            Euclidean manifold point(s)
        dim : int
            Dimension along which to compute the exponential map (default: -1)
        backproject : bool
            Whether to project results back to the Euclidean manifold (default: True)
        Note: The dimension and backprojection are not used in the Euclidean manifold,
              but they are included for consistency with other manifolds.

        Returns
        -------
        res : torch.Tensor
            The point(s) after mapping v to the Euclidean manifold
        """
        v, x = self._2manifold_dtype([v, x])
        res = x + v
        return res

    def expmap_0(self, v: torch.Tensor, dim: int=-1, backproject: bool=True) -> torch.Tensor:
        """
        Map tangent vector(s) v at the Euclidean origin to the Euclidean manifold.
        [Exponential map]

        Parameters
        ----------
        v : torch.Tensor
            Vector(s) in the tangent space of the Euclidean origin
        dim : int
            Dimension along which to compute the exponential map (default: -1)
        backproject : bool
            Whether to project results back to the Euclidean manifold (default: True)
        Note: The dimension and backprojection are not used in the Euclidean manifold,
              but they are included for consistency with other manifolds.

        Returns
        -------
        res : torch.Tensor
            The point(s) after mapping v to the Euclidean manifold
        """
        v, = self._2manifold_dtype([v])
        res = v
        return res

    def retraction(self, v: torch.Tensor, x: torch.Tensor, dim: int=-1, backproject: bool=True) -> torch.Tensor:
        """
        First-order approximation of the exponential map for vector(s) v at Euclidean manifold point(s) x.
        [Retraction map]

        Parameters
        ----------
        v : torch.Tensor
            Vector(s) in the tangent space(s) of x
        x : torch.Tensor
            Euclidean manifold point(s)
        dim : int
            Dimension along which to compute the retraction (default: -1)
        backproject : bool
            Whether to project results back to the Euclidean manifold (default: True)
        Note: The dimension and backprojection are not used in the Euclidean manifold,
              but they are included for consistency with other manifolds.

        Returns
        -------
        res : torch.Tensor
            The point(s) after approximately mapping v to the Euclidean manifold
        """
        v, x = self._2manifold_dtype([v, x])
        res = x + v
        return res

    def logmap(self, y: torch.Tensor, x: torch.Tensor, dim: int=-1) -> torch.Tensor:
        """
        Map Euclidean manifold point(s) y to the tangent space(s) of Euclidean manifold point(s) x.
        [Logarithmic map]

        Parameters
        ----------
        y : torch.Tensor
            Euclidean manifold point(s)
        x : torch.Tensor
            Euclidean manifold point(s)
        dim : int
            Dimension along which to compute the logarithmic map (default: -1)
        Note: The dimension is not used in the Euclidean manifold, but it is included for consistency with other manifolds.

        Returns
        -------
        res : torch.Tensor
            The resulting tangent vector(s) after mapping y to the tangent space(s) of x
        """
        y, x = self._2manifold_dtype([y, x])
        res = y - x
        return res

    def logmap_0(self, y: torch.Tensor, dim: int=-1) -> torch.Tensor:
        """
        Map Euclidean manifold point(s) y to the tangent space of the Euclidean origin.
        [Logarithmic map]

        Parameters
        ----------
        y : torch.Tensor
            Euclidean manifold point(s)
        dim : int
            Dimension along which to compute the logarithmic map (default: -1)

        Returns
        -------
        res : torch.Tensor
            The resulting tangent vector(s) after mapping y to the tangent space of the origin
        """
        y, = self._2manifold_dtype([y])
        res = y
        return res

    def ptransp(self, v: torch.Tensor, x: torch.Tensor, y: torch.Tensor, dim: int=-1) -> torch.Tensor:
        """
        Parallel transport tangent vector(s) v from the tangent space(s) of Euclidean
        manifold point(s) x to the tangent space(s) of Euclidean manifold point(s) y.

        Parameters
        ----------
        v : torch.Tensor
            Vector(s) in the tangent space(s) of x
        x : torch.Tensor
            Euclidean manifold point(s)
        y : torch.Tensor
            Euclidean manifold point(s)
        dim : int
            Dimension along which to compute the parallel transport (default: -1)
        Note: The dimension is not used in the Euclidean manifold, but it is included for consistency with other manifolds.

        Returns
        -------
        res : torch.Tensor
            The parallel transported tangent vector(s)
        """
        v, = self._2manifold_dtype([v])
        res = v
        return res

    def ptransp_0(self, v: torch.Tensor, y: torch.Tensor, dim: int=-1) -> torch.Tensor:
        """
        Parallel transport tangent vector(s) v from the tangent space of
        the Euclidean origin to the tangent space(s) of Euclidean manifold point(s) y.

        Parameters
        ----------
        v : torch.Tensor
            Vector(s) in the tangent space of the Euclidean origin
        y : torch.Tensor
            Euclidean manifold point(s)
        dim : int
            Dimension along which to compute the parallel transport (default: -1)
        Note: The dimension is not used in the Euclidean manifold, but it is included for consistency with other manifolds.

        Returns
        -------
        res : torch.Tensor
            The parallel transported tangent vector(s)
        """
        v, = self._2manifold_dtype([v])
        res = v
        return res

    def tangent_inner(self, u: torch.Tensor, v: torch.Tensor, x: torch.Tensor, dim: int=-1) -> torch.Tensor:
        """
        Compute the inner product(s) between tangent vectors u and v of the tangent space(s)
        at Euclidean manifold point(s) x with respect to the Riemannian metric of the Euclidean manifold.

        Parameters
        ----------
        u : torch.Tensor
            Vector(s) in the tangent space(s) of x
        v : torch.Tensor
            Vector(s) in the tangent space(s) of x
        x : torch.Tensor
            Euclidean manifold point(s)
        dim : int
            Dimension along which to compute the tangent inner product (default: -1)

        Returns
        -------
        res : torch.Tensor
            The tangent inner product(s) of u and v
        """
        u, v = self._2manifold_dtype([u, v])
        res = (u * v).sum(dim=dim, keepdim=True)
        return res

    def tangent_norm(self, v: torch.Tensor, x: torch.Tensor, dim: int=-1) -> torch.Tensor:
        """
        Compute the norm(s) of tangent vector(s) v of the tangent space(s) at Euclidean manifold point(s) x
        with respect to the Riemannian metric of the Euclidean manifold.

        Parameters
        ----------
        v : torch.Tensor
            Vector(s) in the tangent space(s) of x
        x : torch.Tensor
            Euclidean manifold point(s)
        dim : int
            Dimension along which to compute the tangent norm (default: -1)

        Returns
        -------
        res : torch.Tensor
            The tangent norm(s) of v
        """
        v, = self._2manifold_dtype([v])
        res = v.norm(p=2, dim=dim, keepdim=True)
        return res

    def egrad2rgrad(self, grad: torch.Tensor, x: torch.Tensor, dim: int=-1) -> torch.Tensor:
        """
        Compute the Riemannian gradient(s) at Euclidean manifold point(s) x from the Euclidean gradient(s).

        Parameters
        ----------
        grad : torch.Tensor
            Euclidean gradient(s)
        x : torch.Tensor
            Euclidean manifold point(s)
        dim : int
            Dimension along which to compute the Riemannian gradient (default: -1)

        Returns
        -------
        res : torch.Tensor
            The Riemannian gradient(s) at x
        """
        res = grad
        return res

    def proj(self, x: torch.Tensor, dim: int=-1):
        """
        Project point(s) x onto the Euclidean manifold.

        Parameters
        ----------
        x : torch.Tensor
            Point(s)
        dim : int
            Dimension along which to compute the projection (default: -1)
        Note: The dimension is not used in the Euclidean manifold, but it is included for consistency with other manifolds.

        Returns
        -------
        res : torch.Tensor
            The projected Euclidean manifold point(s)
        """
        x, = self._2manifold_dtype([x])
        res = x
        return res

    def is_in_manifold(self, x: torch.Tensor, dim: int=-1) -> bool:
        """
        Check if point(s) x lie in the Euclidean manifold.

        Parameters
        ----------
        x : torch.Tensor
            Euclidean manifold point(s)
        dim : int
            Dimension along which to check if x lies in the Euclidean manifold (default: -1)
        Note: The dimension is not used in the Euclidean manifold, but it is included for consistency with other manifolds.

        Returns
        -------
        res : bool
            True if all points x lie in the Euclidean manifold, False otherwise
        """
        res = True
        return res

    def is_in_tangent_space(self, v: torch.Tensor, x: torch.Tensor, dim: int=-1) -> bool:
        """
        Check if vector(s) v belong to the tangent space(s) at Euclidean manifold point(s) x.

        Parameters
        ----------
        v : torch.Tensor
            Vector(s)
        x : torch.Tensor
            Euclidean manifold point(s)
        dim : int
            Dimension along which to check if v belongs to the tangent space (default: -1)
        Note: The dimension is not used in the Euclidean manifold, but it is included for consistency with other manifolds.

        Returns
        -------
        res : bool
            True if all ectors v belong to their tangent spaces, False otherwise
        """
        res = True
        return res
