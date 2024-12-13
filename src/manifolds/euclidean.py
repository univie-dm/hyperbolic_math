"""Euclidean manifold."""

import torch

from .manifold import Manifold


class Euclidean(Manifold):
    """
    Euclidean manifold class.
    """

    def __init__(self):
        super(Euclidean, self).__init__()
        self.name = "Euclidean"

    def addition(self, x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Add Euclidean manifold point(s) y to Euclidean manifold point(s) x.

        Parameters
        ----------
        x : torch.Tensor
            Euclidean manifold point(s)
        y : torch.Tensor
            Euclidean manifold point(s)
        c : torch.Tensor
            magnitude of sectional curvature

        Returns
        -------
        res : torch.Tensor
            The sum(s) of x and y
        """
        res = x + y
        return res

    def scalar_mul(self, r: torch.Tensor, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Multiply Euclidean manifold point(s) x with scalar(s) r.

        Parameters
        ----------
        r : torch.Tensor
            scalar factor(s)
        x : torch.Tensor
            Euclidean manifold point(s)
        c : torch.Tensor
            magnitude of sectional curvature

        Returns
        -------
        res : torch.Tensor
            The product(s) of r and x
        """
        res = r * x
        return res

    def matvec_mul(self, m: torch.Tensor, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Multiply Euclidean manifold point(s) x with (Euclidean) matrix m from the left.

        Parameters
        ----------
        m : torch.Tensor
            (Euclidean) matrix
        x : torch.Tensor
            Euclidean manifold point(s)
        c : torch.Tensor
            magnitude of sectional curvature

        Returns
        -------
        res : torch.Tensor
            The product(s) of m and x
        """
        res = x @ m
        return res

    def dist(self, x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Compute the geodesic distance(s) between Euclidean manifold points x and y.

        Parameters
        ----------
        x : torch.Tensor
            Euclidean manifold point(s)
        y : torch.Tensor
            Euclidean manifold point(s)
        c : torch.Tensor
            magnitude of sectional curvature

        Returns
        -------
        res : torch.Tensor
            The geodesic distance(s) between x and y
        """
        res = (x - y).norm(p=2, dim=-1, keepdim=True)
        return res

    def dist_0(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Compute the geodesic distance(s) of Euclidean manifold point(s) x from/to the Euclidean origin.

        Parameters
        ----------
        x : torch.Tensor
            Euclidean manifold point(s)
        c : torch.Tensor
            magnitude of sectional curvature

        Returns
        -------
        res : torch.Tensor
            The geodesic distance(s) of x from/to the Euclidean origin
        """
        res = x.norm(p=2, dim=-1, keepdim=True)
        return res

    def expmap(self, v: torch.Tensor, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Map tangent vector(s) v at Euclidean manifold point(s) x to the Euclidean manifold. [Exponential map]

        Parameters
        ----------
        v : torch.Tensor
            vector(s) in the tangent space(s) of x
        x : torch.Tensor
            Euclidean manifold point(s)
        c : torch.Tensor
            magnitude of sectional curvature

        Returns
        -------
        res : torch.Tensor
            The point(s) after mapping v to the Euclidean manifold
        """
        res = x + v
        return res

    def expmap_0(self, v: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Map tangent vector(s) v at the Euclidean origin to the Euclidean manifold. [Exponential map]

        Parameters
        ----------
        v : torch.Tensor
            vector(s) in the tangent space of the Euclidean origin
        c : torch.Tensor
            magnitude of sectional curvature

        Returns
        -------
        res : torch.Tensor
            The point(s) after mapping v to the Euclidean manifold
        """
        res = v
        return res

    def retraction(self, v: torch.Tensor, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        First-order approximation of the exponential map for vector(s) v at manifold point(s) x.
        [Retraction map]

        Parameters
        ----------
        v : torch.Tensor
            vector(s) in the tangent space(s) of x
        x : torch.Tensor
            Euclidean manifold point(s)
        c : torch.Tensor
            magnitude of sectional curvature

        Returns
        -------
        res : torch.Tensor
            The point(s) after approximately mapping v to the Euclidean manifold
        """
        res = x + v
        return res

    def logmap(self, y: torch.Tensor, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Map Euclidean manifold point(s) y to the tangent space(s) of Euclidean manifold point(s) x. [Logarithmic map]

        Parameters
        ----------
        y : torch.Tensor
            Euclidean manifold point(s)
        x : torch.Tensor
            Euclidean manifold point(s)
        c : torch.Tensor
            magnitude of sectional curvature

        Returns
        -------
        res : torch.Tensor
            The resulting tangent vector(s) after mapping y to the tangent space(s) of x
        """
        res = y - x
        return res

    def logmap_0(self, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Map Euclidean manifold point(s) y to the tangent space of the Euclidean origin. [Logarithmic map]

        Parameters
        ----------
        y : torch.Tensor
            Euclidean manifold point(s)
        c : torch.Tensor
            magnitude of sectional curvature

        Returns
        -------
        res : torch.Tensor
            The resulting tangent vector(s) after mapping y to the tangent space of the origin
        """
        res = y
        return res

    def ptransp(self, v: torch.Tensor, x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Parallel transport tangent vector(s) v from the tangent space(s) of Euclidean
        manifold point(s) x to the tangent space(s) of Euclidean manifold point(s) y.

        Parameters
        ----------
        v : torch.Tensor
            vector(s) in the tangent space(s) of x
        x : torch.Tensor
            Euclidean manifold point(s)
        y : torch.Tensor
            Euclidean manifold point(s)
        c : torch.Tensor
            magnitude of sectional curvature

        Returns
        -------
        res : torch.Tensor
            The parallel transported tangent vector(s)
        """
        res = v
        return res

    def ptransp_0(self, v: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Parallel transport tangent vector(s) v from the tangent space of
        the Euclidean origin to the tangent space(s) of Euclidean manifold point(s) y.

        Parameters
        ----------
        v : torch.Tensor
            vector(s) in the tangent space of the Euclidean origin
        y : torch.Tensor
            Euclidean manifold point(s)
        c : torch.Tensor
            magnitude of sectional curvature

        Returns
        -------
        res : torch.Tensor
            The parallel transported tangent vector(s)
        """
        res = v
        return res

    def tangent_inner(self, u: torch.Tensor, v: torch.Tensor, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Compute the inner product(s) between tangent vectors u and v of the tangent space(s)
        at Euclidean manifold point(s) x with respect to the Riemannian metric of the Euclidean manifold.

        Parameters
        ----------
        u : torch.Tensor
            vector(s) in the tangent space(s) of x
        v : torch.Tensor
            vector(s) in the tangent space(s) of x
        x : torch.Tensor
            Euclidean manifold point(s)
        c : torch.Tensor
            magnitude of sectional curvature

        Returns
        -------
        res : torch.Tensor
            The tangent inner product(s) of u and v
        """
        if v is u:
            res = u.pow(2)
        else:
            res = u * v
        res = res.sum(dim=-1, keepdim=True)
        return res

    def tangent_norm(self, v: torch.Tensor, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Compute the norm(s) of tangent vector(s) v of the tangent space(s) at Euclidean manifold point(s) x
        with respect to the Riemannian metric of the Euclidean manifold.

        Parameters
        ----------
        v : torch.Tensor
            vector(s) in the tangent space(s) of x
        x : torch.Tensor
            Euclidean manifold point(s)
        c : torch.Tensor
            magnitude of sectional curvature

        Returns
        -------
        res : torch.Tensor
            The tangent norm(s) of v
        """
        res = v.norm(p=2, dim=-1, keepdim=True)
        return res

    def egrad2rgrad(self, grad: torch.Tensor, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Compute the Riemannian gradient(s) at Euclidean manifold point(s) x from the Euclidean gradient(s).

        Parameters
        ----------
        grad : torch.Tensor
            Euclidean gradient(s)
        x : torch.Tensor
            Euclidean manifold point(s)
        c : torch.Tensor
            magnitude of sectional curvature

        Returns
        -------
        res : torch.Tensor
            The Riemannian gradient(s) at x
        """
        res = grad
        return res

    def proj(self, x: torch.Tensor, c: torch.Tensor):
        """
        Project point(s) x onto the Euclidean manifold.

        Parameters
        ----------
        x : torch.Tensor
            point(s)
        c : torch.Tensor
            magnitude of sectional curvature

        Returns
        -------
        res : torch.Tensor
            The projected Euclidean manifold point(s)
        """
        res = x
        return res

    def is_in_manifold(self, x: torch.Tensor, c: torch.Tensor) -> bool:
        """
        Check if point(s) x lie in the Euclidean manifold.

        Parameters
        ----------
        x : torch.Tensor
            Euclidean manifold point(s)
        c : torch.Tensor
            magnitude of sectional curvature

        Returns
        -------
        res : bool
            True if all points x lie in the Euclidean manifold, False otherwise
        """
        res = True
        return res
