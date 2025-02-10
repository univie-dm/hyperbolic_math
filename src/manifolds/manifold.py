"""Base manifold."""

import torch

from typing import Union


class Manifold:
    """Abstract manifold class."""

    def __init__(self, c: torch.Tensor=1., trainable_c: bool=False):
        if trainable_c:
            self.c = torch.nn.Parameter(c)
        else:
            self.c = c
        self.min_enorm = None
        self.max_enorm_eps = None

    def addition(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Add manifold point(s) y to manifold point(s) x."""
        raise NotImplementedError

    def scalar_mul(self, r: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Multiply manifold point(s) x with scalar(s) r."""
        raise NotImplementedError

    def matvec_mul(self, m: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Multiply manifold point(s) x with matrix m from the left."""
        raise NotImplementedError
    
    def hyperplane_forward(self, x: torch.Tensor, m: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Perform the hyperplane forward pass."""
        raise NotImplementedError

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the geodesic distance(s) between manifold points x and y."""
        raise NotImplementedError

    def dist_0(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the geodesic distance(s) of manifold point(s) x from/to the manifold's origin."""
        raise NotImplementedError

    def expmap(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Map tangent vector(s) v at manifold point(s) x to the manifold. [Exponential map]"""
        raise NotImplementedError

    def expmap_0(self, v: torch.Tensor) -> torch.Tensor:
        """Map tangent vector(s) v at the manifold's origin to the manifold. [Exponential map]"""
        raise NotImplementedError

    def retraction(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Approximately map tangent vector(s) v at manifold point(s) x to the manifold. [Retraction map]"""
        raise NotImplementedError

    def logmap(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Map manifold point(s) y to the tangent space(s) of manifold point(s) x. [Logarithmic map]"""
        raise NotImplementedError

    def logmap_0(self, y: torch.Tensor) -> torch.Tensor:
        """Map manifold point(s) y to the tangent space of the manifold's origin. [Logarithmic map]"""
        raise NotImplementedError

    def ptransp(self, v: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Parallel transport tangent vector(s) v from the tangent space(s) of
           manifold point(s) x to the tangent space(s) of manifold point(s) y."""
        raise NotImplementedError

    def ptransp_0(self, v: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Parallel transport tangent vector(s) v from the tangent space of
           the manifold's origin to the tangent space(s) of manifold point(s) y."""
        raise NotImplementedError

    def tangent_inner(self, u: torch.Tensor, v: Union[torch.Tensor, None], x: torch.Tensor) -> torch.Tensor:
        """Compute the inner product(s) between tangent vectors u and v of the tangent space(s)
           at manifold point(s) x with respect to the Riemannian metric of the manifold."""
        raise NotImplementedError

    def tangent_norm(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute the norm(s) of tangent vector(s) v of the tangent space(s) at manifold point(s) x
           with respect to the Riemannian metric of the manifold."""
        raise NotImplementedError

    def egrad2rgrad(self, grad: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute the Riemannian gradient(s) at manifold point(s) x from the Euclidean gradient(s)."""
        raise NotImplementedError

    def proj(self, x: torch.Tensor):
        """Project point(s) x onto the clipped manifold."""
        raise NotImplementedError

    def is_in_manifold(self, x: torch.Tensor) -> bool:
        """Check if point(s) x lie on the manifold."""
        raise NotImplementedError

    def is_in_tangent_space(self, v: torch.Tensor, x: torch.Tensor) -> bool:
        """Check if vector(s) v belong to the tangent space(s) at manifold point(s) x."""
        raise NotImplementedError


class ManifoldParameter(torch.nn.Parameter):
    """Subclass of torch.nn.Parameter for Riemannian optimization."""

    def __new__(cls, data, requires_grad, manifold):
        return torch.nn.Parameter.__new__(cls, data, requires_grad)

    def __init__(self, data, requires_grad, manifold):
        self.manifold = manifold

    def __repr__(self):
        return f"{self.manifold.name} Parameter containing:\n" + super(torch.nn.Parameter, self).__repr__()
