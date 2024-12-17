from typing import Literal, get_args

import torch

from .manifold import Manifold, ManifoldParameter
from .poincare import PoincareBall

ForwardPassType = Literal[
    "dist2hyperplane", "dist2hyperplane_pp", "direct_matvec_mul", "indirect_matvec_mul", "hnn_matvec_mul"
]


class Embedding(torch.nn.Module):
    """
    Embedding layer that supports different manifolds.
    """

    manifold: Manifold
    c: torch.Tensor
    weight: ManifoldParameter
    bias: ManifoldParameter
    forward_method: ForwardPassType

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        manifold: Manifold,
        c: float,
        requires_grad: bool = True,
        forward_method: ForwardPassType = "dist2hyperplane",
    ):
        super().__init__()
        self.manifold = manifold
        self.register_buffer("c", torch.tensor(c, dtype=torch.float32))  # Curvature for non-Euclidean manifolds

        self._sanity_checks(forward_method)
        # NOTE: Assumes that method names are of the form "forward_{name}", where name specifies the forward pass method
        self.forward_method = getattr(self, f"forward_{forward_method}")

        weight = torch.randn(input_dim, output_dim)
        self.weight = torch.nn.Parameter(weight, requires_grad=requires_grad)

        bias = torch.zeros(input_dim)
        self.bias = ManifoldParameter(bias, requires_grad=requires_grad, manifold=self.manifold, c=self.c)

    def _sanity_checks(self, forward_method: ForwardPassType) -> None:
        """Sanity checks to ensure correct initialization and forward method"""

        assert forward_method in get_args(ForwardPassType), f"Invalid forward method: {forward_method}"
        if forward_method in {"dist2hyperplane", "dist2hyperplane++"}:
            assert isinstance(self.manifold, PoincareBall), "dist2hyperplane methods only work for PoincareBall manifold"

    def forward_dist2hyperplane(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass via dist2hyperplane - only works for PoincareBall"""
        x = self.manifold.expmap_0(x, self.c)
        res = self.manifold.dist2hyperplane(x, self.weight, self.bias, self.c, signed=True, scaled=True)
        return res

    def forward_dist2hyperplane_pp(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass via dist2hyperplane++ - only works for PoincareBall"""
        x = self.manifold.expmap_0(x, self.c)
        res = self.manifold.dist2hyperplane_pp(x, self.weight, self.bias, self.c, signed=True, scaled=True)
        return res

    def forward_direct_matvec_mul(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass via tangent space matrix-vector mult. - more stable than "indirect_matvec_mul" """
        x = self.manifold.expmap_0(x @ self.weight, self.c)
        res = self.manifold.addition(x, self.bias, self.c)
        return res

    def forward_indirect_matvec_mul(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass via tangent space matrix-vector mult. but with logmap/expmap transforms"""
        x = self.manifold.expmap_0(x, self.c)
        x = self.manifold.matvec_mul(self.weight, x, self.c)
        res = self.manifold.addition(x, self.bias, self.c)
        return res

    def forward_hnn_matvec_mul(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass as in hyperbolic neural networks"""
        x = self.manifold.expmap_0(x @ self.weight, self.c)
        x = self.manifold.matvec_mul(self.weight, x, self.c)
        ebias = self.manifold.ptransp_0(self.manifold.logmap_0(self.bias, self.c), x, self.c)
        res = self.manifold.expmap(ebias, x, self.c)
        return res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check if bias is in manifold
        assert self.manifold.is_in_manifold(self.bias, self.c)

        return self.forward_method(x)
