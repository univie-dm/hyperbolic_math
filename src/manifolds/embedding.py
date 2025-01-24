from typing import Literal, get_args

import torch

from .manifold import Manifold, ManifoldParameter
from .poincare import PoincareBall

ForwardPassType = Literal[
    "dist2hyperplane", "dist2hyperplane_correct", "dist2hyperplane_pp", "direct_matvec_mul", "indirect_matvec_mul", "hnn_matvec_mul"
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

        if forward_method in {"dist2hyperplane", "dist2hyperplane_correct", "dist2hyperplane_pp"}:
            bias = torch.zeros(input_dim)
        elif forward_method in {"direct_matvec_mul", "indirect_matvec_mul", "hnn_matvec_mul"}:
            bias = torch.zeros(output_dim)
        self.bias = ManifoldParameter(bias, requires_grad=requires_grad, manifold=self.manifold, c=self.c)

    def _sanity_checks(self, forward_method: ForwardPassType) -> None:
        """Sanity checks to ensure correct initialization and forward method"""

        assert forward_method in get_args(ForwardPassType), f"Invalid forward method: {forward_method}"
        if forward_method in {"dist2hyperplane", "dist2hyperplane_correct", "dist2hyperplane++"}:
            assert isinstance(self.manifold, PoincareBall), "dist2hyperplane methods only work for PoincareBall manifold"
        if forward_method == "dist2hyperplane_correct":
            assert False, "Not implemented yet"

    def forward_dist2hyperplane(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass via dist2hyperplane:
           1) Perform Matrix-Vector multiplication in the tangent space of the bias.
           2) Represent the full affine function from the viewpoint of the hyperplane mx + b = 0.
        [Only works for the PoincareBall]
        """
        assert self.manifold.is_in_manifold(self.bias, self.c)

        x = self.manifold.expmap_0(x, self.c)
        res = self.manifold.dist2hyperplane(x, self.weight, self.bias, self.c, signed=True, scaled=True)
        return res

    def forward_dist2hyperplane_correct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass via forward_dist2hyperplane_correct:
           1) Parallel transport the weight to the tangent space of the bias.
           2) Perform Matrix-Vector multiplication in the tangent space of the bias.
           3) Represent the full affine function from the viewpoint of the hyperplane mx + b = 0.
        [Only works for the PoincareBall]
        """
        assert self.manifold.is_in_manifold(self.bias, self.c)

        tangent_space_weight = self.manifold.ptransp_0(self.weight, self.bias, self.c)
        x = self.manifold.expmap_0(x, self.c)
        res = self.manifold.dist2hyperplane_correct(x, tangent_space_weight, self.bias, self.c)
        return res

    def forward_dist2hyperplane_pp(self, x: torch.Tensor) -> torch.Tensor:
        """
        #TODO
        Forward pass via dist2hyperplane++:
           1) Perform Matrix-Vector multiplication in the tangent space and map back to the manifold.
           2) Add the manifold bias to the previous result.
        [Only works for the PoincareBall]
        """
        assert False, "Not implemented yet"

        assert self.manifold.is_in_manifold(self.bias, self.c)

        x = self.manifold.expmap_0(x, self.c)
        res = self.manifold.dist2hyperplane_pp(x, self.weight, self.bias, self.c, signed=True, scaled=True)
        return res

    def forward_direct_matvec_mul(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass via tangent space matrix-vector mult. without logmap/expmap transforms:
           1) Perform Matrix-Vector multiplication in the tangent space and map back to the manifold.
           2) Add the manifold bias to the previous result.
        """
        assert self.manifold.is_in_manifold(self.bias, self.c)

        x = self.manifold.expmap_0(x @ self.weight, self.c)
        res = self.manifold.addition(x, self.bias, self.c)
        return res

    def forward_indirect_matvec_mul(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass via tangent space matrix-vector mult. but with logmap/expmap transforms:
           1) Map input x from the tangent space to the manifold.
           2) Perform Matrix-Vector multiplication in the tangent space and map back to the manifold.
           3) Add the manifold bias to the previous result.
        """
        assert self.manifold.is_in_manifold(self.bias, self.c)

        x = self.manifold.expmap_0(x, self.c)
        x = self.manifold.matvec_mul(self.weight, x, self.c)
        res = self.manifold.addition(x, self.bias, self.c)
        return res

    def forward_hnn_matvec_mul(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass as in hyperbolic neural networks paper:
           1) Perform Matrix-Vector multiplication in the tangent space and map back to the manifold.
           2) Add the manifold bias to the previous result via parallel transport.
           3) Map the result back to the manifold.
        """
        assert self.manifold.is_in_manifold(self.bias, self.c)

        x = self.manifold.expmap_0(x @ self.weight, self.c)
        bias = self.manifold.ptransp_0(self.manifold.logmap_0(self.bias, self.c), x, self.c)
        res = self.manifold.expmap(bias, x, self.c)
        return res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_method(x)
