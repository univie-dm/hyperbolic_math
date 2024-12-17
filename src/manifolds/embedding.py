import torch

from .manifold import Manifold, ManifoldParameter
from .poincare import PoincareBall


class Embedding(torch.nn.Module):
    """
    Embedding layer that supports different manifolds.
    """

    manifold: Manifold
    c: torch.Tensor
    weight: ManifoldParameter
    bias: ManifoldParameter

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        manifold: str,
        c: float,
        requires_grad: bool = True,
        forward_method: str = "dist2hyperplane",
    ):
        super().__init__()
        self.manifold = manifold
        self.c = torch.tensor(c, dtype=torch.float32)  # Curvature for non-Euclidean manifolds
        self.forward_method = forward_method

        weight = torch.randn(input_dim, output_dim)
        self.weight = torch.nn.Parameter(weight, requires_grad=requires_grad)

        bias = torch.zeros(input_dim)
        self.bias = ManifoldParameter(bias, requires_grad=requires_grad, manifold=self.manifold, c=self.c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.manifold.is_in_manifold(self.bias, self.c) # check if bias is in manifold

        # Method 1: forward pass via dist2hyperplane - only works for PoincareBall
        if self.forward_method == "dist2hyperplane" and isinstance(self.manifold, PoincareBall):
            x = self.manifold.expmap_0(x, self.c)
            res = self.manifold.dist2hyperplane(x, self.weight, self.bias, self.c, signed=True, scaled=True)
            #TODO: Technically we are missing some scalings here with sign(.) * ||weights||

        # Method 2: forward pass via dist2hyperplane++ - only works for PoincareBall
        if self.forward_method == "dist2hyperplane++" and isinstance(self.manifold, PoincareBall):
            x = self.manifold.expmap_0(x, self.c)
            res = self.manifold.dist2hyperplane_pp(x, self.weight, self.bias, self.c, signed=True, scaled=True)

        # Method 3 - forward pass via tangent space matrix-vector mult. - more stable than "indirect_matvec_mul"
        elif self.forward_method == "direct_matvec_mul":
            x = self.manifold.expmap_0(x @ self.weight, self.c)
            res = self.manifold.addition(x, self.bias, self.c)

        # Method 4 - forward pass via tangent space matrix-vector mult. but with logmap/expmap transforms
        elif self.forward_method == "indirect_matvec_mul":
            x = self.manifold.expmap_0(x, self.c)
            x = self.manifold.matvec_mul(self.weight, x, self.c)
            res = self.manifold.addition(x, self.bias, self.c)
        
        # Method 5 - forward pass as in hyperbolic neural networks
        elif self.forward_method == "hnn_matvec_mul":
            x = self.manifold.expmap_0(x @ self.weight, self.c)
            x = self.manifold.matvec_mul(self.weight, x, self.c)
            ebias = self.manifold.ptransp_0(self.manifold.logmap_0(self.bias, self.c), x, self.c)
            res = self.manifold.expmap(ebias, x, self.c)

        return res
