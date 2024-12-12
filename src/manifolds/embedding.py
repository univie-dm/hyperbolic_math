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
    rescale_normal: bool
    out_downscale: float

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        manifold: str,
        c: float,
        rescale_normal: bool,
        out_downscale: float,
        requires_grad: bool = True,
    ):
        super().__init__()
        self.manifold = manifold
        self.c = torch.tensor(c, dtype=torch.float32)  # Curvature for non-Euclidean manifolds
        self.rescale_normal = rescale_normal
        self.out_downscale = out_downscale

        # Init as transpose to make math easier
        weight = torch.randn(output_dim, input_dim)
        self.weight = torch.nn.Parameter(weight, requires_grad=requires_grad)  # , manifold=self.manifold, c=self.c

        # Init as transpose to make math easier
        bias = torch.zeros(output_dim, input_dim)
        self.bias = ManifoldParameter(bias, requires_grad=requires_grad, manifold=self.manifold, c=self.c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: Only works for Poincare ball right now
        # Assume scaled Euclidean inputs and map them to the Poincare ball with
        x = self.manifold.expmap_0(x, self.c)
        # Add extra intermediate dimension for broadcasting in dist2plane Möbius addition
        x = x[:, None, :]

        # Rescale normal (weight) parameters
        if self.rescale_normal:
            conformal_factor = 1 - self.bias.pow(2).sum(dim=-1)
            scaled_normal = self.weight * conformal_factor.unsqueeze(-1)
        conformal_factor = 1 - self.bias.pow(2).sum(dim=-1)
        scaled_normal = self.weight * conformal_factor.unsqueeze(-1)

        # Implement linear layer as signed distance to hyperplane
        assert isinstance(self.manifold, PoincareBall), "Only PoincareBall is supported for now"
        # TODO: How can I still use this with everything being transposed?
        # TODO: Why the fuck can't I make the normal a manifold parameter?
        res = self.manifold.dist2plane(x, scaled_normal, self.bias, self.c, signed=True, scaled=True)
        res = res.squeeze()

        # Rescale distances when using re-scaling of normal parameters
        if self.rescale_normal:
            res_scaled = res * 2 / conformal_factor

        # Return downscaled logits
        return res_scaled * self.out_downscale
