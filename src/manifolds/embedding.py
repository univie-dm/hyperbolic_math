import torch

from .manifold import ManifoldParameter, Manifold


class Embedding(torch.nn.Module):
    """
    Embedding layer that supports different manifolds.
    """

    manifold: Manifold
    c: torch.Tensor
    weight: ManifoldParameter
    bias: ManifoldParameter

    def __init__(self, input_dim: int, output_dim: int, manifold: str, c: float, requires_grad: bool = True):
        super().__init__()
        self.manifold = manifold
        self.c = torch.tensor(c, dtype=torch.float32)  # Curvature for non-Euclidean manifolds

        weight = torch.randn(input_dim, output_dim)
        self.weight = ManifoldParameter(weight, requires_grad=requires_grad, manifold=self.manifold, c=self.c)

        bias = torch.zeros(output_dim)
        self.bias = ManifoldParameter(bias, requires_grad=requires_grad, manifold=self.manifold, c=self.c)

    def forward(self, x):
        # We always need a bias addition, else it's just equivalent to a Euclidean NN
        # TODO: Alternative bias translation: expmap(ptransp_0(bias, x), x, c) -- Stability??

        # Method 1: matvec_mul == expmap_0(W*logmap_0(x))
        # x = self.manifold.expmap_0(x, self.c)
        # x = self.manifold.matvec_mul(self.weight, x, self.c)
        # x = self.manifold.addition(x, self.bias, self.c)

        # Method 2 - less computation since expmap_0(logmap_0(x)==x), hence theoretically more stable - test in experiments:
        x = self.manifold.expmap_0(x @ self.weight, self.c)
        x = self.manifold.addition(x, self.bias, self.c)

        return x
