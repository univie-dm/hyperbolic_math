import torch

from typing import Literal, get_args
from . import Manifold, ManifoldParameter, Euclidean, Hyperboloid, PoincareBall


ForwardPassType = Literal[
    None,
    "hyperplane_forward",
    "hyperplane_forward_correct",
    "hyperplane_forward_pp",
    "matvec_mul",
    "hnn_matvec_mul"
]


class Embedding(torch.nn.Module):
    """
    Embedding layer that supports different manifolds.
    """
    manifold: Manifold
    weight: ManifoldParameter
    bias: ManifoldParameter
    forward_method: ForwardPassType

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        manifold: Manifold,
        requires_grad: bool = True,
        forward_method: ForwardPassType = None,
    ):
        super().__init__()
        self.manifold = manifold
        if ForwardPassType:
            self._sanity_checks(forward_method)
            # NOTE: Assumes that method names are of the form "forward_{name}", where name specifies the forward pass method
            self.forward_method = getattr(self, f"forward_{forward_method}")
        # Defaults
        elif isinstance(self.manifold, Euclidean):
            self.forward_method = getattr(self, "forward_matvec_mul")
        elif isinstance(self.manifold, Hyperboloid):
            assert False, "Not implemented yet"
        else:  # PoincareBall
            self.forward_method = getattr(self, "forward_hyperplane_forward")

        weight = torch.randn(input_dim, output_dim)
        self.weight = torch.nn.Parameter(weight, requires_grad=requires_grad)

        if forward_method in {"matvec_mul", "hnn_matvec_mul"} or isinstance(self.manifold, Euclidean):
            bias = torch.zeros(output_dim)
        else: # PoincareBall hyperplane forward methods
            bias = torch.zeros(input_dim)
        self.bias = ManifoldParameter(bias, requires_grad=requires_grad, manifold=self.manifold)

    def _sanity_checks(self, forward_method: ForwardPassType) -> None:
        """Sanity checks to ensure correct initialization and forward method"""
        assert forward_method in get_args(ForwardPassType), f"Invalid forward method: {forward_method}"

        if forward_method == "hyperplane_forward_correct":
            assert isinstance(self.manifold, PoincareBall), "custom hyperplane_forward methods only work for PoincareBall manifolds"
        elif forward_method == "hyperplane_forward_pp":
            assert False, "Not implemented yet"

    def forward_hyperplane_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass via hyperplane_forward:
            1) Map x to the manifold.
            2) Perform the hyperplane forward pass.
        [Only works for the PoincareBall]
        """
        assert self.manifold.is_in_manifold(self.bias)

        x = self.manifold.expmap_0(x)
        res = self.manifold.hyperplane_forward(x, self.weight, self.bias, signed=True, scaled=True)
        return res

    def forward_hyperplane_forward_correct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass via hyperplane_forward_correct:
            1) Parallel transport the weight to the tangent space of the bias.
            2) Map x to the manifold.
            3) Perform the corrected hyperplane forward pass.
        [Only works for the PoincareBall]
        """
        assert self.manifold.is_in_manifold(self.bias)

        tangent_space_weight = self.manifold.ptransp_0(self.weight, self.bias)
        x = self.manifold.expmap_0(x)
        res = self.manifold.hyperplane_forward_correct(x, tangent_space_weight, self.bias)
        return res

    def forward_dist2hyperplane_pp(self, x: torch.Tensor) -> torch.Tensor:
        """
        #TODO
        Forward pass via hyperplane_forward_pp:
            1) ....
            2) ....
        [Only works for the PoincareBall]
        """
        assert False, "Not implemented yet"

        assert self.manifold.is_in_manifold(self.bias)

        x = self.manifold.expmap_0(x)
        res = self.manifold.hyperplane_forward_pp(x, self.weight, self.bias)
        return res

    def forward_matvec_mul(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass via tangent space matrix-vector mult.:
           1) Perform Matrix-Vector multiplication in the tangent space and map back to the manifold.
           2) Add the manifold bias to the previous result.
        """
        assert self.manifold.is_in_manifold(self.bias)

        x = self.manifold.expmap_0(x @ self.weight)
        res = self.manifold.addition(x, self.bias)
        return res

    def forward_hnn_matvec_mul(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass as in hyperbolic neural networks paper:
           1) Perform Matrix-Vector multiplication in the tangent space and map back to the manifold.
           2) Add the manifold bias to the previous result via parallel transport.
           3) Map the result back to the manifold.
        """
        assert self.manifold.is_in_manifold(self.bias)

        x = self.manifold.expmap_0(x @ self.weight)
        bias = self.manifold.ptransp_0(self.manifold.logmap_0(self.bias), x)
        res = self.manifold.expmap(bias, x)
        return res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_method(x)
