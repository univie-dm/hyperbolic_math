import torch

from typing import Literal, get_args
from . import Manifold, ManifoldParameter, Euclidean, Hyperboloid, PoincareBall


ForwardPassType = Literal[
    None,
    "hyperplane_forward",
    "hyperplane_forward_correct",
    "hyperplane_forward_pp",
    "fully_linear_pp",
    "hyperplane_forward_pp_ours",
    "matvec_mul"
]


class Embedding(torch.nn.Module):
    """
    Embedding layer that supports different manifolds.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        manifold: Manifold,
        dtype: torch.dtype = torch.float64,
        requires_grad: bool = True,
        forward_method: ForwardPassType = None,
        backproject: bool = True
    ):
        super().__init__()
        self.manifold = manifold
        self.dtype = dtype
        self.backproject = backproject

        if isinstance(self.manifold, Euclidean):
            print("Euclidean embedding layer: Forward pass is defaulted to 'matvec_mul' & 'addition'", flush=True)
            self.forward_method = getattr(self, "forward_matvec_mul")
        elif ForwardPassType:
            self._sanity_checks(forward_method)
            self.forward_method = getattr(self, f"forward_{forward_method}")
        elif isinstance(self.manifold, Hyperboloid):
            assert False, "Not implemented yet"
        else:   # PoincareBall with unspecified forward method specified
            print("PoincareBall embedding layer: Forward pass is defaulted to 'hyperplane_forward'", flush=True)
            self.forward_method = getattr(self, "forward_hyperplane_forward")

        weight = torch.randn(input_dim, output_dim, dtype=self.dtype)
        self.weight = torch.nn.Parameter(weight, requires_grad=requires_grad)

        if forward_method in ["matvec_mul", "hyperplane_forward_pp", "fully_linear_pp", "hyperplane_forward_pp_ours"] or isinstance(self.manifold, Euclidean):
            bias = torch.zeros(output_dim, dtype=self.dtype)
        else: # PoincareBall 'hyperplane_forward' & 'hyperplane_forward_correct'
            bias = torch.zeros(input_dim, dtype=self.dtype)
        if forward_method in ["hyperplane_forward_pp", "fully_linear_pp", "hyperplane_forward_pp_ours"]:
            self.bias = torch.nn.Parameter(bias, requires_grad=requires_grad)
        else: # 'hyperplane_forward', 'hyperplane_forward_correct'
            self.bias = ManifoldParameter(bias, requires_grad=requires_grad, manifold=self.manifold)

    def _sanity_checks(self, forward_method: ForwardPassType) -> None:
        """Sanity checks to ensure correct initialization and forward method"""
        assert forward_method in get_args(ForwardPassType), f"Invalid forward method: {forward_method}"
        if forward_method in ["hyperplane_forward_correct", "hyperplane_forward_pp"]:
            assert isinstance(self.manifold, PoincareBall), "custom hyperplane_forward methods only work for PoincareBall manifolds"

    def forward_hyperplane_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass via hyperplane_forward:
            1) Map x to the manifold.
            2) Perform the hyperplane forward pass.
        [Only works for the PoincareBall]
        """
        assert self.manifold.is_in_manifold(self.bias)

        x = self.manifold.expmap_0(x, backproject=self.backproject)
        res = self.manifold.hyperplane_forward(x, self.weight, self.bias, signed=True, scaled=True, backproject=self.backproject)
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
        x = self.manifold.expmap_0(x, backproject=self.backproject)
        res = self.manifold.hyperplane_forward_correct(x, tangent_space_weight, self.bias, backproject=self.backproject)
        return res

    def forward_hyperplane_forward_pp(self, x: torch.Tensor) -> torch.Tensor:
        """
        #TODO
        Forward pass via hyperplane_forward_pp:
            1) ....
            2) ....
        [Only works for the PoincareBall]
        """
        x = self.manifold.expmap_0(x, backproject=self.backproject)
        res = self.manifold.hyperplane_forward_pp(x, self.weight, self.bias)
        return res

    def forward_fully_linear_pp(self, x: torch.Tensor) -> torch.Tensor:
        """
        #TODO
        Forward pass via fully_linear_pp:
            1) ....
            2) ....
        [Only works for the PoincareBall]
        """
        x = self.manifold.expmap_0(x, backproject=self.backproject)
        res = self.manifold.fully_linear_pp(x, self.weight, self.bias, backproject=self.backproject)
        return res

    def forward_hyperplane_forward_pp_ours(self, x: torch.Tensor) -> torch.Tensor:
        """
        #TODO
        Forward pass via hyperplane_forward_pp_ours:
            1) ....
            2) ....
        [Only works for the PoincareBall]
        """
        bias = self.weight @ self.bias
        bias = self.manifold.expmap_0(bias, backproject=self.backproject)
        assert self.manifold.is_in_manifold(bias)

        res = self.manifold.hyperplane_forward_pp_ours(x, self.weight, bias, backproject=self.backproject)
        return res

    def forward_matvec_mul(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass via tangent space Matrix-Vector multiplication.
            1) Perform Matrix-Vector multiplication in the tangent space.
            2) Map the result onto the manifold.
            3) Add the manifold bias to the result.
        """
        assert self.manifold.is_in_manifold(self.bias)

        x = self.manifold.expmap_0(x @ self.weight, backproject=self.backproject)
        res = self.manifold.addition(x, self.bias, backproject=self.backproject)
        return res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass with the specified method 'self.forward_method'
        with precision specified by 'self.dtype'.
        """
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        return self.forward_method(x)
