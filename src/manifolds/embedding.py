import torch

from typing import Literal, get_args
from . import Manifold, ManifoldParameter, Euclidean, Hyperboloid, PoincareBall


ForwardPassType = Literal[
    # Defaults
    "manifold_FC",
    "manifold_MLR",
    # Hyperbolic Reinforcement Learning (HRL)
    "HRL_forward",
    "HRL_forward_rs",
    # Hyperbolic Neural Networks (HNN)
    "HNN_FC",
    "HNN_MLR",
    # Hyperbolic Neural Networks ++ (HNNpp)
    "HNNpp_FC",
    "HNNpp_MLR"
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
        params_dtype: str="float32",
        requires_grad: bool=True,
        forward_method: ForwardPassType="manifold_FC",
        backproject: bool=True
    ):
        super().__init__()
        self.manifold = manifold
        self.backproject = backproject
        if params_dtype == "float16":
            self.params_dtype = torch.float16
        elif params_dtype == "float32":
            self.params_dtype = torch.float32
        elif params_dtype == "float64":
            self.params_dtype = torch.float64
        else:
            raise ValueError(f"Unsupported params_dtype: {params_dtype}."
                              "Supported dtypes are float16, float32, and float64.")

        if isinstance(self.manifold, Euclidean) and forward_method in ["manifold_FC", "manifold_MLR"]:
            print(f"{self.manifold} embedding layer: Default forward pass '{forward_method}' is used.", flush=True)
            self.forward_method = getattr(self, f"forward_{forward_method}")
        elif isinstance(self.manifold, Hyperboloid):
            assert False, "Not implemented yet"
        elif isinstance(self.manifold, PoincareBall) and forward_method == "manifold_FC":
            print(f"{self.manifold} embedding layer: Default forward pass 'HNN_FC' is used.", flush=True)
            self.forward_method = getattr(self, f"forward_HNN_FC")
        elif isinstance(self.manifold, PoincareBall) and forward_method == "manifold_MLR":
            print(f"{self.manifold} embedding layer: Default forward pass 'HNN_MLR' is used.", flush=True)
            self.forward_method = getattr(self, f"forward_HNN_MLR")
        # PoincareBall / Unsupported methods
        elif ForwardPassType:
            # Sanity check
            assert forward_method in get_args(ForwardPassType), f"Invalid {self.manifold.name} forward method: {forward_method}"
            self.forward_method = getattr(self, f"forward_{forward_method}")

        if torch.finfo(self.params_dtype).eps < torch.finfo(manifold.dtype).eps:
            print(f"Warning: Embedding.params_dtype is {self.params_dtype}, but Manifold.dtype is {manifold.dtype}."
                  f"All manifold operations will be performed in lower precision {manifold.dtype}!")

        weight = torch.randn((output_dim, input_dim), dtype=self.params_dtype)
        self.weight = torch.nn.Parameter(weight, requires_grad=requires_grad)

        if forward_method in ["manifold_FC", "manifold_MLR"]:
            bias = torch.zeros((1, output_dim), dtype=self.params_dtype)
            self.bias = torch.nn.Parameter(bias, requires_grad=requires_grad)
        # PoincareBall exclusive forward methods
        elif forward_method in ["HRL_forward", "HRL_forward_rs"]:
            bias = torch.zeros((output_dim, input_dim), dtype=self.params_dtype)
            self.bias = ManifoldParameter(bias, requires_grad=requires_grad, manifold=self.manifold)
        elif forward_method == "HNN_FC":
            bias = torch.zeros((1, output_dim), dtype=self.params_dtype)
            self.bias = ManifoldParameter(bias, requires_grad=requires_grad, manifold=self.manifold)
        elif forward_method == "HNN_MLR":
            bias = torch.zeros((output_dim, input_dim), dtype=self.params_dtype)
            self.bias = ManifoldParameter(bias, requires_grad=requires_grad, manifold=self.manifold)
        elif forward_method in ["HNNpp_FC", "HNNpp_MLR"]:
            # Bias is aligned and reduced to a scalar multiple of the tangent normal
            bias = torch.zeros((output_dim, 1), dtype=self.params_dtype)
            self.bias = torch.nn.Parameter(bias, requires_grad=requires_grad)

    def forward_manifold_FC(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the Euclidean fully connected forward pass based on self.weights and self.bias (standard linear layer).
        Shapes: x: (B, in_dim), self.weight: (out_dim, in_dim), self.bias: (1, out_dim), res: (B, out_dim)
        """
        res = self.manifold.FC_forward(x, self.weight, self.bias)
        return res

    def forward_manifold_MLR(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the Euclidean multinomial linear regressions score(s)
        based on the linear model(s) defined by self.weights and self.bias.
        Shapes: x: (B, in_dim), self.weight: (out_dim, in_dim), self.bias: (1, out_dim), res: (B, out_dim)
        """
        res = self.manifold.MLR_forward(x, self.weight, self.bias)
        return res

    # Hyperbolic Reinforcement Learning (HRL) [Only defined for the PoincareBall]
    def forward_HRL_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the 'Hyperbolic Reinforcement Learning' multinomial linear regressions score(s)
        based on the linear model(s) defined by self.weights and self.bias.
        Shapes: x: (B, in_dim), self.weight: (out_dim, in_dim), self.bias: (out_dim, in_dim), res: (B, out_dim)
        [Only defined for the PoincareBall]
        """
        x = self.manifold.expmap_0(x, axis=-1, backproject=self.backproject)
         # HRL expands the weights to support multiple spaces at once
        # We don't use this. Instead we feed x of shape (B, on_dim)
        res = self.manifold.HRL_forward(x, self.weight, self.bias, version="HRL_forward", backproject=self.backproject)
        return res

    def forward_HRL_forward_rs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the 'Hyperbolic Reinforcement Learning' scaled multinomial linear regressions score(s)
        based on the linear model(s) defined by self.weights and self.bias.
        Shapes: x: (B, in_dim), self.weight: (out_dim, in_dim), self.bias: (out_dim, in_dim), res: (B, out_dim)
        [Only defined for the PoincareBall]
        """
        x = self.manifold.expmap_0(x, axis=-1, backproject=self.backproject)
        # HRL expands the weights to support multiple spaces at once
        # We don't use this. Instead we feed x of shape (B, on_dim)
        res = self.manifold.HRL_forward(x, self.weight, self.bias, version="HRL_forward_rs", backproject=self.backproject)
        return res

    # Hyperbolic Neural Networks (HNN) [Only defined for the PoincareBall]
    def forward_HNN_FC(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the 'Hyperbolic Neural Networks' fully connected forward pass based on self.weights and self.bias.
            1) Perform matrix vector multiplication in the tangent space at the origin.
            2) Map the result to the manifold.
            3) Add the manifold bias to the result.
        Shapes: x: (B, in_dim), self.weight: (out_dim, in_dim), self.bias: (1, out_dim), res: (B, out_dim)
        [Only defined for the PoincareBall]
        """
        assert self.manifold.is_in_manifold(self.bias, axis=-1)
        x = (x.unsqueeze(-1) * self.weight.T.unsqueeze(0)).sum(dim=1) # (B, out_dim)
        x = self.manifold.expmap_0(x, axis=-1, backproject=self.backproject) # (B, out_dim)
        res = self.manifold.addition(x, self.bias, axis=-1, backproject=self.backproject) # (B, out_dim)
        return res

    def forward_HNN_MLR(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the 'Hyperbolic Neural Networks' multinomial linear regressions score(s)
        based on the linear model(s) defined by self.weights and self.bias.
        Shapes: x: (B, in_dim), self.weight: (out_dim, in_dim), self.bias: (out_dim, in_dim), res: (B, out_dim)
        [Only defined for the PoincareBall]
        """
        assert self.manifold.is_in_manifold(self.bias, axis=-1)
        x = self.manifold.expmap_0(x, axis=-1, backproject=self.backproject)
        # Map self.weights from the tangent space at the origin to the tangent space at self.bias
        pt_weight = self.manifold.ptransp_0(self.weight, self.bias, axis=-1) # (out_dim, in_dim)
        res = self.manifold.HNN_MLR(x, pt_weight, self.bias, backproject=self.backproject)
        return res

    # Hyperbolic Neural Networks ++ (HNNpp) [Only defined for the PoincareBall]
    def forward_HNNpp_FC(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the 'Hyperbolic Neural Networks ++' fully connected forward pass based on self.weights and self.bias.
        Shapes: x: (B, in_dim), self.weight: (out_dim, in_dim), self.bias: (out_dim, 1), res: (B, out_dim)
        [Only defined for the PoincareBall]
        """
        x = self.manifold.expmap_0(x, axis=-1, backproject=self.backproject)
        res = self.manifold.HNNpp_forward(x, self.weight, self.bias, version="HNNpp_FC", backproject=self.backproject)
        return res

    def forward_HNNpp_MLR(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the 'Hyperbolic Neural Networks ++' multinomial linear regressions score(s)
        based on the linear model(s) defined by self.weights and self.bias.
        Shapes: x: (B, in_dim), self.weight: (out_dim, in_dim), self.bias: (out_dim, 1), res: (B, out_dim)
        [Only defined for the PoincareBall]
        """
        x = self.manifold.expmap_0(x, axis=-1, backproject=self.backproject)
        res = self.manifold.HNNpp_forward(x, self.weight, self.bias, version="HNNpp_MLR", backproject=self.backproject)
        return res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass with the specified method 'self.forward_method'.
        """
        return self.forward_method(x)
