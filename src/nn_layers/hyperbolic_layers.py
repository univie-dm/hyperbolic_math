import torch

from typing import Dict
from ..manifolds import Manifold, ManifoldParameter


class HyperbolicBaseLayer(torch.nn.Module):
    """
    Base class for hyperbolic neural network layers.

    Parameters
    ----------
    manifold : Manifold
        The hyperbolic manifold
    hyperbolic_axis : int
        Axis along which the input tensor is hyperbolic (default: -1)
    """
    def __init__(
        self,
        manifold: Manifold,
        hyperbolic_axis: int = -1
    ):
        super().__init__()
        self.manifold = manifold
        self.hyperbolic_axis = hyperbolic_axis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the hyperbolic layer."""
        raise NotImplementedError("This method should be implemented by subclasses.")

class Expmap(HyperbolicBaseLayer):
    """
    Module to compute the exponential map at a point on the manifold.

    Parameters
    ----------
    manifold : Manifold
        The hyperbolic manifold
    hyperbolic_axis : int
        Axis along which the input tensor is hyperbolic (default: -1)
    backproject : bool
        Whether to project results back to the manifold (default: True)
    """
    def __init__(self, manifold: Manifold, hyperbolic_axis: int=-1, backproject: bool=True):
        super().__init__(manifold, hyperbolic_axis)
        self.backproject = backproject

    def forward(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.manifold.expmap(v, x, axis=self.hyperbolic_axis, backproject=self.backproject)

class Expmap_0(HyperbolicBaseLayer):
    """
    Module to compute the exponential map at the origin of the manifold.

    Parameters
    ----------
    manifold : Manifold
        The hyperbolic manifold
    hyperbolic_axis : int
        Axis along which the input tensor is hyperbolic (default: -1)
    backproject : bool
        Whether to project results back to the manifold (default: True)
    """
    def __init__(self, manifold: Manifold, hyperbolic_axis: int=-1, backproject: bool=True):
        super().__init__(manifold, hyperbolic_axis)
        self.backproject = backproject

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return self.manifold.expmap_0(v, axis=self.hyperbolic_axis, backproject=self.backproject)

class Retraction(HyperbolicBaseLayer):
    """
    Module to compute the retraction map at a point on the manifold.

    Parameters
    ----------
    manifold : Manifold
        The hyperbolic manifold
    hyperbolic_axis : int
        Axis along which the input tensor is hyperbolic (default: -1)
    backproject : bool
        Whether to project results back to the manifold (default: True)
    """
    def __init__(self, manifold: Manifold, hyperbolic_axis: int=-1, backproject: bool=True):
        super().__init__(manifold, hyperbolic_axis)
        self.backproject = backproject

    def forward(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.manifold.retraction(v, x, axis=self.hyperbolic_axis, backproject=self.backproject)

class Logmap(HyperbolicBaseLayer):
    """
    Module to compute the logarithmic map at a point on the manifold.

    Parameters
    ----------
    manifold : Manifold
        The hyperbolic manifold
    hyperbolic_axis : int
        Axis along which the input tensor is hyperbolic (default: -1)
    """
    def __init__(self, manifold: Manifold, hyperbolic_axis: int=-1):
        super().__init__(manifold, hyperbolic_axis)

    def forward(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.manifold.logmap(y, x, axis=self.hyperbolic_axis)

class Logmap_0(HyperbolicBaseLayer):
    """
    Module to compute the logarithmic map at the origin of the manifold.

    Parameters
    ----------
    manifold : Manifold
        The hyperbolic manifold
    hyperbolic_axis : int
        Axis along which the input tensor is hyperbolic (default: -1)
    """
    def __init__(self, manifold: Manifold, hyperbolic_axis: int=-1):
        super().__init__(manifold, hyperbolic_axis)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.manifold.logmap_0(y, axis=self.hyperbolic_axis)

class Proj(HyperbolicBaseLayer):
    """
    Module to compute the (back)projection onto the manifold to account for numerical instabilities.

    Parameters
    ----------
    manifold : Manifold
        The hyperbolic manifold
    hyperbolic_axis : int
        Axis along which the input tensor is hyperbolic (default: -1)
    """
    def __init__(self, manifold: Manifold, hyperbolic_axis: int=-1):
        super().__init__(manifold, hyperbolic_axis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.manifold.proj(x, axis=self.hyperbolic_axis)


# Dictionary mapping dtype strings to torch dtypes
DTYPE_MAP: Dict[str, torch.dtype] = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}

def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype representation to torch dtype.

    Parameters
    ----------
    dtype_str : str
        String representation of dtype ('float16', 'float32', or 'float64')

    Returns
    -------
    torch.dtype
        Corresponding torch dtype

    Raises
    ------
    ValueError
        If dtype_str is not supported
    """
    if dtype_str not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {dtype_str}. "
                         f"Supported dtypes are: {', '.join(DTYPE_MAP.keys())}")
    return DTYPE_MAP[dtype_str]

class HyperbolicParametrizedLayer(HyperbolicBaseLayer):
    """
    Base class for parametrized hyperbolic neural network layers.
    """
    def __init__(
        self,
        manifold: Manifold,
        input_dim: int,
        output_dim: int,
        hyperbolic_axis: int = -1,
        backproject: bool = True,
        params_dtype: str = "float32",
        requires_grad: bool = True,
        input_space: str = "manifold"
    ):
        assert hyperbolic_axis == -1, "hyperbolic_axis must be -1, reshape your tensor accordingly."
        super().__init__(manifold, hyperbolic_axis)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.backproject = backproject

        self.params_dtype = get_torch_dtype(params_dtype)
        if torch.finfo(self.params_dtype).eps < torch.finfo(manifold.dtype).eps:
            print(f"Warning: Embedding.params_dtype is {self.params_dtype}, but Manifold.dtype is {manifold.dtype}."
                  f"All manifold operations will be performed in lower precision {manifold.dtype}!")

        self.requires_grad = requires_grad
        weight = torch.randn((output_dim, input_dim), dtype=self.params_dtype)
        self.weight = torch.nn.Parameter(weight, requires_grad=requires_grad)

        assert input_space in ["tangent", "manifold"], "input_space must be either 'tangent' or 'manifold'"
        self.input_space = input_space

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the parametrized hyperbolic layer."""
        raise NotImplementedError("This method should be implemented by subclasses.")

class HyperbolicLinear(HyperbolicParametrizedLayer):
    """
    Module to compute the 'Hyperbolic Neural Networks' fully connected layer:
        0) Project the input tensor to the tangent space (optional)
        1) Perform matrix vector multiplication in the tangent space at the origin.
        2) Map the result to the manifold.
        3) Add the manifold bias to the result.

    Parameters
    ----------
    manifold : Manifold
        The hyperbolic manifold to use (e.g. PoincareBall)
    input_dim : int
        Dimension of the input space
    output_dim : int
        Dimension of the output space
    hyperbolic_axis : int
        Axis along which the input tensor is hyperbolic (needs to be -1)
    backproject : bool
        Whether to project results back to the manifold (default: True)
    params_dtype : str
        Data type for the parameters (default: "float32")
    requires_grad : bool
        Whether the parameters should require gradients (default: True)
    input_space : str
        Type of the input tensor, either 'tangent' or 'manifold' (default: 'manifold')

    References
    ----------
    Ganea Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic neural networks."
        Advances in neural information processing systems 31 (2018).
    """
    def __init__(
        self,
        manifold: Manifold,
        input_dim: int,
        output_dim: int,
        hyperbolic_axis: int = -1,
        backproject: bool = True,
        params_dtype: str = "float32",
        requires_grad: bool = True,
        input_space: str = "manifold"
    ):
        super().__init__(manifold, input_dim, output_dim, hyperbolic_axis, backproject, params_dtype, requires_grad, input_space)

        bias = torch.zeros((1, self.output_dim), dtype=self.params_dtype)
        self.bias = ManifoldParameter(bias, requires_grad=self.requires_grad, manifold=self.manifold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: x of shape (B, in_dim) where the hyperbolic_axis is last
        Parameters: self.weight of shape (out_dim, in_dim), self.bias of shape (1, out_dim)
        Output: res of shape (B, out_dim)
        """
        assert self.manifold.is_in_manifold(self.bias, axis=self.hyperbolic_axis)

        if self.input_space == "manifold":
            x = self.manifold.logmap_0(x, axis=self.hyperbolic_axis)
        x = (x.unsqueeze(-1) * self.weight.T.unsqueeze(0)).sum(dim=1) # (B, out_dim)
        x = self.manifold.expmap_0(x, axis=self.hyperbolic_axis, backproject=self.backproject) # (B, out_dim)
        res = self.manifold.addition(x, self.bias, axis=self.hyperbolic_axis, backproject=self.backproject) # (B, out_dim)
        return res
