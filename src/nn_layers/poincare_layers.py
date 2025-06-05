import torch

from .hyperbolic_layers import HyperbolicParametrizedLayer
from ..manifolds import ManifoldParameter, PoincareBall
from ..utils.math_utils import arsinh, sinh, cosh


class HyperbolicRegressionPoincare(HyperbolicParametrizedLayer):
    """
    Module to compute the 'Hyperbolic Neural Networks' multinomial linear regression score(s):
        0) Project the input tensor onto the manifold (optional)
        1) Compute the multinomial linear regression score(s)

    Parameters
    ----------
    manifold : PoincareBall
        The hyperbolic manifold (needs to be PoincareBall)
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
        manifold: PoincareBall,
        input_dim: int,
        output_dim: int,
        hyperbolic_axis: int = -1,
        backproject: bool = True,
        params_dtype: str = "float32",
        requires_grad: bool = True,
        input_space: str = "manifold"
    ):
        assert isinstance(manifold, PoincareBall), "Manifold must be an instance of PoincareBall."
        super().__init__(manifold, input_dim, output_dim, hyperbolic_axis, backproject, params_dtype, requires_grad, input_space)

        bias = torch.zeros((self.output_dim, self.input_dim), dtype=self.params_dtype)
        self.bias = ManifoldParameter(bias, requires_grad=self.requires_grad, manifold=self.manifold)

    def _compute_mlr(self, x: torch.Tensor, a: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Internal method for computing the multinomial linear regression score(s).

        Parameters
        ----------
        x : torch.Tensor (B, in_dim)
            PoincareBall point(s)
        a : torch.Tensor (out_dim, in_dim)
            Hyperplane tangent normal(s) in the tangent space at p
        p : torch.Tensor (out_dim, in_dim)
            Hyperplane PoincareBall translation(s)

        Returns
        -------
        res : torch.Tensor (B, out_dim)
            The multinomial linear regression score(s) of x with respect to the linear model(s) defined by a and p.

        References
        ----------
        Ganea Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic neural networks."
            Advances in neural information processing systems 31 (2018).
        """
        x, a, p = self.manifold._2manifold_dtype([x, a, p])
        sqrt_c = self.manifold.c.sqrt()
        sub = self.manifold.addition(-p.T.unsqueeze(0), x.unsqueeze(-1), axis=1, backproject=self.backproject) # (B, in_dim, out_dim)
        suba = (sub * a.T).sum(dim=1, keepdim=True) # (B, 1, out_dim)
        a_norm = a.norm(p=2, dim=self.hyperbolic_axis, keepdim=True).clamp_min(self.manifold.min_enorm).T # (1, out_dim)
        signed_dist2hyp = arsinh(sqrt_c * self.manifold._lambda(sub, axis=1) * suba / a_norm) / sqrt_c # (B, 1, out_dim)
        res = self.manifold._lambda(p, axis=self.hyperbolic_axis).T * a_norm * signed_dist2hyp.squeeze(1) # (B, out_dim)
        return res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: x of shape (B, in_dim) where the hyperbolic_axis is last
        Parameters: self.weight of shape (out_dim, in_dim), self.bias of shape (out_dim, in_dim)
        Output: res of shape (B, out_dim)
        """
        assert self.manifold.is_in_manifold(self.bias, axis=self.hyperbolic_axis)

        if self.input_space == "tangent":
            x = self.manifold.expmap_0(x, axis=self.hyperbolic_axis, backproject=self.backproject)
        # Map self.weights from the tangent space at the origin to the tangent space at self.bias
        pt_weight = self.manifold.ptransp_0(self.weight, self.bias, axis=self.hyperbolic_axis)
        # Compute the multinomial linear regression score(s)
        res = self._compute_mlr(x, pt_weight, self.bias)
        return res


class PoincareBaseLayerPP(HyperbolicParametrizedLayer):
    """
    Base class for Hyperbolic Neural Networks ++ layers.
    """
    def __init__(
        self,
        manifold: PoincareBall,
        input_dim: int,
        output_dim: int,
        hyperbolic_axis: int = -1,
        backproject: bool = True,
        params_dtype: str = "float32",
        requires_grad: bool = True,
        input_space: str = "manifold"
    ):
        assert isinstance(manifold, PoincareBall), "Manifold must be an instance of PoincareBall."
        super().__init__(manifold, input_dim, output_dim, hyperbolic_axis, backproject, params_dtype, requires_grad, input_space)

        bias = torch.zeros((self.output_dim, 1), dtype=self.params_dtype)
        self.bias = torch.nn.Parameter(bias, requires_grad=self.requires_grad)

    def _compute_mlr(self, x: torch.Tensor, z: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        Internal method for computing the HNN++ multinomial linear regression.

        Parameters
        ----------
        x : torch.Tensor (B, in_dim)
            PoincareBall point(s)
        z : torch.Tensor (out_dim, in_dim)
            Hyperplane tangent normal(s) in the tangent space at the origin
        r : torch.Tensor (out_dim, 1)
            Hyperplane PoincareBall translation(s) defined by the scalar r and z

        Returns
        -------
        res : torch.Tensor (B, out_dim)
            The multinomial linear regression score(s) of x with respect to the linear model(s) defined by a and z.

        References
        ----------
        Shimizu Ryohei, Yusuke Mukuta, and Tatsuya Harada. "Hyperbolic neural networks++."
            arXiv preprint arXiv:2006.08210 (2020).
        """
        sqrt_c = self.manifold.c.sqrt()
        sqrt_c2r = 2 * sqrt_c * r.T # (out_dim, 1)
        z_norm = z.norm(p=2, dim=self.hyperbolic_axis, keepdim=True).clamp_min(self.manifold.min_enorm) # (out_dim, 1)
        lambda_x = self.manifold._lambda(x, axis=self.hyperbolic_axis) # (B, 1)
        z_unitx = (x.unsqueeze(-1) * (z / z_norm).T).sum(dim=1) # (B, out_dim)
        arsinh_arg = (1-lambda_x) * sinh(sqrt_c2r) + sqrt_c * lambda_x * cosh(sqrt_c2r) * z_unitx # (B, out_dim)
        signed_dist2hyp = arsinh(arsinh_arg) / sqrt_c # (B, out_dim)
        res = 2 * z_norm.T * signed_dist2hyp # (B, out_dim)
        return res

class HyperbolicRegressionPoincarePP(PoincareBaseLayerPP):
    """
    Module to compute the 'Hyperbolic Neural Networks ++' multinomial linear regression score(s):
        0) Project the input tensor onto the manifold (optional)
        1) Compute the multinomial linear regression score(s)

    Parameters
    ----------
    manifold : PoincareBall
        The hyperbolic manifold (needs to be PoincareBall)
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
    Shimizu Ryohei, Yusuke Mukuta, and Tatsuya Harada. "Hyperbolic neural networks++."
        arXiv preprint arXiv:2006.08210 (2020).
    """
    def __init__(
        self,
        manifold: PoincareBall,
        input_dim: int,
        output_dim: int,
        hyperbolic_axis: int = -1,
        backproject: bool = True,
        params_dtype: str = "float32",
        requires_grad: bool = True,
        input_space: str = "manifold"
    ):
        super().__init__(manifold, input_dim, output_dim, hyperbolic_axis, backproject, params_dtype, requires_grad, input_space)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: x of shape (B, in_dim) where the hyperbolic_axis is last
        Parameters: self.weight of shape (out_dim, in_dim), self.bias of shape (out_dim, 1)
        Output: res of shape (B, out_dim)
        """
        if self.input_space == "tangent":
            x = self.manifold.expmap_0(x, axis=self.hyperbolic_axis, backproject=self.backproject)
        res = self._compute_mlr(x, self.weight, self.bias)
        return res

class HyperbolicFullyConnectedPoincarePP(PoincareBaseLayerPP):
    """
    Module to compute the 'Hyperbolic Neural Networks ++' fully connected layer:
        0) Project the input tensor onto the manifold (optional)
        1) Compute the multinomial linear regression score(s)

    Parameters
    ----------
    manifold : PoincareBall
        The hyperbolic manifold (needs to be PoincareBall)
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
    Shimizu Ryohei, Yusuke Mukuta, and Tatsuya Harada. "Hyperbolic neural networks++."
        arXiv preprint arXiv:2006.08210 (2020).
    """
    def __init__(
        self,
        manifold: PoincareBall,
        input_dim: int,
        output_dim: int,
        hyperbolic_axis: int = -1,
        backproject: bool = True,
        params_dtype: str = "float32",
        requires_grad: bool = True,
        input_space: str = "manifold"
    ):
        super().__init__(manifold, input_dim, output_dim, hyperbolic_axis, backproject, params_dtype, requires_grad, input_space)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: x of shape (B, in_dim) where the hyperbolic_axis is last
        Parameters: self.weight of shape (out_dim, in_dim), self.bias of shape (out_dim, 1)
        Output: res of shape (B, out_dim)
        """
        if self.input_space == "tangent":
            x = self.manifold.expmap_0(x, axis=self.hyperbolic_axis, backproject=self.backproject)

        v = self._compute_mlr(x, self.weight, self.bias)
        sqrt_c = self.manifold.c.sqrt()
        w = sinh(sqrt_c * v) / sqrt_c # (B, out_dim)
        w2 = w.pow(2).sum(axis=self.hyperbolic_axis, keepdim=True) # (B, 1)
        denom = 1 + (1 + self.manifold.c * w2).sqrt() # (B, 1)
        res = w / denom # (B, out_dim)
        if self.backproject:
            res = self.manifold.proj(res, axis=self.hyperbolic_axis)
        return res
