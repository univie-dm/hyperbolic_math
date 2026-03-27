import torch

from typing import Any, Dict, Iterable, Tuple, Union
from ..manifolds import ManifoldParameter, Euclidean


__all__ = ["RiemannianAdam"]


class RiemannianAdam(torch.optim.Adam):
    """
    Riemannian Adam with the same API as :class:`torch.optim.Adam`.

    Parameters
    ----------
    params : iterable
        Iterable of parameters to optimize or dicts defining parameter groups
    lr : float (optional)
        Learning rate (default: 1e-3)
    betas : Tuple[float, float] (optional)
        Coefficients used for computing running averages of gradient
        and its square (default: (0.9, 0.999))
    eps : float (optional)
        Term added to the denominator to improve
        numerical stability (default: 1e-8)
    weight_decay : float (optional)
        Weight decay (L2 penalty) (default: 0)
    amsgrad : bool (optional)
        Whether to use the AMSGrad variant of this algorithm (default: False)

    Other Parameters
    ----------------
    expmap_update : bool
        Update the parameters with exponential map instead of retraction (default: False)
    hyperbolic_axis : int
        Axis along which the parameters are hyperbolic (default: -1)

    References
    ----------
    Max Kochurov, Rasul Karimov and Serge Kozlukov. "Geoopt: Riemannian Optimization in PyTorch."
        arXiv (2020).
    Sashank J. Reddi, Satyen Kale, and Sanjiv Kumar. "On the convergence of adam and beyond."
        arXiv preprint arXiv:1904.09237 (2019).
    """
    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
        lr: float,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        expmap_update: bool = False,
        hyperbolic_axis: int = -1
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        super().__init__(params, **defaults)
        self.expmap_update = expmap_update
        self.hyperbolic_axis = hyperbolic_axis

    def step(self, closure=None) -> None:
        loss = None

        if closure is not None:
            loss = closure()

        with torch.no_grad():
            for group in self.param_groups:
                beta1, beta2 = group["betas"]
                weight_decay = group["weight_decay"]
                eps = group["eps"]
                learning_rate = group["lr"]
                amsgrad = group["amsgrad"]

                for point in group["params"]:
                    grad = point.grad
                    if grad is None:
                        continue

                    # State initialization
                    state = self.state[point]
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(point)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(point)
                        if amsgrad:
                            # Maintains the max of all exp_avg_sq
                            state["max_exp_avg_sq"] = torch.zeros_like(point)

                    # Actual step
                    state["step"] += 1
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    # Apply weight decay
                    if weight_decay != 0:
                        grad = grad.add(point, alpha=weight_decay)

                    # Check for hyperbolic parameters to distinguish between Euclidean- and RiemannianAdam
                    param_is_hyperbolic = isinstance(point, ManifoldParameter) and not isinstance(point.manifold, Euclidean)

                    if param_is_hyperbolic:
                        manifold = point.manifold
                        # Make the gradient coordinate-system independent and orthogonally project onto the tangent space
                        grad = manifold.egrad2rgrad(grad, point, axis=self.hyperbolic_axis)

                    # Decay the first and second moment running average coefficient
                    exp_avg.lerp_(grad, 1-beta1)
                    if param_is_hyperbolic:
                        # Compute <grad, grad>_x in tangent space
                        grad_tangent_inner = manifold.tangent_inner(grad, grad, point, axis=self.hyperbolic_axis).to(grad.dtype)
                        exp_avg_sq.lerp_(grad_tangent_inner, 1-beta2)
                    else:
                        # Compute grad^2 component-wise
                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)

                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]

                    if amsgrad:
                        # Update max_exp_avg_sq and use it for normalizing moving averages
                        max_exp_avg_sq = state["max_exp_avg_sq"]
                        torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        # Use max_exp_avg_sq for normalizing running averages
                        denom = max_exp_avg_sq.div(bias_correction2).sqrt().add_(eps)
                    else:
                        denom = exp_avg_sq.div(bias_correction2).sqrt().add_(eps)

                    # Get the step size
                    step_size = learning_rate / bias_correction1

                    if param_is_hyperbolic:
                        # Perform Riemannian Adam update with the bias-corrected moment 'exp_avg / denom'
                        if self.expmap_update:
                            # Exact update on the manifold using the exponential map
                            new_point = manifold.expmap(-step_size * exp_avg / denom, point, axis=self.hyperbolic_axis).to(point.dtype)
                        else:
                            # First-order approximation of the update using the retraction mapping
                            new_point = manifold.retraction(-step_size * exp_avg / denom, point, axis=self.hyperbolic_axis).to(point.dtype)

                        # Parallel transport the exponential averaging to the new point
                        new_exp_avg = manifold.ptransp(exp_avg, point, new_point, axis=self.hyperbolic_axis).to(exp_avg.dtype)

                        # Update point and the running average
                        point.copy_(new_point)
                        exp_avg.copy_(new_exp_avg)
                    else:
                        # Standard Adam update in Euclidean space
                        point.addcdiv_(exp_avg, denom, value=-step_size)
        return loss
