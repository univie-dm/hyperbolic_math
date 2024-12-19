from typing import Tuple

import torch.optim

from ..manifolds import ManifoldParameter
from .mixin import OptimMixin


__all__ = ["RiemannianAdam"]


class RiemannianAdam(OptimMixin, torch.optim.Adam):
    r"""
    Riemannian Adam with the same API as :class:`torch.optim.Adam`.

    Parameters
    ----------
    params : iterable
        iterable of parameters to optimize or dicts defining
        parameter groups
    lr : float (optional)
        learning rate (default: 1e-3)
    betas : Tuple[float, float] (optional)
        coefficients used for computing
        running averages of gradient and its square (default: (0.9, 0.999))
    eps : float (optional)
        term added to the denominator to improve
        numerical stability (default: 1e-8)
    weight_decay : float (optional)
        weight decay (L2 penalty) (default: 0)
    amsgrad : bool (optional)
        whether to use the AMSGrad variant of this
        algorithm from the paper `On the Convergence of Adam and Beyond`_
        (default: False)

    Other Parameters
    ----------------
    c : torch.Tensor
        Curvature parameter for mathematical operations on the manifold
    expmap_update : bool = False
        Update the parameters with exponential map instead of retraction
    stabilize : int | None
        Stabilize parameters if they are off-manifold due to numerical
        reasons every ``stabilize`` steps (default: ``None`` -- no stabilize)


    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ

    """

    def __init__(
        self,
        params,
        lr: float,
        c: torch.Tensor,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        expmap_update: bool = False,
        stabilize: bool = None,
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
        super().__init__(params, c=c, expmap_update=expmap_update, stabilize=stabilize, **defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        with torch.no_grad():
            for group in self.param_groups:
                betas = group["betas"]
                weight_decay = group["weight_decay"]
                eps = group["eps"]
                learning_rate = group["lr"]
                amsgrad = group["amsgrad"]
                stabilize = False
                for point in group["params"]:
                    grad = point.grad
                    if grad is None:
                        continue

                    # Flag for hyperbolic parameters
                    param_is_hyperbolic = isinstance(point, ManifoldParameter)
                    if param_is_hyperbolic:
                        manifold = point.manifold
                    else:
                        manifold = self._default_manifold

                    if grad.is_sparse:
                        raise RuntimeError(
                            "RiemannianAdam does not support sparse gradients, use SparseRiemannianAdam instead"
                        )

                    state = self.state[point]

                    # State initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(point)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(point)
                        if amsgrad:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state["max_exp_avg_sq"] = torch.zeros_like(point)
                    state["step"] += 1
                    # make local variables for easy access
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    # actual step
                    grad.add_(point, alpha=weight_decay)
                    grad = manifold.egrad2rgrad(grad, point, self.c)
                    exp_avg.mul_(betas[0]).add_(grad, alpha=1 - betas[0])

                    if param_is_hyperbolic:
                        # Hyperbolic parameter: Compute <grad, grad>_x in tangent space
                        exp_avg_sq_new = manifold.tangent_inner(u=grad, v=grad, x=point, c=self.c)
                    else:
                        # Euclidean parameter: Compute grad^2 parameter-wise
                        exp_avg_sq_new = grad.pow(2)

                    exp_avg_sq.mul_(betas[1]).add_(exp_avg_sq_new, alpha=1 - betas[1])
                    bias_correction1 = 1 - betas[0] ** state["step"]
                    bias_correction2 = 1 - betas[1] ** state["step"]
                    if amsgrad:
                        max_exp_avg_sq = state["max_exp_avg_sq"]
                        # Maintains the maximum of all 2nd moment running avg. till now
                        torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        # Use the max. for normalizing running avg. of gradient
                        denom = max_exp_avg_sq.div(bias_correction2).sqrt_()
                    else:
                        denom = exp_avg_sq.div(bias_correction2).sqrt_()
                    # copy the state, we need it for retraction
                    # get the direction for ascend
                    direction = exp_avg.div(bias_correction1) / denom.add_(eps)
                    # transport the exponential averaging to the new point
                    # 1. Project onto Tangent space
                    # 2. Update with exponential map
                    if self.expmap_update:
                        new_point = manifold.expmap(-learning_rate * direction, point, self.c)
                    else:
                        new_point = manifold.retraction(-learning_rate * direction, point, self.c)
                    exp_avg_new = manifold.ptransp(exp_avg, point, new_point, self.c)
                    # use copy only for user facing point
                    point.copy_(new_point)
                    exp_avg.copy_(exp_avg_new)

                    if group["stabilize"] is not None and state["step"] % group["stabilize"] == 0:
                        stabilize = True
                if stabilize:
                    self.stabilize_group(group)
        return loss

    @torch.no_grad()
    def stabilize_group(self, group):
        for p in group["params"]:
            if not isinstance(p, ManifoldParameter):
                continue
            state = self.state[p]
            if not state:  # due to None grads
                continue
            manifold = p.manifold
            exp_avg = state["exp_avg"]
            p.copy_(manifold.proj(p, self.c))
            # FIXME: Must implement proju
            exp_avg.copy_(manifold.proju(p, exp_avg))
