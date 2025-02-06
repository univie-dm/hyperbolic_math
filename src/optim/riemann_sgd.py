import torch.optim.optimizer

from ..manifolds import ManifoldParameter, Euclidean, Hyperboloid


__all__ = ["RiemannianSGD"]


class RiemannianSGD(torch.optim.Optimizer):
    """
    Riemannian Stochastic Gradient Descent with the same API as :class:`torch.optim.SGD`.

    Parameters
    ----------
    params : iterable
        Iterable of parameters to optimize or dicts defining
        parameter groups
    lr : float
        Learning rate
    momentum : float (optional)
        Momentum factor (default: 0)
    weight_decay : float (optional)
        Weight decay (L2 penalty) (default: 0)
    dampening : float (optional)
        Dampening for momentum (default: 0)
    nesterov : bool (optional)
        Enables Nesterov momentum (default: False)

    Other Parameters
    ----------------
    c : torch.Tensor
        Curvature parameter of the manifold
        (Currently removed)
    expmap_update : bool = False
        Update the parameters with exponential map instead of retraction

    References
    ----------
    Max Kochurov, Rasul Karimov and Serge Kozlukov. "Geoopt: Riemannian Optimization in PyTorch."
        arXiv (2020).
    """

    def __init__(
        self,
        params,
        lr: float,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        expmap_update: bool = False,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)
        self.expmap_update = expmap_update

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        with torch.no_grad():
            for group in self.param_groups:
                if "step" not in group:
                    group["step"] = 0
                weight_decay = group["weight_decay"]
                momentum = group["momentum"]
                dampening = group["dampening"]
                nesterov = group["nesterov"]
                learning_rate = group["lr"]
                group["step"] += 1
                for point in group["params"]:
                    grad = point.grad
                    if grad is None:
                        continue

                    # Flag for hyperbolic parameters
                    if isinstance(point, ManifoldParameter):
                        manifold = point.manifold
                    else:
                        manifold = Euclidean()

                    state = self.state[point]
                    # State initialization
                    if len(state) == 0:
                        if momentum > 0:
                            state["momentum_buffer"] = grad.clone()

                    # Actual step
                    grad.add_(point, alpha=weight_decay)
                    grad = manifold.egrad2rgrad(grad, point)
                    if momentum > 0:
                        momentum_buffer = state["momentum_buffer"]
                        momentum_buffer.mul_(momentum).add_(grad, alpha=1 - dampening)
                        if nesterov:
                            grad = grad.add_(momentum_buffer, alpha=momentum)
                        else:
                            grad = momentum_buffer
                    
                    if isinstance(manifold, Hyperboloid):
                        # Project the gradient direction onto the Tangent space
                        pass
                    if self.expmap_update:
                        # Exact update on the manifold using the exponential map
                        new_point = manifold.expmap(-learning_rate * grad, point)
                    else:
                        # First-order approximation of the update using the retraction mapping
                        new_point = manifold.retraction(-learning_rate * grad, point)

                    if momentum > 0:
                        # Parallel transport the momentum to the new point
                        new_momentum_buffer = manifold.ptransp(momentum_buffer, point, new_point)
                        momentum_buffer.copy_(new_momentum_buffer)
                    # Use copy only for user facing point
                    point.copy_(new_point)
        return loss
