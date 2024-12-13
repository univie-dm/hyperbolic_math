from typing import Union

import torch

from ..manifolds import Euclidean


class OptimMixin:
    _default_manifold = Euclidean()

    def __init__(self, *args, c: torch.Tensor, expmap_update: bool, stabilize: Union[int, None] = None, **kwargs):
        if c.numel() != 1:
            raise ValueError(f"Curvature {c} must be a scalar tensor.")
        self.c = c
        self.expmap_update = expmap_update
        self._stabilize = stabilize
        super().__init__(*args, **kwargs)

    def add_param_group(self, param_group: dict):
        param_group.setdefault("stabilize", self._stabilize)
        return super().add_param_group(param_group)

    def stabilize_group(self, group):
        pass

    def stabilize(self):
        """Stabilize parameters if they are off-manifold due to numerical reasons."""
        for group in self.param_groups:
            self.stabilize_group(group)
