import random

import numpy as np
import torch


def linear_lr_scheduler(
    optimizer: torch.optim.Optimizer, iteration: int, num_iterations: int, lr: float
) -> torch.optim.Optimizer:
    frac = 1.0 - (iteration - 1.0) / num_iterations
    lrnow = frac * lr
    optimizer.param_groups[0]["lr"] = lrnow

    return optimizer


def set_seeds(seed: int, torch_deterministic: bool) -> None:
    """Set the seeds of all RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def set_cuda_configuration(gpu: int) -> torch.device:
    """Set up the device for the desired GPU or all GPUs."""
    if gpu == -1:
        device = torch.device("cpu")
    else:
        assert gpu <= torch.cuda.device_count(), "Invalid CUDA index specified."
        device = torch.device(f"cuda:{gpu}")

    return device
