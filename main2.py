import torch
import numpy as np
from src.manifolds import PoincareBall
from src.utils.vis_utils import create_figure


torch.manual_seed(48) # 46, 48
dtype = torch.float32
manifold = PoincareBall(c=torch.tensor([3.0], dtype=dtype))
num_pts = 6
dim = 2

random_dirs = torch.normal(0, 1, size=(num_pts, dim), dtype=dtype)
random_dirs /= random_dirs.norm(p=2, dim=-1, keepdim=True)
random_radii = torch.rand((num_pts, 1), dtype=dtype).pow(1 / dim)
points = manifold.c**-0.5 * (random_dirs * random_radii)

#points = points[1:4]

labels = np.array([0] * (num_pts // 2) + [1] * (num_pts // 2))

edges = ([i for i in range(5)], [i + 1 for i in range(5)])
#edges = ([2, 1], [0, 2])
#edges = ([1], [2])

if dim == 2:
    hyperplanes = (torch.tensor([[0.0, 0.5],
                                [0.2, 0.2]],
                                dtype=dtype),
                torch.tensor([[0.0, 0.0],
                                [0.3, 0.3]],
                                dtype=dtype))
else:
    hyperplanes = (torch.tensor([[0.0, 0.5, 0.0, 0.0],
                                [0.2, 0.2, 0.0, 0.0]],
                                dtype=dtype),
                torch.tensor([[0.0, 0.0, 0.0, 0.0],
                                [0.3, 0.3, 0.0, 0.0]],
                                dtype=dtype))


fig = create_figure(points, manifold, labels=None, edges=edges, hyperplanes=None)