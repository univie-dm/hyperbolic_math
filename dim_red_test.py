import time
import torch

from icecream import ic
from matplotlib import pyplot as plt
from src.manifolds import PoincareBall
from src.utils.helpers import compute_pairwise_distances
from src.utils.horo_pca import HoroPCA
from src.utils.vis_utils import create_figure, pointsTo2dPoincare


if torch.cuda.is_available():
    torch.set_default_device('cuda:0')
############### Parameters for testing ##########
seeds = [*range(40, 41)]
dtype = torch.float32
curvature = 1.0
input_dim = 10
output_dim = 2
num_pts = 1_000
#################################################
poincare = PoincareBall(c=torch.tensor([curvature], dtype=dtype), dtype=dtype)
time_taken = 0.0
distortion_orig = torch.tensor(0.0, dtype=dtype, device='cuda:0' if torch.cuda.is_available() else 'cpu')
distortion_scaled = torch.tensor(0.0, dtype=dtype, device='cuda:0' if torch.cuda.is_available() else 'cpu')
time_taken = 0.0
for seed in seeds:
    torch.manual_seed(seed)
    # Generate random data in PoincareBall
    random_dirs = torch.normal(0, 1, size=(num_pts, input_dim), dtype=poincare.dtype, device=poincare.c.device)
    random_dirs /= random_dirs.norm(p=2, dim=-1, keepdim=True)
    random_radii = torch.rand((num_pts, 1), dtype=poincare.dtype).pow(1 / input_dim)
    points = poincare.c**-0.5 * (random_dirs * random_radii)
    dist_original = compute_pairwise_distances(points, poincare) + torch.eye(num_pts)
    points = points * 0.9
    dist_scaled = compute_pairwise_distances(points, poincare) + torch.eye(num_pts)

    # Run dimensionality reduction methods + plot
    settings = {
        "plot_manifold_dtype": torch.float64,
        #"dim_red_method": "tangent PCA",
        "dim_red_method": "HoroPCA",
        "title": "Hyperbolic Embeddings",
        "show_origin": True,
        "save_figure": True,
        "file_name": "hyperbolic_embeddings",
        "file_path": None,
        "file_format": "png",
    }
    start_time = time.time()
    poincare_64 = PoincareBall(c=poincare.c.to(settings['plot_manifold_dtype']), dtype=settings['plot_manifold_dtype'])
    points = poincare_64.to_hyperboloid(points)
    pts, _ = pointsTo2dPoincare(points, poincare_64, settings=settings)

    dist_embeddings = compute_pairwise_distances(pts, poincare) + torch.eye(num_pts)
    max_distortion = torch.abs(dist_embeddings - dist_original) / dist_original
    distortion_orig += torch.mean(max_distortion)
    max_distortion = torch.abs(dist_embeddings - dist_scaled) / dist_scaled
    distortion_scaled += torch.mean(max_distortion)

    #create_figure(pts, poincare_64, settings=settings)
    #create_figure(pts, poincare, settings=settings)
    #create_figure(pts, poincare, settings=settings)
    end_time = time.time()
    time_taken += end_time - start_time


ic(f"{time_taken / len(seeds):.2f}")
ic(f"{distortion_orig / len(seeds):.2f}")
ic(f"{distortion_scaled / len(seeds):.2f}")
