import time
import torch

from icecream import ic
from matplotlib import pyplot as plt
from src.manifolds import PoincareBall
from src.utils.helpers import compute_pairwise_distances
from src.utils.horo_pca import HoroPCA


if torch.cuda.is_available():
        torch.set_default_device('cuda')
############### Parameters for testing ##########
seeds = [*range(40, 41)]
dtype = torch.float32
curvature = 1.0
input_dim = 2
output_dim = 2
num_pts = 1_000
max_steps = 100
#################################################
poincare = PoincareBall(c=torch.tensor([curvature], dtype=dtype))
avg_max_distortion = torch.tensor(0.0, dtype=dtype, device='cuda' if torch.cuda.is_available() else 'cpu')
time_taken = 0.0
for seed in seeds:
    torch.manual_seed(seed)
    # Generate random data in PoincareBall
    random_dirs = torch.normal(0, 1, size=(num_pts, input_dim), dtype=poincare.dtype, device=poincare.c.device)
    random_dirs /= random_dirs.norm(p=2, dim=-1, keepdim=True)
    random_radii = torch.rand((num_pts, 1), dtype=poincare.dtype).pow(1 / input_dim)
    points = poincare.c**-0.5 * (random_dirs * random_radii) * 0.9
    # Compute the original pairwise distances
    dist_original = compute_pairwise_distances(points, poincare) + torch.eye(num_pts)
    poincare_closure = (1 / poincare.c.sqrt()).detach().cpu().numpy()
    limit = (-1.1 * poincare_closure, 1.1 * poincare_closure)
    # Run dimensionality reduction methods
    start_time = time.time()
    horoPCA_model = HoroPCA(n_components=output_dim, n_in_features=points.shape[1], manifold=poincare, max_steps=max_steps)
    horoPCA_model.fit(points)
    embeddings = horoPCA_model.transform(points)
    end_time = time.time()
    time_taken += end_time - start_time
    # Compute the pairwise distances of the dimensionality reduced data points
    dist_embeddings = compute_pairwise_distances(embeddings, poincare) + torch.eye(num_pts)
    max_distortion = torch.abs(dist_embeddings - dist_original) / dist_original
    avg_max_distortion += torch.mean(max_distortion)
    # Plot the dimensionality reduced data points
    if input_dim == 2:
        # Get the hyperplanes from the fitted model
        principals = horoPCA_model.Q
        principals_ortho, _ = torch.linalg.qr(principals.T, mode='reduced')
        principals_ortho = principals_ortho.T / poincare.c.sqrt()
        # Plot the original points
        fig = plt.figure(figsize=(8, 8))
        plt.title('Original data points')
        plt.xlim(*limit)
        plt.ylim(*limit)
        circle = plt.Circle((0, 0), radius=poincare_closure, color='black', fill=False)
        plt.gca().add_patch(circle)
        plt.scatter(points[:, 0].detach().cpu().numpy(),
                    points[:, 1].detach().cpu().numpy(),
                    color='blue', label='x')
        plt.quiver(-principals_ortho[0, 0].detach().cpu().numpy(),
                   -principals_ortho[0, 1].detach().cpu().numpy(),
                   2*principals_ortho[0, 0].detach().cpu().numpy(),
                   2*principals_ortho[0, 1].detach().cpu().numpy(),
                   angles='xy', scale_units='xy', scale=1,
                   color='red', label='Principal Component 1')
        plt.quiver(-principals_ortho[1, 0].detach().cpu().numpy(),
                   -principals_ortho[1, 1].detach().cpu().numpy(),
                   2*principals_ortho[1, 0].detach().cpu().numpy(),
                   2*principals_ortho[1, 1].detach().cpu().numpy(),
                   angles='xy', scale_units='xy', scale=1,
                   color='green', label='Principal Component 2')
        plt.legend(loc='upper right')
        plt.grid()
        plt.savefig('horoPCA_points.png')
        plt.close()
    if output_dim == 2:
        # Plot the dimensionality reduced data points
        fig = plt.figure(figsize=(8, 8))
        plt.title('HoroPCA embedded data points')
        plt.xlim(*limit)
        plt.ylim(*limit)
        circle = plt.Circle((0, 0), radius=poincare_closure, color='black', fill=False)
        plt.gca().add_patch(circle)
        plt.scatter(embeddings[:, 0].detach().cpu().numpy(), embeddings[:, 1].detach().cpu().numpy(), color='blue', label='x')
        plt.legend(loc='upper right')
        plt.grid(alpha=0.5)
        plt.savefig('horoPCA_embeddings.png')
        plt.close()

ic(avg_max_distortion / len(seeds))
ic(f"{time_taken / len(seeds):.2f}")
