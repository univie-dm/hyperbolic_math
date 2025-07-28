import torch
import io
import cProfile
import pstats
from src.utils.helpers import compute_pairwise_distances
from src.utils.horo_pca import center_data, compute_frechet_mean, HoroPCA
from src.manifolds import PoincareBall, Hyperboloid
from icecream import ic
from src.utils.horo_pca_chami import HoroChami
import time


if torch.cuda.is_available():
        torch.set_default_device('cuda')
############### Parameters for testing ##########
seeds = [*range(40, 41)]
dtype = torch.float32
curvature = 1.0
input_dim = 32
num_pts = 10_000
max_steps = 100
#################################################
manifold = PoincareBall(c=torch.tensor([curvature], dtype=dtype))
hyperbol = Hyperboloid(c=torch.tensor([curvature], dtype=dtype))
avg_distortion_ours = torch.tensor(0.0, dtype=dtype, device='cuda' if torch.cuda.is_available() else 'cpu')
avg_distortion_chami = torch.tensor(0.0, dtype=dtype, device='cuda' if torch.cuda.is_available() else 'cpu')
timing = 0.0
for seed in seeds:
    torch.manual_seed(seed)
    # Generate random data in Poincare ball
    random_dirs = torch.normal(0, 1, size=(num_pts, input_dim), dtype=manifold.dtype, device=manifold.c.device)
    random_dirs /= random_dirs.norm(p=2, dim=-1, keepdim=True)
    random_radii = torch.rand((num_pts, 1), dtype=manifold.dtype).pow(1 / input_dim)
    points = manifold.c**-0.5 * (random_dirs * random_radii) * 0.9
    
    # Compute the mean and center the data
    frechet_mean, has_converged = compute_frechet_mean(points, manifold, eps=1e-07)
    if not has_converged:
        print(f"Frechet mean did not converge for seed {seed}. Using initial mean instead.")
        frechet_mean = torch.mean(points, dim=0, keepdim=True)
    x_centered = center_data(points, frechet_mean)

    # Compute the original pairwise distances
    dist_original = compute_pairwise_distances(x_centered, manifold)
    dist_original += torch.eye(num_pts)

    # Run dimensionality reduction methods
    if True:
        print("OURS")
        start_time = time.time()
        model_ours = HoroPCA(n_components=2, n_in_features=x_centered.shape[1], manifold=manifold, max_steps=500)
        # Start profiling
        pr = cProfile.Profile()
        pr.enable()
        model_ours.fit(x_centered)
        ours_embeddings = model_ours.transform(x_centered)
        # Create a StringIO object to capture the output
        s = io.StringIO()
        sortby = pstats.SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        profiling_output = s.getvalue()
        output_filename = "profile_fit_ours.txt"
        with open(output_filename, "w") as f:
            f.write(profiling_output)
        end_time = time.time()

        timing += end_time - start_time
        ic(ours_embeddings.shape)
        dist_ours = compute_pairwise_distances(ours_embeddings, manifold)
        dist_ours += torch.eye(num_pts)
        temp_ours = torch.abs(dist_ours - dist_original) / dist_original
        avg_distortion_ours += torch.mean(temp_ours)
    else:
        print("CHAMI")
        start_time = time.time()
        model_chami = HoroChami(dim=x_centered.shape[1], n_components=2, lr=1e-3, max_steps=max_steps)
        model_chami.fit(x_centered)
        chami_embeddings = model_chami.map_to_ball(x_centered)
        end_time = time.time()
        timing += end_time - start_time
        ic(chami_embeddings.shape)
        dist_chami = compute_pairwise_distances(chami_embeddings, manifold)
        dist_chami += torch.eye(num_pts)
        temp_chami = torch.abs(dist_chami - dist_original) / dist_original
        avg_distortion_chami += torch.mean(temp_chami)

ic(avg_distortion_ours / len(seeds))
ic(avg_distortion_chami / len(seeds))
ic(f"{timing / len(seeds):.2f}")
