import torch
from src.hyperbolic_math.src.utils.math_utils import sinh, arsinh, tanh, cosh
from src.hyperbolic_math.src.manifolds.poincare import PoincareBall
from src.hyperbolic_math.src.utils.vis_utils import create_figure
from matplotlib import pyplot as plt
import os
from icecream import ic


def compute_lca_depth(manifold: PoincareBall, points: torch.Tensor, batch_size: int=1_000, debug=False) -> torch.Tensor:
    """Corrected implementation of Chami et al. (2020) LCA depth computation."""
    device = points.device
    lcas = torch.zeros((points.shape[0], points.shape[0]), dtype=points.dtype).to(device)
    indices = torch.triu_indices(points.shape[0], points.shape[0], 1).to(device)
    while indices.shape[1] > 0:
        x = points[indices[0,:batch_size], :]
        y = points[indices[1,:batch_size], :]

        xy = (x * y).sum(dim=-1, keepdim=True)
        xy_norm_prod = (x.norm(p=2, dim=-1, keepdim=True) * y.norm(p=2, dim=-1, keepdim=True)).clamp(min=1e-15)
        sinh_x = sinh(manifold.dist_0(x))
        sinh_y = sinh(manifold.dist_0(y))
        sinh_xy = sinh(manifold.dist(x, y))
        sin_gamma = (1 - (xy / xy_norm_prod) ** 2).clamp(min=0).sqrt()
        sin_gamma = torch.sin(torch.acos(xy / xy_norm_prod))

        sin_alpha = sin_gamma * sinh_y / sinh_xy
        sin_beta = sin_gamma * sinh_x / sinh_xy

        if debug:
            print(f"sinh_x: {sinh_x}, sinh_y: {sinh_y}, sinh_xy: {sinh_xy}")
            print(f"sin_gamma: {sin_gamma}, sin_alpha: {sin_alpha}, sin_beta: {sin_beta}")

        # LCA depth is only correct if the LCA lies on the geodesic segment connecting x and y
        lca_depth = arsinh(sin_alpha * sinh_x)

        if debug:
            print(f"Uncorrected LCA depth: {lca_depth}")

        # Take the closest point to the orthogonal projection of the origin that lies on the geodesic segment
        lca_depth = torch.where(sin_alpha > 0, lca_depth, torch.min(manifold.dist_0(x), manifold.dist_0(y)))
        lca_depth = torch.where(sin_beta > 0, lca_depth, torch.min(manifold.dist_0(x), manifold.dist_0(y)))
        lca_depth = torch.where(torch.logical_and(sin_gamma == 0, xy < 0), torch.zeros_like(lca_depth), lca_depth)

        if debug:
            print(f"x_dist: {manifold.dist_0(x)}]")
            print(f"y_dist: {manifold.dist_0(y)}]")
            print(f"Final LCA depth: {lca_depth}")

        lca_depth = lca_depth.reshape(-1)
        lcas[indices[0,:batch_size], indices[1,:batch_size]] = lca_depth
        lcas[indices[1,:batch_size], indices[0,:batch_size]] = lca_depth
        indices = indices[:, batch_size:]

        print(f"alpha: {torch.asin(sin_alpha)}, beta: {torch.asin(sin_beta)}, gamma: {torch.asin(sin_gamma)}")
    return lcas

def compute_lca_depth_new(manifold: PoincareBall, points: torch.Tensor, batch_size: int=1_000) -> torch.Tensor:
    """Corrected implementation of Chami et al.'s (2020) LCA depth computation."""
    device = points.device
    sqrt_c = manifold.c.sqrt()
    lcas = torch.zeros((points.shape[0], points.shape[0]), dtype=points.dtype).to(device)
    indices = torch.triu_indices(points.shape[0], points.shape[0], 1).to(device)
    while indices.shape[1] > 0:
        x = points[indices[0,:batch_size], :]
        y = points[indices[1,:batch_size], :]

        # Compute the cosine of all angles within the triangle {origin,x,y}
        # Alpha = subtended angle at x
        # Beta = subtended angle at y
        # Gamma = subtended angle at the origin
        dist_x = manifold.dist_0(x)
        dist_y = manifold.dist_0(y)
        dist_xy = manifold.dist(x, y, backproject=False)

        cosh_cx = cosh(sqrt_c * dist_x)
        cosh_cy = cosh(sqrt_c * dist_y)
        cosh_cxy = cosh(sqrt_c * dist_xy)

        sinh_cx = sinh(sqrt_c * dist_x)
        sinh_cy = sinh(sqrt_c * dist_y)
        sinh_cxy = sinh(sqrt_c * dist_xy)

        # Hyperbolic law of cosines
        cos_alpha = (cosh_cx * cosh_cxy - cosh_cy) / (sinh_cx * sinh_cxy)
        cos_beta = (cosh_cy * cosh_cxy - cosh_cx) / (sinh_cy * sinh_cxy)

        # print(f"PRE: cos_alpha: {cos_alpha}, cos_beta: {cos_beta}")
        # cos_alpha -= torch.floor(cos_alpha/(torch.pi*2)) * torch.pi*2
        # cos_beta -= torch.floor(cos_beta/(torch.pi*2)) * torch.pi*2
        # print(f"PRE2: cos_alpha: {cos_alpha}, cos_beta: {cos_beta}")
        # cos_alpha -= torch.pi
        # cos_beta -= torch.pi
        # print(f"POST: cos_alpha: {cos_alpha}, cos_beta: {cos_beta}")

        xy = (x * y).sum(dim=-1, keepdim=True)
        xy_norm_prod = (x.norm(p=2, dim=-1, keepdim=True) * y.norm(p=2, dim=-1, keepdim=True)).clamp(min=1e-16)
        cos_gamma = xy / xy_norm_prod

        # Compute the lca depth w.r.t. the full geodesic
        sin_alpha = (1 - cos_alpha ** 2).clamp(min=0).sqrt()
        # Hyperbolic law of sines
        lca_depth = arsinh(sinh(sqrt_c * dist_x) * sin_alpha) / sqrt_c

        # Get the lca depth w.r.t. the geodesic segment between x and y
        lca_depth = torch.where(cos_alpha > 0, lca_depth, dist_x)
        lca_depth = torch.where(cos_beta > 0, lca_depth, dist_y)
        lca_depth = torch.where(cos_gamma == -1, torch.zeros_like(lca_depth), lca_depth)

        lca_depth = lca_depth.reshape(-1)
        lcas[indices[0,:batch_size], indices[1,:batch_size]] = lca_depth
        lcas[indices[1,:batch_size], indices[0,:batch_size]] = lca_depth
        indices = indices[:, batch_size:]

    return lcas

def compute_lca_depth_chami(manifold: PoincareBall, points: torch.Tensor, batch_size: int=1_000) -> torch.Tensor:
    """LCA depth computation of Chami et al. (2020)."""
    device = points.device
    lcas = torch.zeros((points.shape[0], points.shape[0]), dtype=points.dtype).to(device)
    indices = torch.triu_indices(points.shape[0], points.shape[0], 1).to(device)
    while indices.shape[1] > 0:
        x = points[indices[0,:batch_size], :]
        y = points[indices[1,:batch_size], :]

        # Chami LCA computation
        # 1) reflection_center(x)
        r = x / torch.sum(x ** 2, dim=-1, keepdim=True)
        # 2) isometric_transform(r, y)
        r2 = torch.sum(r ** 2, dim=-1, keepdim=True) - 1.
        u = y - r
        b_inv =  r2 / torch.sum(u ** 2, dim=-1, keepdim=True) * u + r
        # 3) Copy
        o_inv = x
        # 4) euc_reflection(o_inv, b_inv)
        xTa = torch.sum(o_inv * b_inv, dim=-1, keepdim=True)
        norm_a_sq = torch.sum(b_inv ** 2, dim=-1, keepdim=True).clamp_min(1e-15)
        proj = xTa * b_inv / norm_a_sq
        o_inv_ref = 2 * proj - o_inv
        # 5) isometric_transform(r, o_inv_ref)
        r2_ = torch.sum(r ** 2, dim=-1, keepdim=True) - 1.
        u_ = o_inv_ref - r
        o_ref = r2_ / torch.sum(u_ ** 2, dim=-1, keepdim=True) * u_ + r
        # 6) _halve(o_ref)
        proj = o_ref / (1. + torch.sqrt(1 - torch.sum(o_ref ** 2, dim=-1, keepdim=True)))
        # 7) manifold.dist_0(proj)
        lca_depth = manifold.dist_0(proj)

        lca_depth = lca_depth.reshape(-1)
        lcas[indices[0,:batch_size], indices[1,:batch_size]] = lca_depth
        lcas[indices[1,:batch_size], indices[0,:batch_size]] = lca_depth
        indices = indices[:, batch_size:]
    return lcas

def compute_lca_depth_chami_circle_inv(manifold: PoincareBall, points: torch.Tensor, batch_size: int=1_000) -> torch.Tensor:
    device = points.device
    lcas = torch.zeros((points.shape[0], points.shape[0]), dtype=points.dtype).to(device)
    indices = torch.triu_indices(points.shape[0], points.shape[0], 1).to(device)
    while indices.shape[1] > 0:
        x = points[indices[0,:batch_size], :]
        y = points[indices[1,:batch_size], :]

        # Compute the center of geodesic circle Gamma through x and y
        x2 = x.pow(2).sum(dim=-1, keepdim=True)
        y2 = y.pow(2).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)

        denom = 2 * manifold.c * (x2 * y2 - xy ** 2)
        alpha = ((1 + manifold.c * x2) * y2 - (1 + manifold.c * y2) * xy) / denom
        beta = ((1 + manifold.c * y2) * x2 - (1 + manifold.c * x2) * xy) / denom
        z = alpha * x + beta * y

        # Compute the lca and its distance to the origin w.r.t. the full geodesic
        z2 = z.pow(2).sum(dim=-1, keepdim=True)
        lca = (1 - (z2 - 1/manifold.c).sqrt() / z.norm(p=2, dim=-1, keepdim=True)) * z
        lca_depth = manifold.dist_0(lca)

        # Compute the cosine of the angles at the origin in the triangles {x, origin, c}, {y, origin, c}  {x, origin, y}
        xlca = (x * lca).sum(dim=-1, keepdim=True)
        ylca = (y * lca).sum(dim=-1, keepdim=True)
        lca_norm = lca.norm(p=2, dim=-1, keepdim=True)

        cos_x0c = xlca / (x2.sqrt() * lca_norm).clamp(min=1e-16)
        cos_y0c = ylca / (y2.sqrt() * lca_norm).clamp(min=1e-16)
        cos_x0y = xy / (x2.sqrt() * y2.sqrt()).clamp(min=1e-16)

        # Get the lca depth w.r.t. the geodesic segment between x and y
        condition = torch.logical_or(torch.logical_or(cos_x0y > cos_x0c, cos_x0y > cos_y0c), cos_x0y == 1) # True iff the lca lies outside
        lca_depth = torch.where(condition, torch.min(manifold.dist_0(x), manifold.dist_0(y)), lca_depth)
        lca_depth = torch.where(cos_x0y == -1, torch.zeros_like(lca_depth), lca_depth) # Handle antipodal points

        lca_depth = lca_depth.reshape(-1)
        lcas[indices[0,:batch_size], indices[1,:batch_size]] = lca_depth
        lcas[indices[1,:batch_size], indices[0,:batch_size]] = lca_depth
        indices = indices[:, batch_size:]
    return lcas

def plot_routine(res_ours, res_chami, printing=True):
    if printing:
        print("---------------------")
        print(f"Res_ours: {res_ours} \nRes_chami: {res_chami}")

    x_dist_str = f"{manifold.dist_0(points[0]).item():.2f}"
    y_dist_str = f"{manifold.dist_0(points[1]).item():.2f}"
    fig = create_figure(points, manifold,
                        #labels=[f'x: {x_dist_str}', f'y: {y_dist_str}'],
                        edges=([0], [1])
                        )

    ax = fig.gca()
    res_ours_converted = tanh(manifold.c.sqrt() * res_ours / 2) / manifold.c.sqrt()
    res_chami_converted = tanh(manifold.c.sqrt() * res_chami / 2) / manifold.c.sqrt()

    for i in range(res_ours.shape[0]):
        for j in range(i+1, res_ours.shape[1]):
            circle_ours = plt.Circle((0, 0), res_ours_converted[i,j].cpu().numpy(), fill=False, color='green', zorder=5, linewidth=0.5)
            circle_chami = plt.Circle((0, 0), res_chami_converted[i,j].cpu().numpy(), fill=False, color='red', zorder=6, linestyle='--', linewidth=0.5)
            ax.add_artist(circle_ours)
            ax.add_artist(circle_chami)

    # Save the figure
    file_path = os.getcwd()
    path = os.path.join(file_path, 'images')
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, "hyperbolic_embeddings.png"))
    plt.close(fig)



##########
dtype = torch.float32
version_1 = "compute_lca_depth_chami_circle_inv"
version_2 = "compute_lca_depth_chami"
##########


manifold = PoincareBall(c=torch.tensor([1.0], dtype=dtype))
opposites = torch.tensor([[0.5, 0.0], [-0.6, 0.0]], dtype=dtype) # res = 0
same_side_x = torch.tensor([[0.5, 0.0], [0.7, 0.0]], dtype=dtype) # res = x
same_side_y = torch.tensor([[0.7, 0.0], [0.5, 0.0]], dtype=dtype) # res = y

opposites_flipped = torch.tensor([[-0.6, 0.0], [0.5, 0.0]], dtype=dtype) # res = 0
same_side_y_flipped = torch.tensor([[0.7, 0.0], [0.5, 0.0]], dtype=dtype) # res = y
same_side_x_flipped = torch.tensor([[0.5, 0.0], [0.7, 0.0]], dtype=dtype) # res = x

p1 = torch.tensor([[ 0.0706, -0.9943],
        [ 0.8838,  0.0425]], dtype=dtype)

p2 = torch.tensor([[ -0.1, -0.5],
        [ -0.4,  -0.8]], dtype=dtype)

p3 = torch.tensor([[ 0.3, 0.],
        [ 0.2,  -0.007]], dtype=dtype)

##########
points = same_side_x_flipped
##########


# if version_1 == "compute_lca_depth_chami":
#     res_1 = compute_lca_depth_chami(manifold, points)
# elif version_1 == "compute_lca_depth_chami_circle_inv":
#     res_1 = compute_lca_depth_chami_circle_inv(manifold, points)

# if version_2 == "compute_lca_depth_chami":
#     res_2 = compute_lca_depth_chami(manifold, points)
# elif version_2 == "compute_lca_depth_chami_circle_inv":
#     res_2 = compute_lca_depth_chami_circle_inv(manifold, points)

# plot_routine(res_ours=res_1, res_chami=res_2, printing=True)


torch.manual_seed(48)
count = 0
for i in range(100_000):
    random_dirs = torch.normal(0, 1, size=(2, 2), dtype=dtype)
    random_dirs /= random_dirs.norm(p=2, dim=-1, keepdim=True)
    random_radii = torch.rand((2, 1), dtype=dtype).pow(1 / 2)
    points = manifold.c**-0.5 * (random_dirs * random_radii)

    if version_1 == "compute_lca_depth_chami":
        res_1 = compute_lca_depth_chami(manifold, points)
    elif version_1 == "compute_lca_depth_chami_circle_inv":
        res_1 = compute_lca_depth_chami_circle_inv(manifold, points)

    if version_2 == "compute_lca_depth_chami":
        res_2 = compute_lca_depth_chami(manifold, points)
    elif version_2 == "compute_lca_depth_chami_circle_inv":
        res_2 = compute_lca_depth_chami_circle_inv(manifold, points)

    if not torch.allclose(res_1, res_2, atol=1e-5):
        #print("DIFFERENT")
        #print(f"diff: {torch.abs(res_1 - res_2).max()}")
        #plot_routine(res_1, res_2)
        print(i)
        count += 1
        #break
        #time.sleep(4)
print(count)
