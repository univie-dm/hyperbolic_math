import matplotlib.pyplot as plt
from src.manifolds import PoincareBall
import torch
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import time
from src.manifolds.embedding import Embedding
import copy


torch.manual_seed(48)
dtype = "float32"
manifold_dtype = "float32"
num_pts = 500_000
input_dim = 2
output_dim = 1
backproject = False
normal_plot = 0 # only plot the first normal
if dtype == "float32":
    torch_dtype = torch.float32
elif dtype == "float64":
    torch_dtype = torch.float64
manifold = PoincareBall(c=torch.tensor([1.0], dtype=torch_dtype), dtype=manifold_dtype)

# Tangent points
random_dirs = torch.normal(0, 1, size=(num_pts, input_dim), dtype=torch_dtype)
random_dirs /= random_dirs.norm(p=2, dim=-1, keepdim=True)
random_radii = torch.rand((num_pts, 1), dtype=torch_dtype).pow(1 / input_dim)
points_manifold = manifold.c**-0.5 * (random_dirs * random_radii)
points = manifold.logmap_0(points_manifold, dim=-1) # (num_pts, input_dim)

# Tangent normals at the origin
#tangent_origin_weights = torch.randn(output_dim, input_dim, dtype=torch_dtype)
tangent_origin_weights = torch.tensor([[ 1., 1.],
                                    #[ 1., 0.],
                                    #[ 1., 1.]
                                    ])
unit_tangent_origin_weights = tangent_origin_weights / tangent_origin_weights.norm(p=2, dim=-1, keepdim=True) # (output_dim, input_dim)

# Bias with input dimension (that lies in the tangent space at the origin)
r = torch.zeros((output_dim, 1), dtype=torch_dtype)
#r = torch.ones((output_dim, 1), dtype=torch_dtype) * 0.7
ebias_in = r * unit_tangent_origin_weights # (output_dim, input_dim)
mbias_in = manifold.expmap_0(ebias_in, dim=-1) # (output_dim, input_dim)

# Tangent normals that lies in the tangent space at the bias with input dimension
tangent_mbias_in_weights = manifold.ptransp_0(tangent_origin_weights, mbias_in, dim=-1) # (output_dim, input_dim)

# Bias with output dimension (lies in the tangent space at the origin)
ebias_out = (ebias_in * tangent_origin_weights).sum(dim=-1, keepdim=True) # (output_dim, 1)
mbias_out = manifold.expmap_0(ebias_out.T, dim=-1) # (1, output_dim)

# Origin
origin = torch.zeros((input_dim,), dtype=torch_dtype)

### PLOTS
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
for i in range(2):
    for j in range(3):
        axs[i,j].set_xlim(-1.1*manifold.c.sqrt(), 1.1*manifold.c.sqrt())
        axs[i,j].set_ylim(-1.1*manifold.c.sqrt(), 1.1*manifold.c.sqrt())
        #axs[i,j].grid(alpha=0.4)

results = []
method_names = [
    # Hyperbolic Reinforcement Learning (HRL)
    "HRL_forward",
    "HRL_forward_rs",
    # Hyperbolic Neural Networks (HNN)
    "HNN_FC",
    "HNN_MLR",
    # Hyperbolic Neural Networks ++ (HNNpp)
    "HNNpp_FC",
    "HNNpp_MLR"
]

for i, method_name in enumerate(method_names):
    start = time.time()
    embedding = Embedding(input_dim, output_dim, manifold, params_dtype=dtype, requires_grad=False,
                          forward_method=method_name, backproject=backproject)
    # Set a unified weight and bias
    if method_name in ["HNN_MLR", "HRL_forward", "HRL_forward_rs"]:
        embedding.weight = torch.nn.Parameter(tangent_mbias_in_weights, requires_grad=False) # (output_dim, input_dim)
        embedding.bias = torch.nn.Parameter(mbias_in, requires_grad=False) # (output_dim, input_dim)
    elif method_name in ["HNN_FC"]:
        embedding.weight = torch.nn.Parameter(tangent_origin_weights, requires_grad=False) # (output_dim, input_dim)
        embedding.bias = torch.nn.Parameter(mbias_out, requires_grad=False) # (1, output_dim)
    elif method_name in ["HNNpp_FC", "HNNpp_MLR"]:
        embedding.weight = torch.nn.Parameter(tangent_origin_weights, requires_grad=False) # (output_dim, input_dim)
        embedding.bias = torch.nn.Parameter(r, requires_grad=False) # (output_dim, 1)
    res = embedding.forward(points)
    results.append(res)
    print(f"Elapsed time - {method_name}: {time.time()-start}")

#exit()

# Create the colormap
colors = ['blue', 'white', 'red']
cmap_name = 'custom_diverging'
cmap = LinearSegmentedColormap.from_list(cmap_name, colors)

# Define the fixed value range and the number of bins
value_range = torch.linspace(-1, 1, 11)
norm = BoundaryNorm(value_range, ncolors=cmap.N, clip=True)

# Plot the scalar function values
spacing = 10_000
t = torch.linspace(-500, 500, spacing)
if input_dim == 2:
    normal = manifold.expmap(tangent_mbias_in_weights, mbias_in, dim=-1)
    dir = copy.deepcopy(normal)
    dir[:, 0] = -normal[:, 1]
    dir[:, 1] = normal[:, 0]

    normal = normal - mbias_in
    for i, result in enumerate(results):
        sc1 = axs[i//3,i%3].scatter(points_manifold[:,0], points_manifold[:,1], c=result[:,normal_plot],
                                    cmap=cmap, s=0.5, zorder=1, norm=norm)
        # Plot the hyperplane normal (and bias)
        if torch.allclose(mbias_in, torch.zeros_like(mbias_in)):
            axs[i//3,i%3].quiver(origin[0], origin[1],
                                 normal[normal_plot,0], normal[normal_plot,1],
                                 color='green', angles='xy', scale_units='xy', scale=1)
        else:
            axs[i//3,i%3].quiver(origin[0], origin[1],
                                 mbias_in[normal_plot,0], mbias_in[normal_plot,1],
                                 color='black', angles='xy', scale_units='xy', scale=1)
            axs[i//3,i%3].quiver(mbias_in[normal_plot,0], mbias_in[normal_plot,1],
                                 normal[normal_plot,0], normal[normal_plot,1],
                                 color='green', angles='xy', scale_units='xy', scale=1)
        # Plot the hyperplane
        dirs = torch.outer(t, dir[normal_plot]) # (spacing, input_dim)
        plot_anchors = manifold.expmap(dirs, mbias_in[normal_plot].unsqueeze(0), dim=-1)
        axs[i//3,i%3].plot(plot_anchors[:, 0], plot_anchors[:, 1], c='green', alpha=0.6, zorder=3)
        # Set the title of the subplot
        axs[i//3,i%3].title.set_text(f'{method_names[i]}')

    fig.colorbar(sc1, ax=axs, orientation='vertical', fraction=0.05, pad=0.05)
    plt.savefig('forward_methods_comparison.png')
    plt.close()

