
import torch
import numpy as np
from src.manifolds import PoincareBall
from src.utils.visualize_poincare import create_figure
import time


torch.manual_seed(48) # 46, 48
dtype = torch.float32
manifold = PoincareBall(c=torch.tensor([3.0], dtype=dtype))
num_pts = 6
dim = 2

random_dirs = torch.normal(0, 1, size=(num_pts, dim), dtype=dtype)
random_dirs /= random_dirs.norm(p=2, dim=-1, keepdim=True)
random_radii = torch.rand((num_pts, 1), dtype=dtype).pow(1 / dim)
points = manifold.c**-0.5 * (random_dirs * random_radii)

# Transform to float32
points = points[1:4]
#points = points.to(torch.float32)
#points = points.to(torch.float64)
#manifold = PoincareBall(c=torch.tensor([3.0], dtype=torch.float64))
print(f"TYPE: {points.dtype}")


labels = np.array([0] * (num_pts // 2) + [1] * (num_pts // 2))

edges = ([i for i in range(5)], [i + 1 for i in range(5)])
edges = ([2, 1], [0, 2])
#edges = ([1], [2])

if dim == 2:
    hyperplanes = (torch.tensor([[0.0, 0.5],
                                [0.2, 0.2]],
                                dtype=dtype),
                torch.tensor([[0.0, 0.0],
                                [0.3, 0.3]],
                                dtype=dtype))
    # hyperplanes = (torch.tensor([0.0, 0.5], dtype=dtype),
    #             torch.tensor([0.0, 0.0], dtype=dtype))
else:
    hyperplanes = (torch.tensor([[0.0, 0.5, 0.0, 0.0],
                                [0.2, 0.2, 0.0, 0.0]],
                                dtype=dtype),
                torch.tensor([[0.0, 0.0, 0.0, 0.0],
                                [0.3, 0.3, 0.0, 0.0]],
                                dtype=dtype))

# print(hyperplanes[0].norm(p=2, dim=-1))
# print(hyperplanes[1].norm(p=2, dim=-1))
# print(1/manifold.c.sqrt())

fig = create_figure(points, manifold, labels=None, edges=edges, hyperplanes=None)


# print("------------------------------")
# ## Float64
# dir64 = manifold.addition(-points[edges[0]], points[edges[1]], backproject=True)
# print(f"dir64: {dir64}")

# test64 = manifold.expmap(manifold.logmap(dir64, points[edges[0]]), points[edges[0]])
# print(f"test64: {test64}")

# temp64 = manifold.scalar_mul(torch.tensor(1.0), dir64, backproject=True)
# print(f"1.0 * dir64: {temp64}")

# end64 = manifold.addition(points[edges[0]], temp64, backproject=True)
# print(end64)




# ## Float32
# points = points.to(torch.float32)
# dir32 = manifold.addition(-points[edges[0]], points[edges[1]], backproject=True)
# print(f"dir32: {dir32}")

# test32 = manifold.expmap(manifold.logmap(dir32, points[edges[0]]), points[edges[0]])
# print(f"test32: {test32}")

# temp32 = manifold.scalar_mul(torch.tensor(1.0), dir32, backproject=True)
# print(f"1.0 * dir32: {temp32}")

# end32 = manifold.addition(points[edges[0]], temp32, backproject=True)
# print(end32)


# print(f"correct: {points[edges[1]]}")


# for _ in range(100):
#     manifold2 = PoincareBall(c=torch.tensor([3.0], dtype=dtype))
#     num_pts = 200
#     dim = 4
#     random_dirs1 = torch.normal(0, 1, size=(num_pts, dim), dtype=dtype)
#     random_dirs1 /= random_dirs1.norm(p=2, dim=-1, keepdim=True)
#     random_radii1 = torch.rand((num_pts, 1), dtype=dtype).pow(1 / dim)
#     points21 = manifold2.c**-0.5 * (random_dirs1 * random_radii1)
#     labels21 = [0] * (num_pts // 2) + [1] * (num_pts // 2)
#     labels21 = np.array(labels21)
#     edges22 = torch.tensor([[i for i in range(5)], [i + 1 for i in range(5)]])

#     random_dirs2 = torch.normal(0, 1, size=(num_pts, dim), dtype=dtype)
#     random_dirs2 /= random_dirs2.norm(p=2, dim=-1, keepdim=True)
#     random_radii2 = torch.rand((num_pts, 1), dtype=dtype).pow(1 / dim)
#     points22 = manifold2.c**-0.5 * (random_dirs2 * random_radii2)


#     #fig1 = create_figure(points11, manifold1, edges=edges11)
#     fig2 = create_figure(points21, manifold2, labels=labels21, edges=edges22)





# def embedddings_plot(x: torch.Tensor, y: torch.Tensor, manifold: Type[Manifold], class_names=None) -> plt.Figure:
#     if x.shape[1] > 2:
#         pairwise_dists = compute_pairwise_distances(x, manifold).detach().cpu().numpy()
#         x = TSNE(n_components=2, metric="precomputed", init="random").fit_transform(pairwise_dists)
#     else:
#         x = x.detach().cpu().numpy()

#     # Ensure that labels are also on the CPU and numpy
#     y = y.cpu().numpy()

#     # Create figure
#     plt.figure(figsize=(12, 8))
#     ax = plt.gca()

#     # Get unique classes for proper legend handling
#     unique_classes = np.unique(y)

#     # Create scatter plot
#     scatter = plt.scatter(x[:, 0], x[:, 1], c=y, alpha=0.6, cmap='tab10')

#     # Add legend with labels
#     if class_names is not None:
#         # Create custom legend with class names
#         from matplotlib.lines import Line2D
#         legend_elements = [Line2D([0], [0], marker='o', color='w',
#                                   markerfacecolor=plt.cm.tab10(i / 10),
#                                   label=class_names[i][0],
#                                   markersize=10
#                                   ) for i in unique_classes]
#         ax.legend(handles=legend_elements, title="Classes", loc="upper right")
#     else:
#         # Use default numeric labels
#         ax.legend(*scatter.legend_elements(), title="Classes", loc="upper right")

#     # Set equal aspect ratio and expand limits to show full spread
#     ax.set_aspect('equal')

#     # Get the maximum absolute value for x and y coordinates
#     max_val = max(abs(x).max(), abs(y).max())
#     margin = max_val * 0.2  # Add 10% margin

#     # Set limits symmetrically
#     ax.set_xlim(-max_val - margin, max_val + margin)
#     ax.set_ylim(-max_val - margin, max_val + margin)

#     # Add title
#     plt.title("Hyperbolic Embeddings")

#     return plt.gcf()