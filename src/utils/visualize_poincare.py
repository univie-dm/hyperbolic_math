import os
import torch
import numpy as np
import numpy.typing as npt

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Type, Tuple, Union
from ..manifolds import Manifold, Hyperboloid
from .helpers import compute_pairwise_distances


def create_figure(points: torch.Tensor,
                  manifold: Type[Manifold],
                  labels: Union[npt.ArrayLike, None]=None,
                  edges: Union[torch.Tensor, None]=None,
                  hyperplanes: Union[torch.Tensor, None]=None
                  ) -> None:
    # Make settings adjustable
    settings = {
        "method": "PCA",
        "title": "Hyperbolic Embeddings",
        "show_origin": True,
        "save_figure": True,
        "file_name": "hyperbolic_embeddings",
        "file_path": None,
        "file_format": "png",
    }
    assert manifold.is_in_manifold(points), "Points are not in the manifold"

    # Move everything to the CPU
    points = points.cpu().detach().numpy()
    manifold_closure = 1 / manifold.c.sqrt().cpu().detach().numpy()

    # Create figure
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # Add title
    plt.title(settings["title"])

    # Set equal aspect ratio and limits
    ax.set_aspect('equal')
    ax.set_xlim(-1.1 * manifold_closure, 1.1 *manifold_closure)
    ax.set_ylim(-1.1 * manifold_closure, 1.1 *manifold_closure)

    # Draw boundary circle
    circle = plt.Circle((0, 0), manifold_closure, fill=False, color='black')
    ax.add_artist(circle)

    # Draw origin
    if settings["show_origin"]:
        ax.scatter(np.array(0), np.array(0), marker="x", c="black")

    if isinstance(manifold, Hyperboloid):
        # TODO: Project manifold points onto PoincareBall first
        raise NotImplementedError("Hyperboloid manifold not supported yet")

    # Project points to 2d
    if points.shape[-1] > 2:
        points = pointsTo2d(points, manifold, method=settings["method"])
    # TODO: Sometimes projected points are no longer on the manifold
    # Implement PCA without normalization to avoid this
    assert manifold.is_in_manifold(torch.from_numpy(points)), "Points are not in the manifold"

    # Plot geodesics between points
    if edges is not None:
        plot_edges(points[edges[0],:], points[edges[1],:], manifold, plt.gca())

    # Plot the manifold points
    plot_2d_points(points, plt.gca(), labels)

    # Plot hyperplanes
    if hyperplanes is not None:
        plot_hyperplane(hyperplanes[:, 0], hyperplanes[:, 1], manifold, plt.gca())

    # Save the figure
    if settings["save_figure"]:
        save_figure(plt.gcf(), settings["file_name"], file_path=settings["file_path"], format=settings["file_format"])

def pointsTo2d(x: npt.ArrayLike, manifold: Type[Manifold], method: str="PCA") -> npt.ArrayLike:
    """Project the points x to 2d using the specified method."""
    if method == "PCA":
        res = PCA(n_components=2
                  #, power_iteration_normalizer='none' ## is this different from PCA w/o normalization
                  ).fit_transform(x)
    elif method == "tSNE":
        pairwise_dists = compute_pairwise_distances(x, manifold)
        res = TSNE(n_components=2, metric="precomputed", init="random").fit_transform(pairwise_dists)
    else:
        raise ValueError(f"Unknown method {method}")
    return res

def plot_2d_points(x: npt.ArrayLike, ax: plt.Axes, labels: Union[npt.ArrayLike, None]=None) -> None:
    """Plot the 2d manifold points with labels (optional)."""
    if labels is None:
        ax.scatter(x[:, 0], x[:, 1], c="b", alpha=0.6)
    else:
        assert x.shape[0] == len(labels), "Number of labels must match number of points"
        # Get unique display labels and map to colors
        unique_labels = np.unique(labels)
        if len(unique_labels) <= 10:
            cmap = plt.cm.tab10
        else:
            cmap = plt.cm.tab20
        colors = cmap(np.linspace(0, 1, len(unique_labels)))
        color_map = dict(zip(unique_labels, colors))

        # Plot points
        ax.scatter(x[:, 0], x[:, 1], c=[color_map[label] for label in labels], alpha=0.6)

        # Add legend outside of the circle
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[label],
                              markersize=10, alpha=0.6, label=label)
                   for label in unique_labels]
        ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

def plot_edges(x: npt.ArrayLike, y: npt.ArrayLike, manifold: Type[Manifold], ax: plt.Axes) -> None:
    """Plot the geodesic segment(s) connecting the pair(s) x and y."""
    assert x.shape == y.shape, "Start and end points must have the same shape"
    spacing = 20
    device = manifold.c.device
    t = torch.linspace(0, 1, spacing, device=device).reshape(-1, 1)
    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)
    # Compute the geodesics and plot them
    for _x,_y in zip(x, y):
        points = manifold.geodesic_segment(t, _x.repeat(spacing, 1), _y.repeat(spacing, 1))
        points = points.cpu().detach()
        ax.plot(points[:, 0], points[:, 1], c="b", alpha=0.6)

def plot_hyperplane(x: torch.Tensor, normal: torch.Tensor, manifold: Type[Manifold], ax: plt.Axes) -> None:
    """Plot 1-dimensional hyperplane(s) defined by point(s) x and the normal vector(s)."""
    # TODO: finish implementation
    spacing = 40
    device = manifold.c.device
    t = torch.linspace(-100, 100, spacing, device=device).reshape(-1, 1)
    x = torch.from_numpy(x).to(device)
    v = torch.from_numpy(normal).to(device)
    # Compute the geodesics and plot them
    for _x,_v in zip(x, v):
        print(t.shape)
        print(_v.shape)
        print((t*_v).shape)
        
        points = manifold.expmap(t*_v, _x.repeat(spacing, 1)).cpu().detach()
        ax.plot(points[:, 0], points[:, 1], c="b", alpha=0.6)



def save_figure(fig: plt.Figure, file_name: str, file_path: Union[str, None]=None,  format: str="png") -> None:
    """Save the figure to a file."""
    if file_path is None:
        path = os.getcwd()
    path = os.path.join(path, "images")
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, f"{file_name}.{format}"))
