import os
import torch
import numpy as np
import numpy.typing as npt

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Any, Type, Tuple, Union, List
from ..manifolds import Manifold, Euclidean, Hyperboloid
from .helpers import compute_pairwise_distances


def create_figure(points: torch.Tensor,
                  manifold: Type[Manifold],
                  labels: Union[npt.ArrayLike, None]=None,
                  edges: Union[torch.Tensor, None]=None,
                  hyperplanes: Union[Tuple[torch.Tensor, torch.Tensor], None]=None
                  ) -> None:
    # Make settings adjustable
    settings = {
        "method": "tangent PCA",# "tangent PCA",
        "title": "Hyperbolic Embeddings",
        "show_origin": True,
        "save_figure": True,
        "file_name": "hyperbolic_embeddings",
        "file_path": None,
        "file_format": "png",
    }
    assert manifold.is_in_manifold(points), "Points are not in the manifold"

    # Move everything to the CPU
    points = points.cpu().detach()
    manifold_closure = 1 / manifold.c.sqrt().cpu().detach()

    # Create figure
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # Add title
    plt.title(settings["title"]+f" ({settings['method']})")

    # Set equal aspect ratio and limits
    ax.set_aspect('equal')
    ax.set_xlim(-1.1 * manifold_closure, 1.1 *manifold_closure)
    ax.set_ylim(-1.1 * manifold_closure, 1.1 *manifold_closure)

    # Draw boundary circle
    circle = plt.Circle((0, 0), manifold_closure, fill=False, color='black', zorder=5)
    ax.add_artist(circle)

    # Draw origin
    if settings["show_origin"]:
        if isinstance(manifold, Hyperboloid):
            ax.scatter(0, manifold_closure, marker="x", c="black", s=2, zorder=5)
        else:
            ax.scatter(0, 0, marker="x", c="black", s=50, zorder=5)

    if isinstance(manifold, Hyperboloid):
        # TODO: Project manifold points onto PoincareBall first
        raise NotImplementedError("Hyperboloid manifold not supported yet")

    # Project manifold points and hyperplane normals to 2d
    if points.shape[-1] > 2:
        points, hyperplane_basis, hyperplane_normals = pointsTo2d(points, manifold, settings, hyperplanes[0], hyperplanes[1])
    # TODO: For hyperbolic methods some results are no longer on the manifold
    #assert manifold.is_in_manifold(torch.from_numpy(points)), "Points are not in the manifold"

    # Plot the manifold points
    handles = plot_2d_points(points, plt.gca(), labels)

    # Plot geodesics between points
    if edges is not None:
        if settings["method"] in ["tangent tSNE", "hyperbolic tSNE"]:
            # tSNE for edges skews the resulting projection way too much since
            # we sample many points along edges skewing the result tremendiously
            raise NotImplementedError("tSNE for edges does not make any sense")
        plot_edges(points[edges[0]], points[edges[1]], manifold, plt.gca(), handles)

    # Plot hyperplanes
    if hyperplane_basis is not None:
        # Construct the 2d basis vector of the hyperplane
        hyperplane_normals[:, 0] = -hyperplane_normals[:, 0]
        hyperplane_dir = hyperplane_normals[:,::-1].copy()
        plot_hyperplane(hyperplane_basis, hyperplane_dir, manifold, plt.gca(), handles)

    # Save the figure
    if settings["save_figure"]:
        save_figure(plt.gcf(), settings["file_name"], file_path=settings["file_path"], format=settings["file_format"])

def pointsTo2d(x: torch.Tensor, manifold: Type[Manifold], settings: dict,
               hyperplane_normals: Union[torch.Tensor, None],
               hyperplane_basis: Union[torch.Tensor, None]
               ) -> Tuple[npt.ArrayLike, Union[npt.ArrayLike, None]]:
    """Project points and hyperplane normals to 2d using the specified method."""
    # tSNE needs to be applied jointly to points, and hyperplanes
    if hyperplane_normals is not None:
        sample_size = x.shape[0]
        hyperplane_size = hyperplane_normals.shape[0]
        x = torch.cat((x, hyperplane_normals, hyperplane_basis), dim=0)

    # Apply dimensionality reduction method
    if settings["method"] == "tangent tSNE":
        x = manifold.logmap_0(x)
        x = TSNE(n_components=2, init="random").fit_transform(x)
        x = manifold.expmap_0(torch.from_numpy(x)).numpy()
    elif settings["method"] == "hyperbolic tSNE":
        # Doesn't really make sense to use tSNE in the hyperbolic space like this
        # -> Almost all projected points are no longer on the manifold
        pairwise_dists = compute_pairwise_distances(x, manifold)
        x = TSNE(n_components=2, metric="precomputed", init="random").fit_transform(pairwise_dists)
    elif settings["method"] == "tangent PCA":
        x = manifold.logmap_0(x)
        model = PCA(n_components=2).fit(x[:sample_size])
        x = model.transform(x)
        x = manifold.expmap_0(torch.from_numpy(x)).numpy()
    elif settings["method"] == "hyperbolic PCA":
        # Doesn't really make sense to use PCA in the hyperbolic space like this
        # -> Often projected points are no longer on the manifold
        model = PCA(n_components=2).fit(x[:sample_size])
        x = model.transform(x)
    else:
        raise ValueError(f"Unknown method {settings["method"]}")

    if hyperplane_normals is not None:
        # Split data and hyperplanes
        points = x[:sample_size]
        hyperplane_normals = x[sample_size:sample_size+hyperplane_size]
        hyperplane_basis = x[sample_size+hyperplane_size:]
    else:
        points = x

    return points, hyperplane_normals, hyperplane_basis

def plot_2d_points(x: npt.ArrayLike, ax: plt.Axes, labels: Union[npt.ArrayLike, None]=None) -> List[Any]:
    """Plot the 2d manifold points with labels (optional)."""
    if labels is None:
        ax.scatter(x[:, 0], x[:, 1], c="b", alpha=0.6, zorder=4)
        handles = []
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
        ax.scatter(x[:, 0], x[:, 1], c=[color_map[label] for label in labels], alpha=0.6, zorder=4)

        # Add legend outside of the circle
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[label],
                              markersize=10, alpha=0.6, label=f"class: {label}")
                   for label in unique_labels]
        ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    return handles

def plot_edges(x: npt.ArrayLike, y: npt.ArrayLike, manifold: Type[Manifold], ax: plt.Axes,
               handles: List[Any]) -> None:
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
        ax.plot(points[:, 0], points[:, 1], c="b", alpha=0.6, zorder=2)
    handles.append(plt.Line2D([0], [0], color='b', label='Geodesics'))
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

def plot_hyperplane(hyperplane_basis: npt.ArrayLike, hyperplane_dir: npt.ArrayLike, manifold: Type[Manifold],
                    ax: plt.Axes, handles: List[Any]) -> None:
    """Plot 1-dimensional hyperplane(s) containing point(s) x."""
    spacing = 10_000
    device = manifold.c.device
    t = torch.linspace(-500, 500, spacing, device=device).reshape(-1, 1)
    hyperplane_basis = torch.from_numpy(hyperplane_basis).to(device)
    hyperplane_dir = torch.from_numpy(hyperplane_dir).to(device)
    # Compute the geodesics and plot them
    for _base,_dir in zip(hyperplane_basis, hyperplane_dir):
        points = manifold.expmap(t*_dir, _base.repeat(spacing, 1)).cpu().detach()
        ax.plot(points[:, 0], points[:, 1], c="g", alpha=0.6)
    ax.scatter(hyperplane_basis[:, 0], hyperplane_basis[:, 1], c="g", marker="P", s=50, zorder=4)
    # Add legend entry for hyperplanes
    handles.append(plt.Line2D([0], [0], color='g', label='Hyperplanes'))
    handles.append(plt.Line2D([0], [0], color='g', marker='P', linestyle='', markersize=10, label='Base Points'))
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

def save_figure(fig: plt.Figure, file_name: str, file_path: Union[str, None]=None,  format: str="png") -> None:
    """Save the figure to a file."""
    if file_path is None:
        path = os.getcwd()
    path = os.path.join(path, "images")
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, f"{file_name}.{format}"))




# class HyperbolicPCA():
#     def __init__(
#         self,
#         n_components=None,
#         *,
#         copy=True,
#         whiten=False,
#         svd_solver="auto",
#         tol=0.0,
#         iterated_power="auto",
#         n_oversamples=10,
#         power_iteration_normalizer="auto",
#         random_state=None,
#     ):
#         self.c = c

#     def fit(self, X):
#         """Fit the model with X.

#         Parameters
#         ----------
#         X : {array-like, sparse matrix} of shape (n_samples, n_features)
#             Training data, where `n_samples` is the number of samples
#             and `n_features` is the number of features.

#         y : Ignored
#             Ignored.

#         Returns
#         -------
#         self : object
#             Returns the instance itself.
#         """
#         self._fit(X)
#         return self