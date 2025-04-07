import copy
import os
import torch
import numpy as np
import numpy.typing as npt

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Dict, List, Tuple, Union
from .helpers import compute_pairwise_distances
from ..manifolds import Manifold, Hyperboloid


def create_figure(points: torch.Tensor,
                  _manifold: Manifold,
                  labels: Union[npt.ArrayLike, None]=None,
                  edges: Union[Tuple[List[int], List[int]], None]=None,
                  hyperplanes: Union[Tuple[torch.Tensor, torch.Tensor], None]=None,
                  settings: Dict[str, Union[str, bool]]=None
                  ) -> None:
    """
    Create and save a 2D visualization of hyperbolic points, geodesics, and hyperplanes in the PoincareBall.

    Parameters
    ----------
    points : torch.Tensor
        Manifold point(s) to be visualized
    manifold_ : Manifold
        The manifold type (e.g. Hyperboloid, PoincareBall)
    labels : Union[npt.ArrayLike, None] (optional)
        Labels for the manifold point(s) (default: None)
    edges : Tuple[List[int], List[int]] (optional)
        Index pairs for which edges/geodesics should be visualized.
        The Indices refer to the input 'points' (default: None)
    hyperplanes : Tuple[torch.Tensor, torch.Tensor] (optional)
        Tuple containing hyperplane normals and base points (default: None)
    settings : Dict[str, Union[str, bool]] (optional)
        Dictionary of settings for the visualization.
        If not provided, the following default settings are used:
        - "plot_manifold_dtype": torch.float64 (Data type for plotting)
        - "dim_red_method": "tangent PCA" {"tangent PCA", "hyperbolic PCA", "tangent tSNE", "hyperbolic tSNE"} (Dimensionality reduction method)
        - "title": "Hyperbolic Embeddings" (Title of the plot)
        - "show_origin": True (Whether to show the origin)
        - "save_figure": False (Whether to save the figure or return it)
        - "file_name": "hyperbolic_embeddings" (Name of the saved file)
        - "file_path": None (Path to save the file, current directory if None)
        - "file_format": "png" (Format of the saved file)
    """
    default_settings = {
        "plot_manifold_dtype": torch.float64,
        "dim_red_method": "tangent PCA",
        "title": "Hyperbolic Embeddings",
        "show_origin": True,
        "save_figure": False,
        "file_name": "hyperbolic_embeddings",
        "file_path": None,
        "file_format": "png",
    }

    if settings is not None:
        default_settings.update(settings)
    settings = default_settings

    # Create a copy of the manifold with curvature of the same type as 'plot_manifold_dtype'
    # Default "plot_manifold_dtype" is double precision to avoid representational instabilities
    manifold = copy.deepcopy(_manifold)
    manifold.dtype = settings['plot_manifold_dtype']
    manifold.c = manifold.c.to(manifold.dtype)

    assert manifold.is_in_manifold(points), "Points are not in the manifold"
    points = points.detach()
    poincare_closure = 1 / manifold.c.sqrt().cpu().detach()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_aspect('equal')
    ax.set_xlim(-1.1 * poincare_closure, 1.1 * poincare_closure)
    ax.set_ylim(-1.1 * poincare_closure, 1.1 * poincare_closure)

    # Draw the closure of the PoincareBall
    circle = plt.Circle((0, 0), poincare_closure, fill=False, color='black', zorder=5)
    ax.add_artist(circle)

    if settings['show_origin']:
        ax.scatter(0, 0, marker='x', c='black', s=50, zorder=5)

    if isinstance(manifold, Hyperboloid):
        # TODO: Project manifold points onto PoincareBall first
        raise NotImplementedError('Hyperboloid is not supported yet')

    # Plot points in 2D
    if points.shape[-1] > 2:
        points, hyperplanes = pointsTo2d(points, manifold, hyperplanes, settings)
        ax.set_title(f"{settings['title']} ({settings['dim_red_method']})")
        # TODO: For hyperbolic methods some results are no longer on the manifold
        #assert manifold.is_in_manifold(torch.from_numpy(points)), "Points are not in the manifold"
    else:
        points = points.numpy()
        hyperplanes = (hyperplanes[0].numpy(), hyperplanes[1].numpy()) if hyperplanes is not None else None
        ax.set_title(f"{settings['title']}")
    handles = plot_2d_points(points, ax, labels)

    # Plot geodesics between specified points
    if edges is not None:
        if settings['dim_red_method'] in ['tangent tSNE', 'hyperbolic tSNE']:
            # tSNE for edges skews the resulting projection way too much
            # since we sample many points along geodesics
            raise NotImplementedError('tSNE for edges oversamples geodesics')
        plot_edges(points, edges, manifold, ax, handles)

    # Plot hyperplanes
    if hyperplanes is not None:
        plot_hyperplane(hyperplanes, manifold, ax, handles)

    if handles:
        ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    if settings['save_figure']:
        save_figure(fig, settings['file_name'], file_path=settings['file_path'], format=settings['file_format'])
    else:
        return fig


def pointsTo2d(x: torch.Tensor, manifold: Manifold,
               hyperplanes: Union[Tuple[torch.Tensor, torch.Tensor], None],
               settings: dict) -> Tuple[npt.ArrayLike, Union[npt.ArrayLike, None]]:
    """Project points and hyperplanes to 2d using the specified method."""
    sample_size = x.shape[0]

    if hyperplanes is not None:
        hyperplane_size = hyperplanes[0].shape[0]
        x = torch.cat((x, hyperplanes[0], hyperplanes[1]), dim=0)

    if settings['dim_red_method'] == 'tangent tSNE':
        # tSNE projected points are all pushed to the boundary
        # TODO: Implement proper tSNE with adjusted hyperbolic distr. (CO-SNE)
        x = manifold.logmap_0(x).cpu()
        x = TSNE(n_components=2, init='random').fit_transform(x)
        x = manifold.expmap_0(torch.from_numpy(x).to(manifold.c.device))
    elif settings['dim_red_method'] == 'hyperbolic tSNE':
        # tSNE like this in the hyperbolic space does not make any sense
        # -> Almost all projected points are no longer on the manifold
        pairwise_dists = compute_pairwise_distances(x, manifold).cpu()
        x = TSNE(n_components=2, metric='precomputed', init='random').fit_transform(pairwise_dists)
    elif settings['dim_red_method'] == 'tangent PCA':
        x = manifold.logmap_0(x).cpu()
        model = PCA(n_components=2).fit(x[:sample_size])
        x = model.transform(x)
        x = manifold.expmap_0(torch.from_numpy(x).to(manifold.c.device))
    elif settings['dim_red_method'] == 'hyperbolic PCA':
        model = PCA(n_components=2).fit(x[:sample_size].cpu())
        x = model.transform(x)
    else:
        raise ValueError(f"Unknown dimensionality reduction method {settings['dim_red_method']}")

    if hyperplanes is not None:
        points = x[:sample_size].cpu().numpy()
        hyperplanes = (x[sample_size:sample_size + hyperplane_size].cpu().numpy(),
                       x[sample_size + hyperplane_size:].cpu().numpy())
    else:
        points = x.cpu().numpy()

    return points, hyperplanes


def plot_2d_points(x: npt.ArrayLike, ax: plt.Axes, labels: Union[npt.ArrayLike, None]=None) -> List[plt.Line2D]:
    """Plot 2d PoincareBall points with labels (optional)."""
    if labels is None:
        ax.scatter(x[:, 0], x[:, 1], c='blue', alpha=0.6, zorder=4)
        return []
    else:
        assert x.shape[0] == len(labels), "Number of labels must match number of points"
        unique_labels = np.unique(labels)
        cmap = plt.cm.tab10 if len(unique_labels) <= 10 else plt.cm.tab20
        colors = cmap(np.linspace(0, 1, len(unique_labels)))
        color_map = dict(zip(unique_labels, colors))

        ax.scatter(x[:, 0], x[:, 1], c=[color_map[label] for label in labels], alpha=0.6, zorder=3)

        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[label],
                              markersize=10, alpha=0.6, label=f"class: {label}")
                   for label in unique_labels]
        return handles


def plot_edges(points: npt.ArrayLike, edges: Tuple[List[int], List[int]],
               manifold: Manifold, ax: plt.Axes, handles: List[plt.Line2D]) -> None:
    """Plot geodesic segment(s) connecting x and y."""
    assert len(edges[0]) == len(edges[0]), "Start and end points must have the same shape"

    spacing = 100
    device = manifold.c.device
    t = torch.linspace(0, 1.0, spacing, device=device).reshape(-1, 1)
    x = torch.from_numpy(points[edges[0]]).to(device)
    y = torch.from_numpy(points[edges[1]]).to(device)
    dir = manifold.addition(-x, y, backproject=True)

    for _x, _dir in zip(x, dir):
        # Compute points on the geodesic segment connecting x and y
        second_term = manifold.scalar_mul(t, _dir.repeat(spacing, 1), backproject=True)
        geodesic = manifold.addition(_x.repeat(spacing, 1), second_term, backproject=True)
        geodesic = geodesic.cpu().detach()
        ax.plot(geodesic[:, 0], geodesic[:, 1], c='blue', alpha=0.6, zorder=2)

    handles.append(plt.Line2D([0], [0], color='blue', label='Geodesic'))


def plot_hyperplane(hyperplanes: Tuple[npt.ArrayLike, npt.ArrayLike], manifold: Manifold,
                    ax: plt.Axes, handles: List[plt.Line2D]) -> None:
    """Plot hyperplane(s) and their base point(s)."""
    hyperplane_normals, hyperplane_base_points = hyperplanes
    hyperplane_normals = hyperplane_normals.reshape(-1, 2)
    hyperplane_base_points = hyperplane_base_points.reshape(-1, 2)

    hyperplane_normals[:, 0] = -hyperplane_normals[:, 0]
    hyperplane_dirs = hyperplane_normals[:, ::-1].copy()

    spacing = 10_000
    device = manifold.c.device
    t = torch.linspace(-500, 500, spacing, device=device)
    hyperplane_base_points = torch.from_numpy(hyperplane_base_points).to(device)
    hyperplane_dirs = torch.from_numpy(hyperplane_dirs).to(device)

    for _base, _dir in zip(hyperplane_base_points, hyperplane_dirs):
        points = manifold.expmap(torch.outer(t, _dir), _base.repeat(spacing, 1)).cpu().detach()
        ax.plot(points[:, 0], points[:, 1], c='green', alpha=0.6, zorder=3)

    ax.scatter(hyperplane_base_points[:, 0], hyperplane_base_points[:, 1], c='green', marker='P', s=50, zorder=4)

    handles.append(plt.Line2D([0], [0], color='green', label='Hyperplane'))
    handles.append(plt.Line2D([0], [0], color='green', marker='P', linestyle='',
                              markersize=10, label='Hyperplane Base Points'))


def save_figure(fig: plt.Figure, file_name: str, file_path: Union[str, None]=None, format: str='png') -> None:
    """Save the figure to a file."""
    if file_path is None:
        file_path = os.getcwd()
    path = os.path.join(file_path, 'images')
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, f"{file_name}.{format}"))
    plt.close(fig)
