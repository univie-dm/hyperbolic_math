import torch

from ..manifolds import Manifold, PoincareBall
from ..utils.math_utils import smooth_clamp_min


def compute_pairwise_distances(points: torch.Tensor, manifold: Manifold, batch_size: int=1_000_000, version: str='mobius_direct') -> torch.Tensor:
    """
    Computes the pairwise distances between points on a given manifold.

    Parameters
    ----------
    points : torch.Tensor
        Manifold points
    manifold : Manifold
        The manifold on which the points lie
    batch_size : int (optional)
        The batch size for computing distances in chunks (default: 1_000_000)
    version : str
            Version of the geodesic distance to compute (default: "mobius_direct")
            For PoincareBall:
                ['mobius_direct': Symmetric Mobius distance that doesn't compute self.addition(),
                'mobius':        Mobius distance,
                'metric_tensor': Metric-tensor induced distance,
                'lorentzian_proxy': Lorentzian proxy distance]
            For Hyperboloid:
                    ['normal': Standard Hyperboloid distance,
                    'smoothed': Smoothed Hyperboloid distance]
    Returns
    -------
    distmat : torch.Tensor
        The tensor containing the pairwise distances between points
    """
    device = points.device
    distmat = torch.zeros((points.shape[0], points.shape[0]), dtype=manifold.dtype).to(device)
    indices = torch.triu_indices(points.shape[0], points.shape[0], 1).to(device)
    while indices.shape[1] > 0:
        dist_batch = manifold.dist(points[indices[0,:batch_size]], points[indices[1,:batch_size]], version=version).reshape(-1)
        distmat[indices[0,:batch_size], indices[1,:batch_size]] = dist_batch
        distmat[indices[1,:batch_size], indices[0,:batch_size]] = dist_batch
        indices = indices[:, batch_size:]
    return distmat

def compute_smoothed_pairwise_distances(points: torch.Tensor, manifold: Manifold, batch_size: int=1_000_000) -> torch.Tensor:
    """
    Computes the pairwise distances between points on a given manifold.
    If manifold is a PoincareBall, the distances are computed as usual.
    If manifold is a Hyperboloid, the arguments of the acosh() are smoothly clamped to help with convergence of some algorithms.

    Parameters
    ----------
    points : torch.Tensor
        Manifold points
    manifold : Manifold
        The manifold on which the points lie
    batch_size : int (optional)
        The batch size for computing distances in chunks (default: 1_000_000)

    Returns
    -------
    distmat : torch.Tensor
        The tensor containing the pairwise distances between points
    """
    if isinstance(manifold, PoincareBall):
        distmat = compute_pairwise_distances(points, manifold, batch_size=batch_size)
    else: # Hyperboloid
        device = points.device
        distmat = torch.zeros((points.shape[0], points.shape[0]), dtype=manifold.dtype).to(device)
        indices = torch.triu_indices(points.shape[0], points.shape[0], 1).to(device)
        while indices.shape[1] > 0:
            dist_arg = -manifold.c * manifold._minkowski_inner(points[indices[0,:batch_size]], points[indices[1,:batch_size]])
            smoothed_dist_arg = smooth_clamp_min(dist_arg, 1.0)
            dist_batch = (torch.acosh(smoothed_dist_arg) / manifold.c.sqrt()).reshape(-1)
            distmat[indices[0,:batch_size], indices[1,:batch_size]] = dist_batch
            distmat[indices[1,:batch_size], indices[0,:batch_size]] = dist_batch
            indices = indices[:, batch_size:]
    return distmat

def get_delta(points: torch.Tensor, manifold: Manifold, sample_size=1500, version="average"):
    """
    Computes the delta hyperbolicity value for a set of points on a given manifold.

    Parameters
    ----------
    points : torch.Tensor
        Manifold points
    manifold : Manifold
        The manifold on which the points lie
    sample_size : int (optional)
        The number of points to sample for computing delta (default: 1_500)
    version : str, optional
        The version of delta to compute (default: "average")
        ['average': average delta, 'smallest': smallest delta]

    Returns
    -------
    delta, diam, rel_delta : tuple
        Tuple containing the delta value, the diameter of the distance matrix, and the relative delta
    """
    # Subsample points and compute pairwise distances
    sub_points = points[torch.randperm(points.shape[0])[:sample_size], :]
    distmat = compute_pairwise_distances(sub_points, manifold)
    # Compute the smallest/average delta satisfying the Gromov 4-point condition
    delta = compute_hyperbolic_delta(distmat, version)
    # Compute the relative delta by scaling delta with the diameter
    diam = distmat.max()
    rel_delta = delta / diam
    # TODO: Scale with best possible delta
    #eps = torch.finfo(points.dtype).eps
    #best_possible_delta = (8*(1-eps)**2)/((1-(1-eps)**2)**2)
    #best_possible_delta = acosh(best_possible_delta+1)
    #best_possible_delta = 2*torch.log(1+2**0.5)/best_possible_delta
    #relative_delta -= best_possible_delta
    return delta, diam, rel_delta

def compute_hyperbolic_delta(distmat: torch.Tensor, version: str) -> torch.Tensor:
    """
    Computes the delta hyperbolicity value from a distance matrix.

    Parameters
    ----------
    distmat : torch.Tensor
        Tensor containing the pairwise distances between points
    version : str
        The version of delta to compute

    Returns
    -------
    res : torch.Tensor
        The delta hyperbolicity value
    """
    # Set the first point as reference point and compute the pair-wise Gromov product
    distmat_i0 = distmat[:,0].expand(distmat.shape[0], distmat.shape[0]).T
    distmat_0j = distmat[:,0].expand(distmat.shape[0], distmat.shape[0])
    gromov_prod_mat = (distmat_i0 + distmat_0j - distmat) / 2
    # Compute the (max,min)-product of the Gromov product matrix with itself
    max_min_prod = torch.min(gromov_prod_mat.unsqueeze(1), gromov_prod_mat.unsqueeze(0)).max(dim=2).values
    # Compute the average/smallest delta satisfying the Gromov 4-point condition 
    if version == "average":
        delta = (max_min_prod - gromov_prod_mat).mean()
    else:   # smallest delta
        delta = (max_min_prod - gromov_prod_mat).max()
    # Rescale delta since a reference point was fixed
    res = 2 * delta
    return res
