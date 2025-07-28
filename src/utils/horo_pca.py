import torch
import torch.nn as nn

from icecream import ic##
from typing import Tuple
from .math_utils import cosh, sinh
from .helpers import compute_pairwise_distances
from ..manifolds import Manifold, PoincareBall, Hyperboloid


class HoroPCA(nn.Module):
    """
    Horospherical projections dimensionality reduction class.
    Reimplmentation of the HoroPCA method with adjustments for stability, speed, and curvature.

    References
    ----------
    Ines Chami, et al. "Horopca: Hyperbolic dimensionality reduction via horospherical projections."
        International Conference on Machine Learning (2021).
    """
    def __init__(
        self,
        n_components: int,
        n_in_features: int,
        manifold: Manifold,
        lr: float = 1e-3,
        max_steps: int = 100, # TODO: 500 params -- test these params - ic var_loss decrease
    ):
        super().__init__()
        self.n_components = n_components
        self.n_in_features = n_in_features
        self.manifold = manifold
        self.lr = lr
        self.max_steps = max_steps
        # Initialize the manifolds for horo projection and the principal components (ideal points)
        if isinstance(self.manifold, PoincareBall):
            self.hyperboloid = Hyperboloid(c=self.manifold.c, dtype=self.manifold.dtype)
            self.Q = nn.Parameter(torch.randn(self.n_components, self.n_in_features))
        elif isinstance(self.manifold, Hyperboloid):
            self.hyperboloid = self.manifold
            self.Q = nn.Parameter(torch.randn(self.n_components, self.n_in_features-1))
        else:
            raise ValueError("Unsupported manifold type. Use PoincareBall or Hyperboloid.")

    def _to_hyperboloid_ideals(self, ideals: torch.Tensor) -> torch.Tensor:
        """
        Convert the orthonormalized PoincareBall ideal point(s) to Hyperboloid ideal point(s).
        Ideal points in the Hyperboloid are represented by the directions of the corresponding
        1-dimensional null cones. Hence the PoincareBall ideal points need not lie in the closure
        of the PoincareBall. They only need to be orthonormalized to conform with the mapping below.

        Parameters
        ----------
        x : torch.Tensor
            Orthonormalized PoincareBall ideal point(s)

        Returns
        -------
        res : torch.Tensor
            The Hyperboloid ideal point(s)
        """
        res = torch.cat([torch.ones_like(ideals[:,:1]), ideals], dim=-1)
        return res

    def _to_poincare_ideals(self, ideals: torch.Tensor) -> torch.Tensor:
        """
        Convert the Hyperboloid ideal point(s) to PoincareBall ideal point(s).

        Parameters
        ----------
        x : torch.Tensor
            Hyperboloid ideal point(s)

        Returns
        -------
        res : torch.Tensor
            The PoincareBall ideal point(s)
        """
        res = ideals[:,1:] / (ideals[:,:1] * self.manifold.c.sqrt())
        return res

    def _horo_projection(self, x: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """
        Compute the horospherical projection(s) based on horosphere intersections in the Hyperboloid.
        The target submanifold has dimension self.n_components and is a geodesic submanifold passing through
        the Hyperboloid's ideal points and its origin (1/sqrt(c),0,0,0,...). The geodesic submanifold spanned
        by the ideals must not contain the Hyperboloid's origin and the ideals have to be linearly independent.

        Parameters
        ----------
        x : torch.Tensor
            Hyperboloid point(s)
        Q : torch.Tensor
            Ideal point(s) in the Hyperboloid with orthonormalized space coordinates

        Returns
        -------
        res : torch.Tensor
            The horospherical projection(s) of x
        """
        # Compute the orthogonal geodesic projection [x B Q^T (Q B Q^T)^-1 Q] of x onto the geodesic
        # submanifold ("spine") spanned by the ideals ("open book" interpretation), where B is the
        # Minkowski inner product matrix [batched version of Prop. A.23.2 in the HoroPCA paper]
        # 1) Compute the coefficients [x_coeffs = x B Q^T (Q B Q^T)^-1] of the projection.
        #    Since the space coordinates of Q are normalized we can solve the linear system directly
        #    using the Sherman–Morrison formula to compute (Q B Q^T)^-1. The matrix to be inverted here
        #    is just the identity matrix plus the outer product between [-1,...,-1] and [1,...,1].T.
        xBQt = self.hyperboloid._minkowski_inner(x.unsqueeze(-1), Q.T.unsqueeze(0), axis=1).squeeze(1)
        QBQt_inverse = (torch.eye(self.n_components, device=x.device, dtype=x.dtype) + 1/(1-self.n_components))
        x_coeffs = xBQt @ QBQt_inverse
        # 2) Compute the orthogonal geodesic projection onto the spine
        mink_proj = x_coeffs @ Q
        mink_proj_norm = (-self.hyperboloid.c * self.hyperboloid._minkowski_inner(mink_proj, mink_proj)).sqrt()
        spine_proj = mink_proj / mink_proj_norm
        # Compute the tangent vectors of the hyperboloid with base point spine_proj that are pointing
        # towards hyperboloid_origin, are tangent to the target submanifold, and are orthogonal to the spine
        # Note: We orthogonalize the origin to the spine instead of the chords to save compute
        hyperboloid_origin = torch.zeros_like(spine_proj)
        hyperboloid_origin[:,0] = 1 / self.hyperboloid.c.sqrt()
        originBQt = self.hyperboloid._minkowski_inner(hyperboloid_origin.unsqueeze(-1), Q.T.unsqueeze(0), axis=1).squeeze(1)
        origin_coeffs = originBQt @ QBQt_inverse
        tangents = hyperboloid_origin - (origin_coeffs @ Q)
        # Assign the tangent vectors unit speed and map them to the Hyperboloid via the exponential map such
        # that the horospherical projection of x is at distance 'spine_dist' apart from the original point x
        unit_tangents = tangents / self.hyperboloid._minkowski_norm(tangents).sqrt()
        cspine_dist = self.hyperboloid.dist(x, spine_proj) * self.hyperboloid.c.sqrt()
        res = cosh(cspine_dist) * spine_proj + sinh(cspine_dist) * unit_tangents / self.hyperboloid.c.sqrt()
        return res

    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the generalized variance of the projected point(s).

        Parameters
        ----------
        x : torch.Tensor
            Manifold point(s)

        Returns
        -------
        -var : torch.Tensor
            The negative generalized variance of the projected point(s)
        """
        # Orthonormalize the principal components
        Q_ortho, _ = torch.linalg.qr(self.Q.T, mode='reduced')
        # Map the principal components to the null cone
        hyperboloid_ideals = self._to_hyperboloid_ideals(Q_ortho.T)
        # Project x onto the submanifold spanned by the Hyperboloid's principal components
        x_proj = self._horo_projection(x, hyperboloid_ideals)
        # Compute the pairwise distances directly in the Hyperboloid
        distances = compute_pairwise_distances(x_proj, self.hyperboloid)
        # Compute the biased generalized variance of the projected points
        var = torch.mean(distances ** 2)
        return -var

    def fit(self, x: torch.Tensor) -> None:
        """
        Find the principal component(s) using gradient-descent-based optimization.

        Parameters
        ----------
        x : torch.Tensor
            Manifold point(s) of shape (n_samples, n_in_features)
        """
        assert self.manifold.is_in_manifold(x), "Input points must be in the manifold of the model."
        if isinstance(self.manifold, PoincareBall):
            x = self.manifold.to_hyperboloid(x)
        # The parameters of the model are ideal points that lie in the manifold's closure, i.e. they
        # are part of the Euclidean ambient space and do not lie in the hyperbolic space itself
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        # Iteratively compute the projected variance loss and update the parameters
        for _ in range(self.max_steps):
            optim.zero_grad()
            loss = self.compute_loss(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1e05)
            optim.step()

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project the point(s) x onto the submanifold containing the origin that is
        spanned by the generalized principal components of the PoincareBall.

        Parameters
        ----------
        x : torch.Tensor
            Manifold point(s) of shape (n_samples, self.n_in_features)

        Returns
        -------
        res : torch.Tensor (dtype=self.manifold.dtype)
            The projected PoincareBall point(s) of shape (n_samples, self.n_components)
        """
        assert self.manifold.is_in_manifold(x), "Input points must be in the same manifold that was used during fit()."
        if isinstance(self.manifold, PoincareBall):
            x = self.manifold.to_hyperboloid(x)
        # Orthonormalize the principal components
        Q_ortho, _ = torch.linalg.qr(self.Q.T, mode='reduced')
        # Map the principal components to the null cone
        hyperboloid_ideals = self._to_hyperboloid_ideals(Q_ortho.T)
        # Project x onto the submanifold spanned by the Hyperboloid's principal components
        x_proj = self._horo_projection(x, hyperboloid_ideals)
        # Map the projected points back to the PoincareBall
        x_poincare = self.hyperboloid.to_poincare(x_proj)
        # Compute the coordinates in the lower-dimensional PoincareBall #TODO: check if this is correct or needs scaling
        res = x_poincare @ Q_ortho
        assert self.manifold.is_in_manifold(x_poincare), "Projected points must be in the manifold." ##TODO: remove after testing
        return res


def compute_frechet_mean(x: torch.Tensor, manifold: Manifold, lr: float=1e-01,
                         eps: float=1e-05, max_steps: int=5000) -> Tuple[torch.Tensor, bool]:
    """
    Compute the Frechet mean of manifold point(s) x using gradient descent.

    Parameters
    ----------
    x : torch.Tensor
        Manifold point(s)
    manifold : Manifold
        The manifold on which the point(s) lie
    lr : float (optional)
        Learning rate for gradient descent (default: 1e-01)
    eps : float (optional)
        Tolerance for convergence (default: 1e-05)
    max_steps : int (optional)
        Maximum number of gradient descent steps (default: 5000)

    Returns
    -------
    mean, has_converged : Tuple[torch.Tensor, bool]
        Tuple containing the frechet mean and result of the convergence check

    References
    ----------
    P. Thomas Fletcher, et al. "Principal geodesic analysis for the study of nonlinear statistics of shape."
        IEEE transactions on medical imaging 23.8 (2004).
    """
    assert manifold.is_in_manifold(x), "Input points must be in the manifold."
    batch_size = x.shape[0]
    mean_init = torch.mean(x, dim=0, keepdim=True)
    has_converged = False
    # Try multiple learning rates
    for lr in [lr, 2*lr, lr/2, 4*lr, lr/4]:
        mean = mean_init
        for _ in range(max_steps):
            # Compute the logarithmic map of x with respect to the current mean
            logx = torch.sum(manifold.logmap(x, mean), dim=0, keepdim=True)
            delta_mean = lr / batch_size * logx
            # Update the mean using the exponential map
            mean = manifold.expmap(delta_mean, mean)
            if delta_mean.norm(p=2, dim=-1, keepdim=False) < eps:
                has_converged = True
                break
        if has_converged:
            break
    else:
        # If neither learning rate suceeded return the initial mean
        mean = mean_init
    return mean, has_converged

def center_data(x: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    """
    #TODO: only works for curvature==1 as of now; adjust inversion
    Center the data around the Frechet mean.

    Parameters
    ----------
    x : torch.Tensor
        Manifold point(s)
    mean : torch.Tensor
        The Frechet mean of the manifold point(s)

    Returns
    -------
    res : torch.Tensor
        Centered manifold point(s)

    References
    ----------
    P. Thomas Fletcher, et al. "Principal geodesic analysis for the study of nonlinear statistics of shape."
        IEEE transactions on medical imaging 23.8 (2004).
    """
    # Compute the center of the inversion circle that maps the mean to the origin
    center = mean / mean.pow(2).sum(dim=-1, keepdim=True)
    # Apply the isometry that takes the mean to origin on x
    r2 = center.pow(2).sum(dim=-1, keepdim=True) - 1.
    u = x - center
    u2 = u.pow(2).sum(dim=-1, keepdim=True)
    res = r2 / u2 * u + center
    return res

