import pytest
import torch

from typing import Tuple
from src.manifolds import Manifold, Euclidean, Hyperboloid, PoincareBall


def _test_addition(manifold: Manifold, tolerance: Tuple[float, float],
                  uniform_points: torch.Tensor) -> None:
    """Test addition operation."""
    atol, rtol = tolerance
    identity = torch.zeros_like(uniform_points)
    x, y = uniform_points.split(uniform_points.shape[0] // 2, dim=0)
    # Additive identity
    torch.testing.assert_close(
        manifold.addition(identity, uniform_points), uniform_points, atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        manifold.addition(uniform_points, identity), uniform_points, atol=atol, rtol=rtol
    )
    # Additive inverse
    torch.testing.assert_close(
        manifold.addition(-uniform_points, uniform_points)+1, identity+1, atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        manifold.addition(uniform_points, -uniform_points)+1, identity+1, atol=atol, rtol=rtol
    )
    # Left cancellation law - Poincare fails in higher dimensions with abs./rel. diff up to 1e-04
    #torch.testing.assert_close(manifold.addition(-x, manifold.addition(x, y)), y, atol=atol, rtol=rtol)
    # Distributive law
    torch.testing.assert_close(-manifold.addition(x, y), manifold.addition(-x, -y), atol=atol, rtol=rtol)
    # Gyrotriangle inequality
    assert torch.all(
        manifold.addition(x, y).norm(p=2, dim=-1, keepdim=True)
        <= manifold.addition(x.norm(p=2, dim=-1, keepdim=True), y.norm(p=2, dim=-1, keepdim=True)) + atol
    )

def test_scalar_mul(seed: None, manifold: Manifold, tolerance: Tuple[float, float],
                    uniform_points: torch.Tensor) -> None:
    """Test the scalar_mul operation."""
    atol, rtol = tolerance
    identity = torch.ones((uniform_points.shape[0], 1), dtype=uniform_points.dtype)
    r1 = torch.rand((uniform_points.shape[0], 1), dtype=uniform_points.dtype)
    r2 = torch.rand((uniform_points.shape[0], 1), dtype=uniform_points.dtype)
    origin = torch.zeros_like(uniform_points)
    if isinstance(manifold, Hyperboloid):
        origin[:,0] = 1 / manifold.c.sqrt()
    # Multiplicative identity
    torch.testing.assert_close(
        manifold.scalar_mul(identity, uniform_points), uniform_points, atol=atol, rtol=rtol
    )

    ############### Addition tests ###############
    # # N-Gyroaddition
    # n = torch.randint(3, 10, (1,)).item()
    # n_sum = torch.zeros_like(uniform_points)
    # for _ in range(n):
    #     n_sum = manifold.addition(n_sum, uniform_points)
    # torch.testing.assert_close(n_sum, manifold.scalar_mul(n * identity, uniform_points),
    #                            atol=atol, rtol=rtol)
    # # Distributive laws
    # torch.testing.assert_close(
    #     manifold.scalar_mul(r1 + r2, uniform_points),
    #     manifold.addition(
    #         manifold.scalar_mul(r1, uniform_points),
    #         manifold.scalar_mul(r2, uniform_points),
    #     ),
    #     atol=atol,
    #     rtol=rtol
    # )
    #############################################

    if isinstance(manifold, (Euclidean, PoincareBall)):
        # Hyperboloid: -uniform_points are not on the manifold since they are past-pointing
        torch.testing.assert_close(
            manifold.scalar_mul(-r1, uniform_points),
            manifold.scalar_mul(r1, -uniform_points),
            atol=atol,
            rtol=rtol
        )
    # Associative laws
    torch.testing.assert_close(
        manifold.scalar_mul(r1 * r2, uniform_points),
        manifold.scalar_mul(r1, manifold.scalar_mul(r2, uniform_points)),
        atol=atol,
        rtol=rtol
    )
    torch.testing.assert_close(
        manifold.scalar_mul(r1 * r2, uniform_points),
        manifold.scalar_mul(r2, manifold.scalar_mul(r1, uniform_points)),
        atol=atol,
        rtol=rtol
    )
    if isinstance(manifold, (Euclidean, PoincareBall)):
        # Scaling property
        left_side = manifold.scalar_mul(torch.abs(r1), uniform_points)
        left_side /= manifold.scalar_mul(r1, uniform_points).norm(p=2, dim=-1, keepdim=True)
        torch.testing.assert_close(
            left_side, uniform_points / uniform_points.norm(p=2, dim=-1, keepdim=True),
            atol=atol,
            rtol=rtol
        )
        # Homogenity property
        torch.testing.assert_close(
            manifold.scalar_mul(r1, uniform_points).norm(p=2, dim=-1, keepdim=True),
            manifold.scalar_mul(torch.abs(r1), uniform_points.norm(p=2, dim=-1, keepdim=True)),
            atol=atol,
            rtol=rtol
        )
    # Numerical stability
    r_zero = torch.tensor(0, dtype=uniform_points.dtype)
    r_small = torch.tensor(atol, dtype=uniform_points.dtype)
    r_large = torch.tensor(10, dtype=uniform_points.dtype)
    v_eps_norm = torch.zeros((1, uniform_points.shape[1]), dtype=uniform_points.dtype)
    v_eps_norm[0, 0] = atol
    if isinstance(manifold, Hyperboloid):
        v_eps_norm[0, 0] = v_eps_norm[0, 0] + 1 / manifold.c.sqrt()
        v_eps_norm = manifold.proj(v_eps_norm)
    # Stability of multiplication with zero scalars
    res = manifold.scalar_mul(r_zero, uniform_points)
    assert torch.isfinite(res).all()
    assert manifold.is_in_manifold(res)
    torch.testing.assert_close(res+1, origin+1, atol=atol, rtol=rtol)
    res = manifold.scalar_mul(r_zero, v_eps_norm)
    assert torch.isfinite(res).all()
    assert manifold.is_in_manifold(res)
    torch.testing.assert_close(res+1, origin[:1]+1, atol=atol, rtol=rtol)
    # Stability of multiplication with small scalars
    res = manifold.scalar_mul(r_small, v_eps_norm)
    assert torch.isfinite(res).all()
    assert manifold.is_in_manifold(res)
    assert res[0, 0] > r_zero
    torch.testing.assert_close(res[0, 1:], torch.zeros_like(res[0, 1:]), atol=atol, rtol=rtol)
    # Stability of multiplication with large scalars
    if isinstance(manifold, (Euclidean, PoincareBall)) or uniform_points.dtype == torch.float64:
        # Hyperboloid: float32 fails b/c of the numerical instabilities introduced by
        # the minkowski inner product within the is_in_manifold check
        res = manifold.scalar_mul(r_large, uniform_points)
        assert torch.isfinite(res).all()
        assert manifold.is_in_manifold(res)
    res = manifold.scalar_mul(r_large, v_eps_norm)
    assert torch.isfinite(res).all()
    assert manifold.is_in_manifold(res)
    assert res[0, 0] > r_zero
    torch.testing.assert_close(res[0, 1:]+1, origin[0, 1:]+1, atol=atol, rtol=rtol)

def test_dist(manifold: Manifold, tolerance: Tuple[float, float],
              uniform_points: torch.Tensor) -> None:
    """Test the dist and dist_0 operations."""
    atol, rtol = tolerance
    x, y, z = uniform_points.split(uniform_points.shape[0] // 3, dim=0)
    assert torch.isfinite(manifold.dist(x, y)).all()
    assert torch.isfinite(manifold.dist_0(x)).all()
    # Reflexivity
    torch.testing.assert_close(
        manifold.dist(uniform_points, uniform_points)+1,
        torch.ones((uniform_points.shape[0], 1), dtype=uniform_points.dtype),
        atol=atol,
        rtol=rtol
    )
    # Symmetry
    torch.testing.assert_close(manifold.dist(x, y), manifold.dist(y, x), atol=atol, rtol=rtol)
    # Triangle inequality
    assert torch.all(manifold.dist(x, z) <= manifold.dist(x, y) + manifold.dist(y, z) + atol)
    # Consistency of dist with dist_0
    origin = torch.zeros_like(uniform_points)
    if isinstance(manifold, Hyperboloid):
        origin[:,0] = 1 / manifold.c.sqrt()
    torch.testing.assert_close(
        manifold.dist(uniform_points, origin),
        manifold.dist_0(uniform_points),
        atol=atol,
        rtol=rtol
    )

def test_expmap_retraction_logmap(manifold: Manifold, tolerance: Tuple[float, float],
                                  uniform_points: torch.Tensor) -> None:
    """Test the expmap, expmap_0, retraction, logmap and logmap_0 operations."""
    atol, rtol = tolerance
    x, y = uniform_points.split(uniform_points.shape[0] // 2, dim=0)
    origin = torch.zeros_like(uniform_points)
    bound = 10
    v = torch.empty_like(uniform_points).uniform_(-bound, bound)
    v0 = v.clone()
    if isinstance(manifold, Hyperboloid):
        origin[:,0] = 1 / manifold.c.sqrt()
        # Project the candidate tangent vectors onto the tangent space at the Hyperboloid origin
        v0 = manifold.tangent_proj(v, origin)
        # Project the candidate tangent vectors onto the tangent space at x
        v = manifold.tangent_proj(v, uniform_points)
    assert manifold.is_in_tangent_space(v, uniform_points)
    assert manifold.is_in_tangent_space(v0, origin)
    # Numerical stability of expmap/expmap_0/retraction
    if isinstance(manifold, PoincareBall):
        # The normal Hyperboloid.exmap and Hyperboloid.exmap_0 do not pass the is_in_manifold check
        # in float32, only the retraction does. On the other hand, the retraction does not pass the
        # expmap_0(logmap_0())-check in float64. The convergence, in practice, is much better with
        # the non-approximated functions, e.g. with horopca we can get ~2x the distortion otherwise.
        v_manif = manifold.expmap(v, uniform_points)
        assert torch.isfinite(v_manif).all()
        assert manifold.is_in_manifold(v_manif) # Normal expmap in float32 violates this check
        v0_manif = manifold.expmap_0(v0)
        assert torch.isfinite(v0_manif).all()
        assert manifold.is_in_manifold(v0_manif) # Normal expmap in float32 violates this check
        v_manif = manifold.retraction(v, uniform_points)
        assert torch.isfinite(v_manif).all()
        assert manifold.is_in_manifold(v_manif)
    v0_manif = manifold.retraction(v0, origin)
    assert torch.isfinite(v0_manif).all()
    assert manifold.is_in_manifold(v0_manif)
    # Numerical stability of logmap/logmap_0
    if isinstance(manifold, Hyperboloid):
        manifold.is_in_tangent_space(manifold.logmap(y, x), x)
        manifold.is_in_tangent_space(manifold.logmap_0(uniform_points), origin)
    # Stability of inverse operations
    # Note: expmap/expmap_0 apply backproj. which is not injective
    res = manifold.expmap(manifold.logmap(y, x), x)
    assert torch.isfinite(res).all()
    assert manifold.is_in_manifold(res)
    res = manifold.expmap_0(manifold.logmap_0(uniform_points))
    assert torch.isfinite(res).all()
    assert manifold.is_in_manifold(res)
    torch.testing.assert_close(res, uniform_points, atol=atol, rtol=rtol) # Retraction in float64 violates this check
    # Consistency of expmap/logmap with expmap_0/logmap_0
    torch.testing.assert_close(manifold.expmap(v0, origin), manifold.expmap_0(v0), atol=atol, rtol=rtol)
    torch.testing.assert_close(
        manifold.logmap(uniform_points, origin),
        manifold.logmap_0(uniform_points),
        atol=atol,
        rtol=rtol
    )

def test_ptransp(manifold: Manifold, tolerance: Tuple[float, float],
                 uniform_points: torch.Tensor) -> None:
    """Test the ptransp and ptransp_0 operations."""
    atol, rtol = tolerance
    origin = torch.zeros_like(uniform_points)
    # Large tangent vectors can lead to numerical instability
    # -> Riemannian metric may not be preserved when parallel transporting
    bound = 200
    u = torch.empty_like(uniform_points).uniform_(-bound, bound)
    v = torch.empty_like(uniform_points).uniform_(-bound, bound)
    if isinstance(manifold, Hyperboloid):
        origin[:,0] = 1 / manifold.c.sqrt()
        # Project the candidate tangent vectors onto the tangent space at the Hyperboloid origin
        u = manifold.tangent_proj(v, origin)
        v = manifold.tangent_proj(v, origin)
    # Preservation of local geometry under parallel transport
    assert manifold.is_in_tangent_space(u, origin)
    assert manifold.is_in_tangent_space(v, origin)
    u_pt = manifold.ptransp_0(u, uniform_points)
    assert manifold.is_in_tangent_space(u_pt, uniform_points)
    v_pt = manifold.ptransp_0(v, uniform_points)
    assert manifold.is_in_tangent_space(v_pt, uniform_points)
    torch.testing.assert_close(
        manifold.tangent_inner(u, v, origin),
        manifold.tangent_inner(u_pt, v_pt, uniform_points),
        atol=atol,
        rtol=rtol
    )
    # Consistency of ptransp with ptransp_0
    torch.testing.assert_close(
        manifold.ptransp(u, origin, uniform_points),
        u_pt,
        atol=atol,
        rtol=rtol
    )
    # Numerical stability
    torch.testing.assert_close(
        manifold.ptransp(u_pt, uniform_points, origin),
        u,
        atol=atol,
        rtol=rtol
    )
    assert manifold.is_in_tangent_space(manifold.ptransp(u_pt, uniform_points, origin), origin)

def test_tangent_norm(manifold: Manifold, tolerance: Tuple[float, float],
                      uniform_points: torch.Tensor) -> None:
    """Test the tangent_inner and tangent_norm operations."""
    atol, rtol = tolerance
    x, y = uniform_points.split(uniform_points.shape[0] // 2, dim=0)
    # Consistency of tangent_norm with logmap/logmap_0 and dist/dist_0
    torch.testing.assert_close(
        manifold.dist(x, y),
        manifold.tangent_norm(manifold.logmap(y, x), x),
        atol=atol,
        rtol=rtol
    )
    origin = torch.zeros_like(uniform_points)
    if isinstance(manifold, Hyperboloid):
        origin[:,0] = 1 / manifold.c.sqrt()
    torch.testing.assert_close(
        manifold.dist_0(uniform_points),
        manifold.tangent_norm(manifold.logmap_0(uniform_points), origin),
        atol=atol,
        rtol=rtol
    )


# Manifold-specific tests
def test_gyration(seed: None, manifold: Manifold, tolerance: Tuple[float, float],
                  uniform_points: torch.Tensor) -> None:
    """Test the gyration operation of the PoincareBall."""
    if isinstance(manifold, (Euclidean, Hyperboloid)):
        pytest.skip()
    atol, rtol = tolerance
    x, y, z, a = uniform_points.split(uniform_points.shape[0] // 4, dim=0)
    # Gyration identity - Poincare fails with abs./rel. diff up to 1e-02
    # torch.testing.assert_close(
    #     manifold._gyration(x, y, z),
    #     manifold.addition(-manifold.addition(x, y), manifold.addition(x, manifold.addition(y, z))),
    #     atol=atol,
    #     rtol=rtol
    # )
    # (Gyro-)commutative law
    torch.testing.assert_close(
        manifold.addition(x, y),
        manifold._gyration(x, y, manifold.addition(y, x)),
        atol=atol,
        rtol=rtol
    )
    # Gyrosum inversion law
    torch.testing.assert_close(
        -manifold.addition(x, y),
        manifold._gyration(x, y, manifold.addition(-y, -x)),
        atol=atol,
        rtol=rtol
    )
    # Left (gyro-)associative law
    torch.testing.assert_close(
        manifold.addition(x, manifold.addition(y, z)),
        manifold.addition(manifold.addition(x, y), manifold._gyration(x, y, z)),
        atol=atol,
        rtol=rtol
    )
    # Right (gyro-)associative law
    torch.testing.assert_close(
        manifold.addition(manifold.addition(x, y), z),
        manifold.addition(x, manifold.addition(y, manifold._gyration(y, x, z))),
        atol=atol,
        rtol=rtol
    )
    # Mobius addition under gyrations
    torch.testing.assert_close(
        manifold._gyration(x, y, manifold.addition(z, a)),
        manifold.addition(manifold._gyration(x, y, z), manifold._gyration(x, y, a)),
        atol=atol,
        rtol=rtol
    )
    # Left loop property
    torch.testing.assert_close(
        manifold._gyration(x, y, z),
        manifold._gyration(manifold.addition(x, y), y, z),
        atol=atol,
        rtol=rtol
    )
    # Right loop property
    torch.testing.assert_close(
        manifold._gyration(x, y, z),
        manifold._gyration(x, manifold.addition(y, x), z),
        atol=atol,
        rtol=rtol
    )
    # Identity gyroautomorphism property
    r1 = torch.rand((x.shape[0], 1), dtype=x.dtype)
    r2 = torch.rand((x.shape[0], 1), dtype=x.dtype)
    torch.testing.assert_close(
        manifold._gyration(manifold.scalar_mul(r1, x), manifold.scalar_mul(r2, x), y),
        y,
        atol=atol,
        rtol=rtol
    )
    # Gyroautomorphism property
    torch.testing.assert_close(
        manifold._gyration(x, y, manifold.scalar_mul(r1, z)),
        manifold.scalar_mul(r1, manifold._gyration(x, y, z)),
        atol=atol,
        rtol=rtol
    )
    # First gyrogroup theorems
    torch.testing.assert_close(manifold._gyration(x, torch.zeros_like(x), z), z, atol=atol, rtol=rtol)
    torch.testing.assert_close(manifold._gyration(torch.zeros_like(x), x, z), z, atol=atol, rtol=rtol)
    torch.testing.assert_close(manifold._gyration(x, x, z), z, atol=atol, rtol=rtol)
    torch.testing.assert_close(manifold._gyration(x, y, torch.zeros_like(x)), torch.zeros_like(x), atol=atol, rtol=rtol)
    torch.testing.assert_close(manifold._gyration(x, y, -z), -manifold._gyration(x, y, z), atol=atol, rtol=rtol)
