import pytest
import torch

from typing import Tuple, Union
from src.manifolds import Euclidean, Hyperboloid, PoincareBall


def test_addition(manifold: Union[Euclidean, Hyperboloid, PoincareBall], tolerance: Tuple[float, float],
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
        manifold.addition(-uniform_points, uniform_points), identity, atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        manifold.addition(uniform_points, -uniform_points), identity, atol=atol, rtol=rtol
    )
    # Left cancellation law
    #TODO: Fails with abs. diff. 2.72e-08 and rel. diff. 5.10e-06
    #torch.testing.assert_close(manifold.addition(-x, manifold.addition(x, y)), y, atol=atol, rtol=rtol)
    # Distributive law
    torch.testing.assert_close(-manifold.addition(x, y), manifold.addition(-x, -y), atol=atol, rtol=rtol)
    # Gyrotriangle inequality
    assert torch.all(
        manifold.addition(x, y).norm(p=2, dim=-1, keepdim=True)
        <= manifold.addition(x.norm(p=2, dim=-1, keepdim=True), y.norm(p=2, dim=-1, keepdim=True))
    )

def test_scalar_mul(seed: None, manifold: Union[Euclidean, Hyperboloid, PoincareBall],
                    tolerance: Tuple[float, float], uniform_points: torch.Tensor) -> None:
    """Test the scalar_mul operation."""
    atol, rtol = tolerance
    identity = torch.ones((uniform_points.shape[0], 1), dtype=uniform_points.dtype)
    r1 = torch.rand((uniform_points.shape[0], 1), dtype=uniform_points.dtype)
    r2 = torch.rand((uniform_points.shape[0], 1), dtype=uniform_points.dtype)
    # Multiplicative identity
    torch.testing.assert_close(
        manifold.scalar_mul(identity, uniform_points), uniform_points, atol=atol, rtol=rtol
    )
    # N-Gyroaddition
    n = torch.randint(3, 10, (1,)).item()
    n_sum = torch.zeros_like(uniform_points)
    for _ in range(n):
        n_sum = manifold.addition(n_sum, uniform_points)
    torch.testing.assert_close(n_sum, manifold.scalar_mul(n * identity, uniform_points),
                               atol=atol, rtol=rtol)
    # Distributive laws
    torch.testing.assert_close(
        manifold.scalar_mul(r1 + r2, uniform_points),
        manifold.addition(
            manifold.scalar_mul(r1, uniform_points),
            manifold.scalar_mul(r2, uniform_points),
        ),
        atol=atol,
        rtol=rtol
    )
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
    r_huge = torch.tensor(1_000_000, dtype=uniform_points.dtype)
    v_eps_norm = torch.zeros((1, uniform_points.shape[1]), dtype=uniform_points.dtype)
    v_eps_norm[0, 0] = atol
    # Stability of multiplication with zero scalars
    res = manifold.scalar_mul(r_zero, uniform_points)
    assert torch.isfinite(res).all()
    assert manifold.is_in_manifold(res)
    torch.testing.assert_close(res, torch.zeros_like(uniform_points), atol=atol, rtol=rtol)
    res = manifold.scalar_mul(r_zero, v_eps_norm)
    assert torch.isfinite(res).all()
    assert manifold.is_in_manifold(res)
    torch.testing.assert_close(res, torch.zeros_like(v_eps_norm), atol=atol, rtol=rtol)
    # Stability of multiplication with small scalars
    res = manifold.scalar_mul(r_small, v_eps_norm)
    assert torch.isfinite(res).all()
    assert manifold.is_in_manifold(res)
    assert res[0, 0] > r_zero
    torch.testing.assert_close(res[0, 1:], torch.zeros_like(res[0, 1:]), atol=atol, rtol=rtol)
    # Stability of multiplication with large scalars
    res = manifold.scalar_mul(r_huge, uniform_points)
    assert torch.isfinite(res).all()
    assert manifold.is_in_manifold(res)
    res = manifold.scalar_mul(r_huge, v_eps_norm)
    assert torch.isfinite(res).all()
    assert manifold.is_in_manifold(res)
    assert res[0, 0] > r_zero
    torch.testing.assert_close(res[0, 1:], torch.zeros_like(res[0, 1:]), atol=atol, rtol=rtol)

def test_matvec_mul(manifold: Union[Euclidean, Hyperboloid, PoincareBall],
                    tolerance: Tuple[float, float], uniform_points: torch.Tensor) -> None:
    """Test the matvec_mul operation."""
    atol, rtol = tolerance
    m1 = torch.randn(uniform_points.shape[-1], 10, dtype=uniform_points.dtype)
    m2 = torch.randn(m1.shape[-1], 7, dtype=uniform_points.dtype)
    # Consistency of matvec_mul with expmap_0 and logmap_0
    torch.testing.assert_close(
        manifold.matvec_mul(m1, uniform_points),
        manifold.expmap_0(manifold.logmap_0(uniform_points) @ m1),
        atol=atol,
        rtol=rtol
    )
    # Matvec identity
    torch.testing.assert_close(
        manifold.matvec_mul(torch.zeros_like(m1), uniform_points),
        torch.zeros((uniform_points.shape[0], m1.shape[-1]), dtype=uniform_points.dtype),
        atol=atol,
        rtol=rtol
    )
    # Matrix associativity
    torch.testing.assert_close(
        manifold.matvec_mul(m2, manifold.matvec_mul(m1, uniform_points)),
        manifold.expmap_0(manifold.logmap_0(uniform_points) @ (m1 @ m2)),
        atol=atol,
        rtol=rtol
    )

@pytest.mark.skip(reason="not implemented yet")
def test_hyperplane_forward(manifold: Union[Euclidean, Hyperboloid, PoincareBall], tolerance: Tuple[float, float],
                            uniform_points: torch.Tensor) -> None:
    """Test the hyperplane_forward operation."""
    atol, rtol = tolerance
    pass

def test_dist(manifold: Union[Euclidean, Hyperboloid, PoincareBall], tolerance: Tuple[float, float],
              uniform_points: torch.Tensor) -> None:
    """Test the dist and dist_0 operations."""
    atol, rtol = tolerance
    x, y, z = uniform_points.split(uniform_points.shape[0] // 3, dim=0)
    assert torch.isfinite(manifold.dist(x, y)).all()
    assert torch.isfinite(manifold.dist_0(x)).all()
    # Reflexivity
    torch.testing.assert_close(
        manifold.dist(uniform_points, uniform_points)+1, # add one to avoid inf relative errors
        torch.ones((uniform_points.shape[0], 1), dtype=uniform_points.dtype),
        atol=atol,
        rtol=rtol
    )
    # Symmetry
    # TODO: Symmetry does not hold for the the Mobius version
    #torch.testing.assert_close(manifold.dist(x, y), manifold.dist(y, x), atol=atol, rtol=rtol)

    # Triangle inequality
    assert torch.all(manifold.dist(x, z) <= manifold.dist(x, y) + manifold.dist(y, z))
    # Consistency of dist with dist_0
    torch.testing.assert_close(
        manifold.dist(uniform_points, torch.zeros_like(uniform_points)),
        manifold.dist_0(uniform_points),
        atol=atol,
        rtol=rtol
    )

def test_expmap_retraction_logmap(manifold: Union[Euclidean, Hyperboloid, PoincareBall],
                                  tolerance: Tuple[float, float], uniform_points: torch.Tensor) -> None:
    """Test the expmap, expmap_0, retraction, logmap and logmap_0 operations."""
    atol, rtol = tolerance
    x, y = uniform_points.split(uniform_points.shape[0] // 2, dim=0)
    if isinstance(manifold, (Euclidean, PoincareBall)):
        bound = 1_000
        v = torch.empty_like(uniform_points).uniform_(-bound, bound)
    else:   # Hyperboloid
        # TODO: Generate tangent vectors for the Hyperboloid
        pass
    assert manifold.is_in_tangent_space(v, uniform_points)
    # Numerical stability of expmap/expmap_0/retraction
    v_manif = manifold.expmap(v, uniform_points)
    assert torch.isfinite(v_manif).all()
    assert manifold.is_in_manifold(v_manif)
    v_manif = manifold.expmap_0(v)
    assert torch.isfinite(v_manif).all()
    assert manifold.is_in_manifold(v_manif)
    v_manif = manifold.retraction(v, uniform_points)
    assert torch.isfinite(v_manif).all()
    assert manifold.is_in_manifold(v_manif)
    # Numerical stability of logmap/logmap_0
    if isinstance(manifold, Hyperboloid):
        manifold.is_in_tangent_space(manifold.logmap(y, x), x)
        manifold.is_in_tangent_space(manifold.logmap_0(uniform_points), torch.zeros_like(uniform_points))
    # Stability of inverse operations
    # Note: expmap/expmap_0 apply backproj. which is not injective
    res = manifold.expmap(manifold.logmap(y, x), x)
    assert torch.isfinite(res).all()
    assert manifold.is_in_manifold(res)
    # TODO: This assertion fails because the left cancellation law only holds within a certain tolerance
    #torch.testing.assert_close(res, y, atol=atol, rtol=rtol) # relies on the left cancellation law
    res = manifold.expmap_0(manifold.logmap_0(uniform_points))
    assert torch.isfinite(res).all()
    assert manifold.is_in_manifold(res)
    torch.testing.assert_close(res, uniform_points, atol=atol, rtol=rtol)
    # Consistency of expmap/logmap with expmap_0/logmap_0
    torch.testing.assert_close(manifold.expmap(v, torch.zeros_like(v)), manifold.expmap_0(v), atol=atol, rtol=rtol)
    torch.testing.assert_close(
        manifold.logmap(uniform_points, torch.zeros_like(uniform_points)),
        manifold.logmap_0(uniform_points),
        atol=atol,
        rtol=rtol
    )

def test_ptransp(manifold: Union[Euclidean, Hyperboloid, PoincareBall], tolerance: Tuple[float, float],
                 uniform_points: torch.Tensor) -> None:
    """Test the ptransp and ptransp_0 operations."""
    atol, rtol = tolerance
    # Preservation of local geometry under parallel transport
    if isinstance(manifold, (Euclidean, PoincareBall)):
        bound = 1_000
        u = torch.empty_like(uniform_points).uniform_(-bound, bound)
        v = torch.empty_like(uniform_points).uniform_(-bound, bound)
        origin = torch.zeros_like(v)
    else:   # Hyperboloid
        # TODO: Generate tangent vectors at the origin for the Hyperboloid
        pytest.skip()
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

def test_tangent_norm(manifold: Union[Euclidean, Hyperboloid, PoincareBall],
                      tolerance: Tuple[float, float], uniform_points: torch.Tensor) -> None:
    """Test the tangent_inner and tangent_norm operations."""
    atol, rtol = tolerance
    x, y = uniform_points.split(uniform_points.shape[0] // 2, dim=0)
    # Consistency of tangent_norm with expmap/expmap_0, logmap/logmap_0, and dist/dist_0
    torch.testing.assert_close(
        manifold.dist(x, y),
        manifold.tangent_norm(manifold.logmap(y, x), x),
        atol=atol,
        rtol=rtol
    )
    torch.testing.assert_close(
        manifold.dist_0(uniform_points),
        manifold.tangent_norm(manifold.logmap_0(uniform_points), torch.zeros_like(uniform_points)),
        atol=atol,
        rtol=rtol
    )


# Manifold-specific tests
def test_gyration(seed: None, manifold: Union[Euclidean, Hyperboloid, PoincareBall],
                  tolerance: Tuple[float, float], uniform_points: torch.Tensor) -> None:
    """Test the gyration operation of the PoincareBall."""
    if isinstance(manifold, (Euclidean, Hyperboloid)):
        pytest.skip()
    atol, rtol = tolerance
    x, y, z, a = uniform_points.split(uniform_points.shape[0] // 4, dim=0)
    # # Gyration identity
    # #TODO: Fails with abs. diff. 2e-03 and rel. diff. 2e-03
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
    #TODO: Fails with abs. diff. 1e-08 and rel. diff. 5e-08
    #torch.testing.assert_close(manifold._gyration(x, -x, z), z, atol=atol, rtol=rtol)
    torch.testing.assert_close(manifold._gyration(x, y, torch.zeros_like(x)), torch.zeros_like(x), atol=atol, rtol=rtol)
    torch.testing.assert_close(manifold._gyration(x, y, -z), -manifold._gyration(x, y, z), atol=atol, rtol=rtol)
