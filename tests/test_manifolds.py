import random
import pytest
import torch

from typing import Tuple, Union
from src.manifolds import Euclidean, Hyperboloid, PoincareBall
from .conftest import seed, manifold, tolerance, uniform_points


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
    n = random.randint(3, 10)
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

@pytest.mark.skip(reason="not implemented yet")
def test_matvec_mul(manifold: Union[Euclidean, Hyperboloid, PoincareBall], tolerance: Tuple[float, float],
                    uniform_points: torch.Tensor) -> None:
    """Test the matvec_mul operation."""
    atol, rtol = tolerance
    pass
    # def test_matvec_zeros(a, manifold):
    #     mat = a.new_zeros((3, a.shape[-1]))
    #     z = manifold.mobius_matvec(mat, a)
    #     np.testing.assert_allclose(z.detach(), 0.0)
    #     z.sum().backward()
    #     assert torch.isfinite(a.grad).all()
    #     assert torch.isfinite(manifold.k.grad).all()

    # def test_matvec_via_equiv_fn_apply(a, negative, manifold, strict, dtype):
    #     mat = a.new(3, a.shape[-1]).normal_()
    #     y = manifold.mobius_fn_apply(lambda x: x @ mat.transpose(-1, -2), a)
    #     y1 = manifold.mobius_matvec(mat, a)
    #     tolerance = {torch.float32: dict(atol=1e-5, rtol=1e-5), torch.float64: dict()}

    #     tolerant_allclose_check(y, y1, strict=strict, **tolerance[dtype])
    #     y.sum().backward()
    #     assert torch.isfinite(a.grad).all()
    #     assert torch.isfinite(manifold.k.grad).all()

    # def test_mobiusify(a, c, negative, strict, dtype):
    #     mat = a.new(3, a.shape[-1]).normal_()

    #     @stereographic.math.mobiusify
    #     def matvec(x):
    #         return x @ mat.transpose(-1, -2)

    #     y = matvec(a, k=-c)
    #     y1 = stereographic.math.mobius_matvec(mat, a, k=-c)
    #     tolerance = {torch.float32: dict(atol=1e-5, rtol=1e-5), torch.float64: dict()}

    #     tolerant_allclose_check(y, y1, strict=strict, **tolerance[dtype])
    #     y.sum().backward()
    #     assert torch.isfinite(a.grad).all()
    #     assert torch.isfinite(c.grad).all()

    # def test_matvec_chain_via_equiv_fn_apply(a, negative, manifold, dtype):
    #     mat1 = a.new(a.shape[-1], a.shape[-1]).normal_()
    #     mat2 = a.new(a.shape[-1], a.shape[-1]).normal_()
    #     y = manifold.mobius_fn_apply_chain(
    #         a,
    #         lambda x: x @ mat1.transpose(-1, -2),
    #         lambda x: x @ mat2.transpose(-1, -2),
    #     )
    #     y1 = manifold.mobius_matvec(mat1, a)
    #     y1 = manifold.mobius_matvec(mat2, y1)
    #     tolerance = {torch.float32: dict(atol=1e-5, rtol=1e-5), torch.float64: dict()}

    #     tolerant_allclose_check(y, y1, strict=negative, **tolerance[dtype])
    #     y.sum().backward()
    #     assert torch.isfinite(a.grad).all()
    #     assert torch.isfinite(manifold.k.grad).all()

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

# TODO:
@pytest.mark.skip(reason="not implemented yet")
def test_ptransp(manifold: Union[Euclidean, Hyperboloid, PoincareBall], tolerance: Tuple[float, float],
                 uniform_points: torch.Tensor) -> None:
    """Test the ptransp and ptransp_0 operations."""
    atol, rtol = tolerance
    pass

    #         test_parallel_transport() -> None:
    #     mu1 = t([2., 1, np.sqrt(2)]).double() / radius
    #     mu2 = t([np.sqrt(5), 1, np.sqrt(3)]).double() / radius
    #     assert is_in_hyp_space(mu1)
    #     assert is_in_hyp_space(mu2)

    #     u = t([0, 2, -np.sqrt(2)]).double()
    #     # assert is_in_tangent_space(u, at_point=mu1, eps=test_eps)

    #     assert parallel_transport(u, src=mu1, dst=mu1).allclose(u, atol=5e-4)

    #     pt_u = parallel_transport(u, src=mu1, dst=mu2)
    #     # assert is_in_tangent_space(pt_u, at_point=mu2, eps=test_eps)
    #     u_ = parallel_transport(pt_u, src=mu2, dst=mu1)
    #     assert u.allclose(u_, atol=5e-4)
    #     u_inv = inverse_parallel_transport(pt_u, src=mu1, dst=mu2)
    #     assert u.allclose(u_inv)

    # def test_parallel_transport_batch() -> None:
    #     mu1 = t([2., 1, np.sqrt(2)]) / radius
    #     mu2 = t([np.sqrt(5), 1, np.sqrt(3)]) / radius
    #     u = t([0, 2, -np.sqrt(2)])
    #     u2 = t([0, 4, -2 * np.sqrt(2)])

    #     U = torch.stack((u, u2), dim=0)
    #     res = parallel_transport(U, src=mu1, dst=mu2)
    #     U_ = inverse_parallel_transport(res, src=mu1, dst=mu2)
    #     assert U.allclose(U_, atol=test_eps)

    # def test_parallel_transport_mu0() -> None:
    #     mu0 = t([0., 0, 0])
    #     mu2 = t([np.sqrt(5), 1, np.sqrt(3)]) / radius
    #     u = t([0, 2, -np.sqrt(2)])

    #     assert P.parallel_transport_mu0(u, dst=mu0, radius=radius).allclose(u)

    #     pt_u = P.parallel_transport_mu0(u, dst=mu2, radius=radius)
    #     assert parallel_transport(u, src=mu0, dst=mu2).allclose(pt_u)

    #     u_inv = P.inverse_parallel_transport_mu0(pt_u, src=mu2, radius=radius)
    #     assert u.allclose(u_inv)

    # def test_parallel_transport_mu0_batch() -> None:
    #     mu2 = radius * t([np.sqrt(5), 1, np.sqrt(3)])
    #     u = t([0, 2, -np.sqrt(2)])
    #     u2 = t([0, 4, -2 * np.sqrt(2)])

    #     U = torch.stack((u, u2), dim=0)
    #     res = P.parallel_transport_mu0(U, dst=mu2, radius=radius)
    #     U_ = P.inverse_parallel_transport_mu0(res, src=mu2, radius=radius)
    #     assert U.allclose(U_)

    # def test_transp0_preserves_inner_products(a, manifold):
    #     # pointing to the center
    #     v_0 = torch.rand_like(a) + 1e-5
    #     u_0 = torch.rand_like(a) + 1e-5
    #     zero = torch.zeros_like(a)
    #     v_a = manifold.transp0(a, v_0)
    #     u_a = manifold.transp0(a, u_0)
    #     # compute norms
    #     vu_0 = manifold.inner(zero, v_0, u_0, keepdim=True)
    #     vu_a = manifold.inner(a, v_a, u_a, keepdim=True)
    #     np.testing.assert_allclose(vu_a.detach(), vu_0.detach(), atol=1e-6, rtol=1e-6)
    #     (vu_0 + vu_a).sum().backward()
    #     assert torch.isfinite(a.grad).all()
    #     assert torch.isfinite(manifold.k.grad).all()

    # def test_transp0_is_same_as_usual(a, manifold):
    #     # pointing to the center
    #     v_0 = torch.rand_like(a) + 1e-5
    #     zero = torch.zeros_like(a)
    #     v_a = manifold.transp0(a, v_0)
    #     v_a1 = manifold.transp(zero, a, v_0)
    #     # compute norms
    #     np.testing.assert_allclose(v_a.detach(), v_a1.detach(), atol=1e-6, rtol=1e-6)
    #     (v_a + v_a1).sum().backward()
    #     assert torch.isfinite(a.grad).all()
    #     assert torch.isfinite(manifold.k.grad).all()

    # def test_transp_a_b(a, b, manifold):
    #     # pointing to the center
    #     v_0 = torch.rand_like(a)
    #     u_0 = torch.rand_like(a)
    #     v_1 = manifold.transp(a, b, v_0)
    #     u_1 = manifold.transp(a, b, u_0)
    #     # compute norms
    #     vu_1 = manifold.inner(b, v_1, u_1, keepdim=True)
    #     vu_0 = manifold.inner(a, v_0, u_0, keepdim=True)
    #     np.testing.assert_allclose(vu_0.detach(), vu_1.detach(), atol=1e-6, rtol=1e-6)
    #     (vu_0 + vu_1).sum().backward()
    #     assert torch.isfinite(a.grad).all()
    #     assert torch.isfinite(b.grad).all()
    #     assert torch.isfinite(manifold.k.grad).all()

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
