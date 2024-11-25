import random
from typing import Tuple

import pytest
import numpy as np
import torch

from src.manifolds.poincare import PoincareBall


# @pytest.fixture(scope="function", autouse=True, params=range(30, 40))
@pytest.fixture(scope="module")
def seed() -> int:
    """Set the seed for all tests."""
    # Make it for one seed (14) only now
    # seed = request.param
    seed = 14
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed


@pytest.fixture(scope="module")
def c(seed: int) -> torch.Tensor:
    """Generate a random curvature value."""
    return torch.DoubleTensor(1, 1).uniform_(torch.finfo(torch.float64).eps, 5)


# @pytest.fixture(scope="function", params=[torch.float64, torch.float32], ids=["float64", "float32"])
@pytest.fixture(scope="module")
def dtype(request) -> torch.dtype:
    """Allow testing for different data types."""
    # return request.param
    return torch.float64


@pytest.fixture(scope="module")
def tolerance(dtype: torch.dtype) -> Tuple[float, float]:
    """Set the tolerance for the tests."""
    return torch.finfo(dtype).eps, torch.finfo(torch.float32).eps


@pytest.fixture(scope="module")
def manifold(seed: int) -> PoincareBall:
    """Instantiate the PoincareBall manifold."""
    manifold = PoincareBall()
    manifold.c = torch.DoubleTensor(1, 1).uniform_(torch.finfo(torch.float64).eps, 5)

    return manifold


@pytest.fixture(scope="module")
def uniform_ball_points(c: torch.Tensor) -> torch.Tensor:
    """Generate random points on the Poincare ball."""
    num_pts = 2500 * 4
    dim = random.randint(1, 10)
    random_dirs = torch.normal(
        mean=torch.zeros((num_pts, dim), dtype=torch.float64), std=torch.ones((num_pts, dim), dtype=torch.float64)
    )
    random_dirs /= random_dirs.norm(dim=-1, p=2, keepdim=True)
    # Generate random radii with probability proportional to the surface area
    random_radii = torch.rand((num_pts, 1), dtype=torch.float64).pow(1 / dim)
    return c**-0.5 * (random_dirs * random_radii)


def test_is_in_manifold(manifold: PoincareBall, uniform_ball_points: torch.Tensor, c: torch.Tensor) -> None:
    """Check that points are"""
    assert manifold.is_in_manifold(uniform_ball_points, c)


def test_mobius_add(
    manifold: PoincareBall, uniform_ball_points: torch.Tensor, c: torch.Tensor, tolerance: Tuple[float, float]
) -> None:
    """Test the Mobius addition operation."""

    atol, rtol = tolerance

    identity = torch.zeros_like(uniform_ball_points)
    x, y = uniform_ball_points.split(uniform_ball_points.shape[0] // 2, dim=0)
    # Additive identity
    torch.testing.assert_close(
        manifold.mobius_add(identity, uniform_ball_points, c), uniform_ball_points, atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        manifold.mobius_add(uniform_ball_points, identity, c), uniform_ball_points, atol=atol, rtol=rtol
    )
    # Additive inverse
    torch.testing.assert_close(
        manifold.mobius_add(-uniform_ball_points, uniform_ball_points, c), identity, atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        manifold.mobius_add(uniform_ball_points, -uniform_ball_points, c), identity, atol=atol, rtol=rtol
    )
    # # Left cancellation law
    # torch.testing.assert_close(manifold.mobius_add(-x, manifold.mobius_add(x, y, c), c),
    #                            y,
    #                            atol=atol, rtol=rtol)

    #     res = manifold.mobius_add(-a, manifold.mobius_add(a, b))
    # tolerance = {torch.float32: dict(atol=5e-5, rtol=5e-4), torch.float64: dict()}
    # np.testing.assert_allclose(res.detach(), b.detach(), **tolerance[dtype])

    # Distributive law
    torch.testing.assert_close(-manifold.mobius_add(x, y, c), manifold.mobius_add(-x, -y, c), atol=atol, rtol=rtol)
    # Additive closedness
    assert manifold.is_in_manifold(manifold.mobius_add(x, y, c), c)
    # Gyrotriangle inequality
    assert torch.all(
        manifold.mobius_add(x, y, c).norm(dim=-1, keepdim=True, p=2)
        <= manifold.mobius_add(x.norm(dim=-1, keepdim=True, p=2), y.norm(dim=-1, keepdim=True, p=2), c)
    )

    # def test_add_infinity_and_beyond(a, b, c, negative, manifold, dtype):


#     _a = a
#     if torch.isclose(c, c.new_zeros(())).any():
#         pytest.skip("zero not checked")
#     infty = b * 10000000
#     for i in range(100):
#         z = manifold.expmap(a, infty, project=False)
#         z = manifold.projx(z)
#         assert not torch.isnan(z).any(), ("Found nans", i, z)
#         assert torch.isfinite(z).all(), ("Found Infs", i, z)
#         z = manifold.mobius_scalar_mul(
#             torch.tensor(1000.0, dtype=z.dtype), z, project=False
#         )
#         z = manifold.projx(z)
#         assert not torch.isnan(z).any(), ("Found nans", i, z)
#         assert torch.isfinite(z).all(), ("Found Infs", i, z)

#         infty = manifold.transp(a, z, infty)
#         assert torch.isfinite(infty).all(), (i, infty)
#         a = z
#     z = manifold.expmap(a, -infty)
#     # they just need to be very far, exact answer is not supposed
#     tolerance = {
#         torch.float32: dict(rtol=3e-1, atol=2e-1),
#         torch.float64: dict(rtol=1e-1, atol=1e-3),
#     }
#     if negative:
#         np.testing.assert_allclose(z.detach(), -a.detach(), **tolerance[dtype])
#     else:
#         assert not torch.isnan(z).any(), "Found nans"
#         assert not torch.isnan(a).any(), "Found nans"


# @pytest.mark.skip(reason="Currently failing")
def test_gyration(
    manifold: PoincareBall, uniform_ball_points: torch.Tensor, c: torch.Tensor, tolerance: Tuple[float, float]
) -> None:
    x, y, z, a = uniform_ball_points.split(uniform_ball_points.shape[0] // 4, dim=0)
    atol, rtol = tolerance
    # # Gyration identity
    # first_term = -manifold.mobius_add(x, y, c)
    # second_term = manifold.mobius_add(x, manifold.mobius_add(y, z, c), c)
    # torch.testing.assert_close(
    #     manifold._gyration(x, y, z, c), manifold.mobius_add(first_term, second_term, c), atol=atol, rtol=rtol
    # )
    # (Gyro-)commutative law
    torch.testing.assert_close(
        manifold.mobius_add(x, y, c),
        manifold._gyration(x, y, manifold.mobius_add(y, x, c), c),
        atol=atol,
        rtol=rtol,
    )
    # Gyrosum inversion law
    torch.testing.assert_close(
        -manifold.mobius_add(x, y, c),
        manifold._gyration(x, y, manifold.mobius_add(-y, -x, c), c),
        atol=atol,
        rtol=rtol,
    )
    # Left (gyro-)associative law
    torch.testing.assert_close(
        manifold.mobius_add(x, manifold.mobius_add(y, z, c), c),
        manifold.mobius_add(manifold.mobius_add(x, y, c), manifold._gyration(x, y, z, c), c),
        atol=atol,
        rtol=rtol,
    )
    # Right (gyro-)associative law
    torch.testing.assert_close(
        manifold.mobius_add(manifold.mobius_add(x, y, c), z, c),
        manifold.mobius_add(x, manifold.mobius_add(y, manifold._gyration(y, x, z, c), c), c),
        atol=atol,
        rtol=rtol,
    )
    # Left cancellation law
    torch.testing.assert_close(
        manifold._gyration(x, y, manifold.mobius_add(z, a, c), c),
        manifold.mobius_add(manifold._gyration(x, y, z, c), manifold._gyration(x, y, a, c), c),
        atol=atol,
        rtol=rtol,
    )
    # Left loop property
    torch.testing.assert_close(
        manifold._gyration(x, y, z, c),
        manifold._gyration(manifold.mobius_add(x, y, c), y, z, c),
        atol=atol,
        rtol=rtol,
    )
    # Right loop property
    torch.testing.assert_close(
        manifold._gyration(x, y, z, c),
        manifold._gyration(x, manifold.mobius_add(y, x, c), z, c),
        atol=atol,
        rtol=rtol,
    )
    # Identity gyroautomorphism property
    r1 = torch.rand((x.shape[0], 1), dtype=torch.float64)
    r2 = torch.rand((x.shape[0], 1), dtype=torch.float64)
    torch.testing.assert_close(
        manifold._gyration(manifold.mobius_left_scalarmul(r1, x, c), manifold.mobius_left_scalarmul(r2, x, c), y, c),
        y,
        atol=atol,
        rtol=rtol,
    )
    # Gyroautomorphism property
    torch.testing.assert_close(
        manifold._gyration(x, y, manifold.mobius_left_scalarmul(r1, z, c), c),
        manifold.mobius_left_scalarmul(r1, manifold._gyration(x, y, z, c), c),
        atol=atol,
        rtol=rtol,
    )
    # First gyrogroup theorems
    torch.testing.assert_close(manifold._gyration(torch.zeros_like(x), y, z, c), z, atol=atol, rtol=rtol)
    torch.testing.assert_close(manifold._gyration(-y, y, z, c), z, atol=atol, rtol=rtol)
    torch.testing.assert_close(manifold._gyration(x, x, z, c), z, atol=atol, rtol=rtol)
    torch.testing.assert_close(manifold._gyration(x, y, torch.zeros_like(x), c), torch.zeros_like(x), atol=atol, rtol=rtol)
    torch.testing.assert_close(manifold._gyration(x, y, -z, c), -manifold._gyration(x, y, z, c), atol=atol, rtol=rtol)
    torch.testing.assert_close(manifold._gyration(x, torch.zeros_like(x), z, c), z, atol=atol, rtol=rtol)
    # TODO: conformality


# @pytest.mark.skip(reason="Currently failing")
def test_left_scalarmul(
    manifold: PoincareBall, uniform_ball_points: torch.Tensor, c: torch.Tensor, tolerance: Tuple[float, float]
) -> None:
    """Test the left scalar multiplication operation."""
    atol, rtol = tolerance
    identity = torch.ones((uniform_ball_points.shape[0], 1), dtype=torch.float64)
    r1 = torch.rand((uniform_ball_points.shape[0], 1), dtype=torch.float64)
    r2 = torch.rand((uniform_ball_points.shape[0], 1), dtype=torch.float64)
    # Multiplicative identity
    torch.testing.assert_close(
        manifold.mobius_left_scalarmul(identity, uniform_ball_points, c), uniform_ball_points, atol=atol, rtol=rtol
    )
    # Gyroaddition
    n = random.randint(3, 10)
    n_sum = torch.zeros_like(uniform_ball_points)
    for _ in range(n):
        n_sum = manifold.mobius_add(n_sum, uniform_ball_points, c)
    torch.testing.assert_close(
        n_sum, manifold.mobius_left_scalarmul(n * identity, uniform_ball_points, c), atol=atol, rtol=rtol
    )
    # Distributive laws
    torch.testing.assert_close(
        manifold.mobius_left_scalarmul(r1 + r2, uniform_ball_points, c),
        manifold.mobius_add(
            manifold.mobius_left_scalarmul(r1, uniform_ball_points, c),
            manifold.mobius_left_scalarmul(r2, uniform_ball_points, c),
            c,
        ),
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(
        manifold.mobius_left_scalarmul(-r1, uniform_ball_points, c),
        manifold.mobius_left_scalarmul(r1, -uniform_ball_points, c),
        atol=atol,
        rtol=rtol,
    )
    # Associative law
    torch.testing.assert_close(
        manifold.mobius_left_scalarmul(r1 * r2, uniform_ball_points, c),
        manifold.mobius_left_scalarmul(r1, manifold.mobius_left_scalarmul(r2, uniform_ball_points, c), c),
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(
        manifold.mobius_left_scalarmul(r1 * r2, uniform_ball_points, c),
        manifold.mobius_left_scalarmul(r2, manifold.mobius_left_scalarmul(r1, uniform_ball_points, c), c),
        atol=atol,
        rtol=rtol,
    )
    # Scaling property
    left_side = manifold.mobius_left_scalarmul(torch.abs(r1), uniform_ball_points, c)
    left_side /= manifold.mobius_left_scalarmul(r1, uniform_ball_points, c).norm(dim=-1, keepdim=True, p=2)
    torch.testing.assert_close(
        left_side, uniform_ball_points / uniform_ball_points.norm(dim=-1, keepdim=True, p=2), atol=atol, rtol=rtol
    )
    # Homogenity property
    torch.testing.assert_close(
        manifold.mobius_left_scalarmul(r1, uniform_ball_points, c).norm(dim=-1, keepdim=True, p=2),
        manifold.mobius_left_scalarmul(torch.abs(r1), uniform_ball_points.norm(dim=-1, keepdim=True, p=2), c),
        atol=atol,
        rtol=rtol,
    )
    # Multiplicative closedness
    assert manifold.is_in_manifold(manifold.mobius_left_scalarmul(r1, uniform_ball_points, c), c)
    assert manifold.is_in_manifold(manifold.mobius_left_scalarmul(r2, uniform_ball_points, c), c)


# @pytest.mark.skip(reason="Currently failing")
def test_dist(
    manifold: PoincareBall, uniform_ball_points: torch.Tensor, c: torch.Tensor, tolerance: Tuple[float, float]
) -> None:
    x, y = uniform_ball_points.split(uniform_ball_points.shape[0] // 2, dim=0)
    atol, rtol = tolerance
    assert torch.isfinite(manifold.dist_from_0(x, c)).all()
    assert torch.isfinite(manifold.dist(x, y, c)).all()
    torch.testing.assert_close(manifold.dist(x, y, c), manifold.dist(y, x, c), atol=atol, rtol=rtol)
    torch.testing.assert_close(
        manifold.dist(uniform_ball_points, uniform_ball_points, c),
        torch.zeros((uniform_ball_points.shape[0], 1), dtype=torch.float64),
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(
        manifold.dist(uniform_ball_points, torch.zeros_like(uniform_ball_points), c),
        manifold.dist_from_0(uniform_ball_points, c),
        atol=atol,
        rtol=rtol,
    )


# TODO: Implement as pytest
def test_expmap_and_logmap() -> None:
    pass
    # def test_expmap_logmap(a, b, manifold, dtype):


#     # this test appears to be numerical unstable once a and b may appear on the opposite sides
#     bh = manifold.expmap(x=a, u=manifold.logmap(a, b))
#     tolerance = {torch.float32: dict(rtol=1e-5, atol=5e-5), torch.float64: dict()}
#     np.testing.assert_allclose(bh.detach(), b.detach(), **tolerance[dtype])
#     bh.sum().backward()
#     assert torch.isfinite(a.grad).all()
#     assert torch.isfinite(b.grad).all()
#     assert torch.isfinite(manifold.k.grad).all()


# def test_expmap0_logmap0(a, manifold, dtype):
#     # this test appears to be numerical unstable once a and b may appear on the opposite sides
#     v = manifold.logmap0(a)
#     norm = manifold.norm(torch.zeros_like(v), v, keepdim=True)
#     dist = manifold.dist0(a, keepdim=True)
#     bh = manifold.expmap0(v)
#     tolerance = {torch.float32: dict(atol=1e-5, rtol=1e-5), torch.float64: dict()}
#     np.testing.assert_allclose(bh.detach(), a.detach(), **tolerance[dtype])
#     np.testing.assert_allclose(norm.detach(), dist.detach(), **tolerance[dtype])
#     (bh.sum() + dist.sum()).backward()
#     assert torch.isfinite(a.grad).all()
#     assert torch.isfinite(manifold.k.grad).all()


# TODO: Implement as pytest
def test_ptransp() -> None:
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


# TODO: Implement as pytest
def test_tangent_norm() -> None:
    pass


#         tangent = unary_case.v
#         point = unary_case.x
#         tangent_norm = unary_case.manifold.norm(point, tangent)
#         try:
#             new_point = unary_case.manifold.expmap(point, tangent)
#             dist = unary_case.manifold.dist(point, new_point)
#             np.testing.assert_allclose(dist.detach(), tangent_norm.detach())
#         except NotImplementedError:
#             pytest.skip("dist is not implemented for {}".format(unary_case.manifold))


# TODO: Fix this
# def test_matvec(self) -> None:
#     pass


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
