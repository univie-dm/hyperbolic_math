## Call this test via command line: python -m pytest tests/test_poincare.py -s

import random
from typing import Tuple, Union

import pytest
import numpy as np
import torch

from src.manifolds import Euclidean, PoincareBall, Hyperboloid


#@pytest.fixture(scope="module", autouse=True, params=range(30, 40))
@pytest.fixture(scope="module")
def seed() -> int:
    """Set the seed(s) for all tests."""
    # Make it for one seed (14) only now
    # seed = request.param
    seed = 14
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed

#@pytest.fixture(scope="module", params=[torch.float32, torch.float64], ids=["float32", "float64"])
@pytest.fixture(scope="module", params=[torch.float64], ids=["float64"])
def dtype(request: pytest.FixtureRequest) -> torch.dtype:
    """Test different data types."""
    return request.param

#TODO add Hyperboloid
@pytest.fixture(scope="module", params=[Euclidean, PoincareBall], ids=["Euclidean", "PoincareBall"])
def manifold(request: pytest.FixtureRequest) -> Union[Euclidean, PoincareBall, Hyperboloid]:
    """Instantiate the manifold(s)."""
    if request.param == Euclidean:
        manifold = Euclidean()
    elif request.param == PoincareBall:
        manifold = PoincareBall()
    elif request.param == Hyperboloid:
        manifold = Hyperboloid()
    return manifold

@pytest.fixture(scope="module")
def c(seed: int, dtype: torch.dtype) -> torch.Tensor:
    """Generate a random curvature magnitude(s)."""
    return torch.empty(1, dtype=dtype).uniform_(torch.finfo(dtype).eps, 5)

@pytest.fixture(scope="module")
def tolerance(dtype: torch.dtype) -> Tuple[float, float]:
    """Set the absolute and relative tolerance(s) for the tests."""
    if dtype == torch.float32:
        #TODO: Test limits for float32
        atol = torch.finfo(dtype).eps
        rtol = torch.finfo(dtype).eps
    elif dtype == torch.float64:
        atol = torch.finfo(dtype).eps
        rtol = rtol = 1e-10
    return atol, rtol

@pytest.fixture(scope="module")
def uniform_manifold_points(
        seed: int,
        dtype: torch.dtype,
        manifold: Union[Euclidean, PoincareBall, Hyperboloid],
        c: torch.Tensor,
    ) -> torch.Tensor:
    """Generate points distributed uniformly at random on the manifold(s)."""
    dim = random.randint(1, 10)
    num_pts = 2_500 * 6
    if isinstance(manifold, Euclidean):
        limit = 1_000
        points = torch.empty((num_pts, dim), dtype=dtype).uniform_(-limit, limit)
    elif isinstance(manifold, PoincareBall):
        random_dirs = torch.normal(
            mean=torch.zeros((num_pts, dim), dtype=dtype),
            std=torch.ones((num_pts, dim), dtype=dtype)
        )
        random_dirs /= random_dirs.norm(p=2, dim=-1, keepdim=True)
        # Generate random radii with probability proportional to the surface area
        random_radii = torch.rand((num_pts, 1), dtype=dtype).pow(1 / dim)
        points = c**-0.5 * (random_dirs * random_radii)
    elif isinstance(manifold, Hyperboloid):
        #TODO
        pass
    return points

def test_is_in_manifold(
        manifold: Union[Euclidean, PoincareBall],
        c: torch.Tensor,
        uniform_manifold_points: torch.Tensor
    ) -> None:
    """Check that all points are on the manifold(s)."""
    assert manifold.is_in_manifold(uniform_manifold_points, c)

def test_addition(
        manifold: Union[Euclidean, PoincareBall, Hyperboloid],
        uniform_manifold_points: torch.Tensor,
        c: torch.Tensor,
        tolerance: Tuple[float, float],
    ) -> None:
    """Test the addition operation."""
    atol, rtol = tolerance
    identity = torch.zeros_like(uniform_manifold_points)
    x, y = uniform_manifold_points.split(uniform_manifold_points.shape[0]//2, dim=0)
    # Additive identity
    torch.testing.assert_close(
        manifold.addition(identity, uniform_manifold_points, c),
        uniform_manifold_points,
        atol=atol,
        rtol=rtol
    )
    torch.testing.assert_close(
        manifold.addition(uniform_manifold_points, identity, c),
        uniform_manifold_points,
        atol=atol,
        rtol=rtol
    )
    # Additive inverse
    torch.testing.assert_close(
        manifold.addition(-uniform_manifold_points, uniform_manifold_points, c),
        identity,
        atol=atol,
        rtol=rtol
    )
    torch.testing.assert_close(
        manifold.addition(uniform_manifold_points, -uniform_manifold_points, c),
        identity,
        atol=atol,
        rtol=rtol
    )
    # # Left cancellation law
    # #TODO: Fails with abs. diff. 2.72e-08 and rel. diff. 5.10e-06
    # torch.testing.assert_close(
    #     manifold.addition(-x, manifold.addition(x, y, c), c),
    #     y,
    #     atol=atol,
    #     rtol=rtol
    # )
    # Distributive law
    torch.testing.assert_close(
        -manifold.addition(x, y, c),
        manifold.addition(-x, -y, c),
        atol=atol,
        rtol=rtol
    )
    # Gyrotriangle inequality
    assert torch.all(
        manifold.addition(x, y, c).norm(p=2, dim=-1, keepdim=True)
        <= manifold.addition(x.norm(p=2, dim=-1, keepdim=True), y.norm(p=2, dim=-1, keepdim=True), c)
    )
    # Numerical additive closedness
    assert manifold.is_in_manifold(manifold.addition(x, y, c), c)

def test_scalar_mul(
        manifold: Union[Euclidean, PoincareBall, Hyperboloid],
        uniform_manifold_points: torch.Tensor,
        c: torch.Tensor,
        tolerance: Tuple[float, float]
    ) -> None:
    """Test the scalar_mul operation."""
    atol, rtol = tolerance
    identity = torch.ones((uniform_manifold_points.shape[0], 1),
                          dtype=uniform_manifold_points.dtype)
    r1 = torch.rand((uniform_manifold_points.shape[0], 1),
                    dtype=uniform_manifold_points.dtype)
    r2 = torch.rand((uniform_manifold_points.shape[0], 1),
                    dtype=uniform_manifold_points.dtype)
    # Multiplicative identity
    torch.testing.assert_close(
        manifold.scalar_mul(identity, uniform_manifold_points, c),
        uniform_manifold_points,
        atol=atol,
        rtol=rtol
    )
    # N-Gyroaddition
    n = random.randint(3, 10)
    n_sum = torch.zeros_like(uniform_manifold_points)
    for _ in range(n):
        n_sum = manifold.addition(n_sum, uniform_manifold_points, c)
    torch.testing.assert_close(
        n_sum,
        manifold.scalar_mul(n * identity, uniform_manifold_points, c),
        atol=atol,
        rtol=rtol
    )
    # Distributive laws
    torch.testing.assert_close(
        manifold.scalar_mul(r1 + r2, uniform_manifold_points, c),
        manifold.addition(
            manifold.scalar_mul(r1, uniform_manifold_points, c),
            manifold.scalar_mul(r2, uniform_manifold_points, c),
            c,
        ),
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(
        manifold.scalar_mul(-r1, uniform_manifold_points, c),
        manifold.scalar_mul(r1, -uniform_manifold_points, c),
        atol=atol,
        rtol=rtol,
    )
    # Associative laws
    torch.testing.assert_close(
        manifold.scalar_mul(r1 * r2, uniform_manifold_points, c),
        manifold.scalar_mul(r1, manifold.scalar_mul(r2, uniform_manifold_points, c), c),
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(
        manifold.scalar_mul(r1 * r2, uniform_manifold_points, c),
        manifold.scalar_mul(r2, manifold.scalar_mul(r1, uniform_manifold_points, c), c),
        atol=atol,
        rtol=rtol,
    )
    # Scaling property
    left_side = manifold.scalar_mul(torch.abs(r1), uniform_manifold_points, c)
    left_side /= manifold.scalar_mul(r1, uniform_manifold_points, c).norm(p=2, dim=-1, keepdim=True)
    torch.testing.assert_close(
        left_side,
        uniform_manifold_points / uniform_manifold_points.norm(p=2, dim=-1, keepdim=True),
        atol=atol,
        rtol=rtol
    )
    # Homogenity property
    torch.testing.assert_close(
        manifold.scalar_mul(r1, uniform_manifold_points, c).norm(p=2, dim=-1, keepdim=True),
        manifold.scalar_mul(torch.abs(r1), uniform_manifold_points.norm(p=2, dim=-1, keepdim=True), c),
        atol=atol,
        rtol=rtol,
    )
    # Numerical multiplicative closedness
    assert manifold.is_in_manifold(manifold.scalar_mul(r1, uniform_manifold_points, c), c)
    assert manifold.is_in_manifold(manifold.scalar_mul(r2, uniform_manifold_points, c), c)
    # Numerical stability
    r_zero = torch.tensor(0, dtype=uniform_manifold_points.dtype)
    r_small = torch.tensor(atol, dtype=uniform_manifold_points.dtype)
    r_huge = torch.tensor(1_000_000, dtype=uniform_manifold_points.dtype)
    v_eps_norm = torch.zeros((1, uniform_manifold_points.shape[1]), dtype=uniform_manifold_points.dtype)
    v_eps_norm[0, 0] = atol
    # Stability of multiplication with zero scalars
    res = manifold.scalar_mul(r_zero, uniform_manifold_points, c)
    assert torch.isfinite(res).all()
    assert manifold.is_in_manifold(res, c)
    torch.testing.assert_close(
        res,
        torch.zeros_like(uniform_manifold_points),
        atol=atol,
        rtol=rtol
    )
    res = manifold.scalar_mul(r_zero, v_eps_norm, c)
    assert torch.isfinite(res).all()
    assert manifold.is_in_manifold(res, c)
    torch.testing.assert_close(
        res,
        torch.zeros_like(v_eps_norm),
        atol=atol,
        rtol=rtol
    )
    # Stability of multiplication with small scalars
    res = manifold.scalar_mul(r_small, v_eps_norm, c)
    assert torch.isfinite(res).all()
    assert manifold.is_in_manifold(res, c)
    assert res[0, 0] > r_zero
    torch.testing.assert_close(
        res[0, 1:],
        torch.zeros_like(res[0, 1:]),
        atol=atol,
        rtol=rtol
    )
    # Stability of multiplication with large scalars
    res = manifold.scalar_mul(r_huge, uniform_manifold_points, c)
    assert torch.isfinite(res).all()
    assert manifold.is_in_manifold(res, c)
    res = manifold.scalar_mul(r_huge, v_eps_norm, c)
    assert torch.isfinite(res).all()
    assert manifold.is_in_manifold(res, c)
    assert res[0, 0] > r_zero
    torch.testing.assert_close(
        res[0, 1:],
        torch.zeros_like(res[0, 1:]),
        atol=atol,
        rtol=rtol
    )

@pytest.mark.skip(reason="not implemented yet")
def test_matvec_mul(
        manifold: Union[Euclidean, PoincareBall, Hyperboloid],
        uniform_manifold_points: torch.Tensor,
        c: torch.Tensor,
        tolerance: Tuple[float, float]
    ) -> None:
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

def test_dist(
        manifold: Union[Euclidean, PoincareBall, Hyperboloid],
        uniform_manifold_points: torch.Tensor,
        c: torch.Tensor,
        tolerance: Tuple[float, float]
    ) -> None:
    """Test the dist and dist_0 operations."""
    atol, rtol = tolerance
    x, y, z = uniform_manifold_points.split(uniform_manifold_points.shape[0]//3, dim=0)
    assert torch.isfinite(manifold.dist_0(x, c)).all()
    assert torch.isfinite(manifold.dist(x, y, c)).all()
    # Reflexivity
    torch.testing.assert_close(
        manifold.dist(uniform_manifold_points, uniform_manifold_points, c),
        torch.zeros((uniform_manifold_points.shape[0], 1), dtype=uniform_manifold_points.dtype),
        atol=atol,
        rtol=rtol,
    )
    # Symmetry
    torch.testing.assert_close(
        manifold.dist(x, y, c),
        manifold.dist(y, x, c),
        atol=atol,
        rtol=rtol
    )
    # Triangle inequality
    assert torch.all(
        manifold.dist(x, z, c) <= manifold.dist(x, y, c) + manifold.dist(y, z, c)
    )
    # Consistency of dist with dist_0
    torch.testing.assert_close(
        manifold.dist(uniform_manifold_points, torch.zeros_like(uniform_manifold_points), c),
        manifold.dist_0(uniform_manifold_points, c),
        atol=atol,
        rtol=rtol,
    )

def test_expmap_and_logmap(
        manifold: Union[Euclidean, PoincareBall, Hyperboloid],
        uniform_manifold_points: torch.Tensor,
        c: torch.Tensor,
        tolerance: Tuple[float, float]
    ) -> None:
    """Test the expmap, expmap_0, logmap and logmap_0 operations."""
    atol, rtol = tolerance
    x, y = uniform_manifold_points.split(uniform_manifold_points.shape[0]//2, dim=0)
    v = torch.empty_like(uniform_manifold_points).uniform_(-1_000, 1_000)
    # Numerical stability of expmap/expmap_0
    v_manif = manifold.expmap(v, uniform_manifold_points, c)
    assert torch.isfinite(v_manif).all()
    assert manifold.is_in_manifold(v_manif, c)
    v_manif = manifold.expmap_0(v, c)
    assert torch.isfinite(v_manif).all()
    assert manifold.is_in_manifold(v_manif, c)
    # Numerical stability of logmap/logmap_0
    if isinstance(manifold, Hyperboloid):
        manifold.is_in_tangent_space(manifold.logmap(uniform_manifold_points, c), c)
        manifold.is_in_tangent_space(manifold.logmap_0(uniform_manifold_points, c), c)
    # Stability of inverse operations
    # Note: logmap(expmap(y, x), x) != y and logmap_0(expmap_0(v)) != v
    # This is because exmap/expmap_0 uses backproj. which is not injective
    res = manifold.expmap(manifold.logmap(y, x, c), x, c)
    assert torch.isfinite(res).all()
    assert manifold.is_in_manifold(res, c)
    # TODO: The following assertion should hold, but does not. WHY?
    # torch.testing.assert_close(
    #     res,
    #     y,
    #     atol=atol,
    #     rtol=rtol
    # )
    res = manifold.expmap_0(manifold.logmap_0(uniform_manifold_points, c), c)
    assert torch.isfinite(res).all()
    assert manifold.is_in_manifold(res, c)
    torch.testing.assert_close(
        res,
        uniform_manifold_points,
        atol=atol,
        rtol=rtol
    )
    # Consistency of expmap/logmap with expmap_0/logmap_0
    torch.testing.assert_close(
        manifold.expmap(v, torch.zeros_like(v), c),
        manifold.expmap_0(v, c),
        atol=atol,
        rtol=rtol
    )
    torch.testing.assert_close(
        manifold.logmap(uniform_manifold_points, torch.zeros_like(uniform_manifold_points), c),
        manifold.logmap_0(uniform_manifold_points, c),
        atol=atol,
        rtol=rtol
    )
    # Consistency of dist_0 and logmap_0
    torch.testing.assert_close(
        manifold.dist_0(uniform_manifold_points, c),
        manifold.tangent_norm(manifold.logmap_0(uniform_manifold_points, c), torch.zeros_like(uniform_manifold_points), c),
        atol=atol,
        rtol=rtol
    )



@pytest.mark.skip(reason="not implemented yet")
def test_tangent_inner_and_norm(
        manifold: Union[Euclidean, PoincareBall, Hyperboloid],
        uniform_manifold_points: torch.Tensor,
        c: torch.Tensor,
        tolerance: Tuple[float, float]
    ) -> None:
    """Test the tangent_inner and tangent_norm operations."""
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

@pytest.mark.skip(reason="not implemented yet")
def test_ptransp(
        manifold: Union[Euclidean, PoincareBall, Hyperboloid],
        uniform_manifold_points: torch.Tensor,
        c: torch.Tensor,
        tolerance: Tuple[float, float]
    ) -> None:
    """Test the ptransp and ptransp_0 operations."""
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

#TODO:
@pytest.mark.skip(reason="needs recheck")
def test_gyration(
        manifold: Union[Euclidean, PoincareBall, Hyperboloid],
        uniform_manifold_points: torch.Tensor,
        c: torch.Tensor,
        tolerance: Tuple[float, float]
    ) -> None:
    """Test the gyration operation of the PoincareBall."""
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


