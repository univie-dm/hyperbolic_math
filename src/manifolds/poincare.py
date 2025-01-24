"""PoincareBall manifold."""

import traceback

import torch

from ..utils.math_utils import arcosh, artanh, tanh, arsinh
from .manifold import Manifold


class PoincareBall(Manifold):
    """
    PoincareBall manifold class.
    [Gyrospace/Mobius version]

    Convention: x0^2 + x1^2 + ... + xd^2 < 1/c  with c > 0 and sectional curvature -c.

    Parameters
    ----------
    dtype : torch.dtype
        Data type that is used for the computations. Sets the tolerances for numerical errors.
    """

    def __init__(self):
        super().__init__()
        self.name = "PoincareBall"
        self.min_enorm = 1e-15
        self.max_enorm_eps = 5e-15

    def _lambda(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Compute the conformal factor(s) at the PoincareBall point(s) x.

        Parameters
        ----------
        x : torch.Tensor
            PoincareBall point(s)
        c : torch.Tensor
            magnitude of sectional curvature

        Returns
        -------
        res : torch.Tensor
            The conformal factor(s)

        References
        ----------
        Ganea, Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic neural networks."
            Advances in neural information processing systems 31 (2018).

        Stability
        ---------
        Roughly bounded from above by 1/(c.sqrt()*self.max_enorm_eps)
        """
        x2 = x.pow(2).sum(dim=-1, keepdim=True)
        denom = (1.0 - c * x2).clamp_min(2 * c.sqrt() * self.max_enorm_eps - c * self.max_enorm_eps ** 2)
        res = 2 / denom
        return res

    def _gyration(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Compute the gyration gyr[x,y]z of PoincareBall points x, y and z.
        [Operator to restore commutativity and associativity of mobius addition/scalar_mul]

        Parameters
        ----------
        x : torch.Tensor
            PoincareBall point(s)
        y : torch.Tensor
            PoincareBall point(s)
        z : torch.Tensor
            PoincareBall point(s)
        c : torch.Tensor
            magnitude of sectional curvature

        Returns
        -------
        res : torch.Tensor
            The gyration gyr[x,y]z

        References
        ----------
        Ungar, Abraham. A gyrovector space approach to hyperbolic geometry. Springer Nature, 2022.

        Stability
        ---------
        """
        c2 = c**2
        xx = x.pow(2).sum(dim=-1, keepdim=True)
        yy = y.pow(2).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        xz = (x * z).sum(dim=-1, keepdim=True)
        yz = (y * z).sum(dim=-1, keepdim=True)
        a = -c2 * xz * yy + c * yz + 2 * c2 * xy * yz
        b = -c2 * yz * xx - c * xz
        num = 2 * (a * x + b * y)
        denom = 1 + 2 * c * xy + c2 * xx * yy
        res = z + num / denom
        if not torch.all(torch.isfinite(res)):
            print(f"_gyration: ZeroDivisionError")
            traceback.print_stack(limit=-1)
            # Stable case
            res = z + num / denom.clamp_min(self.min_enorm)
        return res

    def addition(self, x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Add PoincareBall point(s) y to PoincareBall point(s) x using mobius gyrovector addition.
        Non-commutative and non-associative!
        [Mobius version]

        Parameters
        ----------
        x : torch.Tensor
            PoincareBall point(s)
        y : torch.Tensor
            PoincareBall point(s)
        c : torch.Tensor
            magnitude of sectional curvature

        Returns
        -------
        res : torch.Tensor
            The sum(s) of x and y

        References
        ----------
        Ungar, Abraham. A gyrovector space approach to hyperbolic geometry. Springer Nature, 2022.

        Stability
        TODO
        ---------
        """
        x2 = x.pow(2).sum(dim=-1, keepdim=True)
        y2 = y.pow(2).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c**2 * x2 * y2
        res = num / denom
        if not torch.all(torch.isfinite(res)):
            print(f"addition: ZeroDivisionError")
            traceback.print_stack(limit=-1)
            # Stable case
            res = num / denom.clamp_min(self.min_enorm)
        return res

    def scalar_mul(self, r: torch.Tensor, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Multiply PoincareBall point(s) x with scalar(s) r.

        Parameters
        ----------
        r : torch.Tensor
            scalar factor(s)
        x : torch.Tensor
            PoincareBall point(s)
        c : torch.Tensor
            magnitude of sectional curvature

        Returns
        -------
        res : torch.Tensor
            The product(s) of r and x

        References
        ----------
        Ganea, Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic neural networks."
            Advances in neural information processing systems 31 (2018).

        Stability
        ---------
        The PoincareBall multiplication converges towards the tangent space multiplication
            as the norm of vector(s) x approaches zero, since tanh(z) ~ artanh(z) ~ z for small z.
        Backprojection via self.proj() is applied if the result would be rounded to the boundary.
        """
        x_norm = x.norm(p=2, dim=-1, keepdim=True)
        c_norm_prod = c.sqrt() * x_norm
        res_unclipped = tanh(r * artanh(c_norm_prod)) / c_norm_prod * x
        if not torch.all(torch.isfinite(res_unclipped)):
            print(f"scalar_mul: ZeroDivisionError")
            traceback.print_stack(limit=-1)
            # Stable case
            x_norm = x.norm(p=2, dim=-1, keepdim=True).clamp_min(self.min_enorm)
            c_norm_prod = c.sqrt() * x_norm
            res_unclipped = tanh(r * artanh(c_norm_prod)) / c_norm_prod * x
        res = self.proj(res_unclipped, c)
        if torch.not_equal(res, res_unclipped).all():
            print(f"scalar_mul: Norm clipping applied")
        return res

    def matvec_mul(self, m: torch.Tensor, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Multiply PoincareBall point(s) x with (Euclidean) matrix m from the left.

        Parameters
        ----------
        m : torch.Tensor
            (Euclidean) matrix
        x : torch.Tensor
            PoincareBall point(s)
        c : torch.Tensor
            magnitude of sectional curvature

        Returns
        -------
        res : torch.Tensor
            The product(s) of m and x defined as expmap_0(m * logmap_0(x))

        References
        ----------
        Ganea, Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic neural networks."
            Advances in neural information processing systems 31 (2018).

        Stability
        ---------
        TODO: We have sigma_min*||x|| <= ||Mx|| <= ||M||*||x|| <= sigma_max*||x||
        ||x||-> 0 implies that ||Mx|| -> 0 for reasonably bounded M
        ||Mx|| -> 0 may cause problems if ||x|| is very large. How to deal with this?
        """
        sqrt_c = c.sqrt()
        mx = x @ m
        x_norm = x.norm(p=2, dim=-1, keepdim=True)
        mx_norm = mx.norm(p=2, dim=-1, keepdim=True)
        res = tanh(artanh(sqrt_c * x_norm) / x_norm * mx_norm) / (mx_norm * sqrt_c) * mx

        condition = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
        res_0 = torch.zeros(1, dtype=res.dtype, device=res.device)
        res_1 = torch.where(condition, res_0, res)

        if not torch.equal(res, res_1):
            # Pretty sure its the same, but why did geoopt do the res_1 way?
            print(f"res {res}")
            print(f"res_1 {res_1}")

        if not torch.all(torch.isfinite(res)):
            print(f"matvec_mul: ZeroDivisionError")
            traceback.print_stack(limit=-1)
            # Stable case
            x_norm = x.norm(p=2, dim=-1, keepdim=True).clamp_min(self.min_enorm)
            mx_norm = mx.norm(p=2, dim=-1, keepdim=True).clamp_min(self.min_enorm)
            res_c = tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) / (mx_norm * sqrt_c) * mx
            condition = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
            res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
            res = torch.where(condition, res_0, res_c)
        return res

    def dist2hyperplane(self, x: torch.Tensor, m: torch.Tensor, p: torch.Tensor, c: torch.Tensor,
                        signed: bool = False, scaled: bool = False) -> torch.Tensor:
        """
        #TODO
        """
        sqrt_c = c.sqrt()
        m_norm = m.norm(p=2, dim=0, keepdim=True).clamp_min(self.min_enorm)
        sub = self.addition(-p, x, c)
        msub = sub @ m
        if not signed:
            msub = msub.abs()
        num = 2.0 * sqrt_c * msub
        sub_norm2 = sub.pow(2).sum(dim=-1, keepdim=True)
        denom = m_norm * (1 - c * sub_norm2).clamp_min(2 * sqrt_c * self.max_enorm_eps - c * self.max_enorm_eps ** 2)
        res = arsinh(num / denom) / sqrt_c
        if scaled:
            res = res * m_norm
        return res
    
    def dist2hyperplane_correct(self, x: torch.Tensor, m: torch.Tensor, p: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        #TODO
        """
        # Perform matrix-vector multiplication in the tangent space
        # The row vectors of m are the hyperplane normals
        sub = self.addition(-p, x, c)
        msub = sub @ m
        # Determine which side of the hyperplanes the point(s) on
        orientation = torch.sign(msub)
        # Get the lengths of the hyperplane normals
        m_norm = self.tangent_norm(m.T, p, c).T
        # Compute the geodesic distance(s) of the point(s) to the hyperplane
        sqrt_c = c.sqrt()
        denom = m.norm(p=2, dim=0, keepdim=True).clamp_min(self.min_enorm)
        dist2hyp = arsinh(self._lambda(sub, c) * sqrt_c * msub.abs() / denom) / sqrt_c
        res = orientation * dist2hyp * m_norm
        return res

    # def dist2hyperplane_pp(self, x: torch.Tensor, m: torch.Tensor, p: torch.Tensor, c: torch.Tensor,
    #                        signed: bool = False, scaled: bool = False) -> torch.Tensor:
    #     z_norm = z.norm(dim=-2, keepdim=True, p=2)
    #     z_unit = z / z_norm.clamp_min(1e-15)

    #     x2 = x.pow(2).sum(dim=-1, keepdim=True)

    #     distance = (
    #         arsin_k(
    #             2 / (1 + k * p.pow(2).sum(dim=-2, keepdim=True)).clamp_min(1e-15) * (
    #                 torch.matmul(x, z_unit)
    #                 - (1 + 2 * k * torch.matmul(x, p) - k * x2) 
    #                 / (1 + k * x2).clamp_min(1e-15)
    #                     * (p * z_unit).sum(dim=-2, keepdim=True)
    #             ), 
    #             k
    #             )
    #     )

    #     return 2 * distance * z_norm

    def dist(self, x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, version: str="metric_tensor") -> torch.Tensor:
        """
        Compute the geodesic distance(s) between PoincareBall points x and y.
        [Mobius version]

        Parameters
        ----------
        x : torch.Tensor
            PoincareBall point(s)
        y : torch.Tensor
            PoincareBall point(s)
        c : torch.Tensor
            magnitude of sectional curvature
        version : str
            version of the geodesic distance to compute (default: "metric_tensor")
            ['mobius': Mobius-dist, 'mobius_symmetric': Symmetrized mobius-dist, metric_tensor: Metric-tensor-induced-dist]

        Returns
        -------
        res : torch.Tensor
            The geodesic distance(s) between x and y

        References
        ----------
        Ganea, Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic neural networks."
            Advances in neural information processing systems 31 (2018).

        Stability
        ---------
        Mobius-dist version is more stable for boundary points, but the Metric-tensor-induced-dist is 75% faster.
        """
        if version == "mobius":
            # Mobius-dist
            sqrt_c = c.sqrt()
            dist_c = artanh(sqrt_c * self.addition(-x, y, c).norm(p=2, dim=-1, keepdim=True))
            res = 2 * dist_c / sqrt_c
        elif version == "mobius_symmetric":
            # TODO check if this is algebraically allowed, numerically OK
            # Symmetrized mobius-dist
            sqrt_c = c.sqrt()
            dist_c_1 = artanh(sqrt_c * self.addition(-x, y, c).norm(p=2, dim=-1, keepdim=True))
            dist_c_2 = artanh(sqrt_c * self.addition(-y, x, c).norm(p=2, dim=-1, keepdim=True))
            res = (dist_c_1 + dist_c_2) / sqrt_c
        elif version == "metric_tensor":
            # Metric-tensor-induced-dist
            x_sqnorm = x.pow(2).sum(dim=-1, keepdim=True)
            y_sqnorm = y.pow(2).sum(dim=-1, keepdim=True)
            xy_diff_sqnorm = (x - y).pow(2).sum(dim=-1, keepdim=True)
            res = 1 + 2 * c * xy_diff_sqnorm / ((1 - c * x_sqnorm) * (1 - c * y_sqnorm))
            condition = res < 1 + self.min_enorm
            res = torch.where(condition, torch.zeros_like(res), arcosh(res) / c.sqrt())
        else:
            raise ValueError(f"Unknown version: {version}")
        return res

    def dist_0(self, x: torch.Tensor, c: torch.Tensor, version: str="metric_tensor") -> torch.Tensor:
        """
        Compute the geodesic distance(s) of PoincareBall point(s) x from/to the PoincareBall origin.

        Parameters
        ----------
        x : torch.Tensor
            PoincareBall point(s)
        c : torch.Tensor
            magnitude of sectional curvature
        version : str
            version of the geodesic distance to compute (default: "metric_tensor")
            ['mobius': Mobius-dist, 'mobius_symmetric': Symmetrized mobius-dist, metric_tensor: Metric-tensor-induced-dist]

        Returns
        -------
        res : torch.Tensor
            The geodesic distance(s) of x from/to the PoincareBall origin

        References
        ----------
        Ganea, Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic neural networks."
            Advances in neural information processing systems 31 (2018).
        """
        if version in ["mobius", "mobius_symmetric"]:
            # Mobius-dist/Symmetrized mobius-dist
            sqrt_c = c.sqrt()
            dist_c = artanh(sqrt_c * x.norm(p=2, dim=-1, keepdim=True))
            res = 2 * dist_c / sqrt_c
        elif version == "metric_tensor":
            # Metric-tensor-induced-dist
            x_sqnorm = x.pow(2).sum(dim=-1, keepdim=True)
            res = 1 + 2 * c * x_sqnorm / (1 - c * x_sqnorm)
            condition = res < 1 + self.min_enorm
            res = torch.where(condition, torch.zeros_like(res), arcosh(res) / c.sqrt())
        else:
            raise ValueError(f"Unknown version: {version}")
        return res

    def expmap(self, v: torch.Tensor, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Map tangent vector(s) v at PoincareBall point(s) x to the PoincareBall.
        [Exponential map - Mobius version]

        Parameters
        ----------
        v : torch.Tensor
            vector(s) in the tangent space(s) of x
        x : torch.Tensor
            PoincareBall point(s)
        c : torch.Tensor
            magnitude of sectional curvature

        Returns
        -------
        res : torch.Tensor
            The resulting PoincareBall point(s) after mapping v to the PoincareBall

        References
        ----------
        Ganea, Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic neural networks."
            Advances in neural information processing systems 31 (2018).

        Stability
        ---------
        TODO: check which clamping works better
        expmap converges towards the mobius addition x+v as the norm of vectors v and x approaches zero,
            since tanh(z) ~ z for small z.
        TODO expmap converges towards ??? as the norm of vector(s) v approaches zero and
            lambda approaches 1/(c.sqrt()*self.max_enorm_eps) since ???.
        self._lambda() is roughly bounded from above by 1/(c.sqrt()*self.max_enorm_eps)
        TODO stability of addition
        """
        v_norm = v.norm(p=2, dim=-1, keepdim=True)
        c_norm_prod = c.sqrt() * v_norm
        second_term_unclipped = tanh(c_norm_prod * self._lambda(x, c) / 2) / c_norm_prod * v
        if not torch.all(torch.isfinite(second_term_unclipped)):
            print(f"expmap: ZeroDivisionError")
            traceback.print_stack(limit=-1)

            # Stable case 1 - norm clamping
            # v_norm = v.norm(p=2, dim=-1, keepdim=True).clamp_min(self.min_enorm)
            # c_norm_prod = c.sqrt() * v_norm
            # second_term_unclipped = tanh(c_norm_prod * self._lambda(x, c) / 2) / c_norm_prod * v

            # Stable case 2 - cnorm clamping
            v_norm = v.norm(p=2, dim=-1, keepdim=True)
            c_norm_prod = (c.sqrt() * v_norm).clamp_min(self.min_enorm)
            second_term_unclipped = tanh(c_norm_prod * self._lambda(x, c) / 2) / c_norm_prod * v

            # Stable case 3 - denom clamping
            # v_norm = v.norm(p=2, dim=-1, keepdim=True)
            # c_norm_prod = c.sqrt() * v_norm
            # second_term_unclipped = tanh(c_norm_prod * self._lambda(x, c) / 2) / (c_norm_prod).clamp_min(self.min_enorm) * v

        second_term = self.proj(second_term_unclipped, c)
        if torch.not_equal(second_term, second_term_unclipped).all():
            print(f"expmap: Norm clipping applied to second_term")

        res_unclipped = self.addition(x, second_term, c)
        res = self.proj(res_unclipped, c)
        if torch.not_equal(res, res_unclipped).all():
            print(f"expmap: Norm clipping applied to res")
        return res

    def expmap_0(self, v: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Map tangent vector(s) v at the PoincareBall origin to the PoincareBall.
        [Exponential map]

        Parameters
        ----------
        v : torch.Tensor
            vector(s) in the tangent space of the PoincareBall origin
        c : torch.Tensor
            magnitude of sectional curvature

        Returns
        -------
        res : torch.Tensor
            The resulting PoincareBall point(s) after mapping v to the PoincareBall

        References
        ----------
        Ganea, Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic neural networks."
            Advances in neural information processing systems 31 (2018).

        Stability
        ---------
        TODO: check which clamping works better
        expmap_0 converges towards the identity map as the norm of vector(s) v approaches zero,
            since tanh(z) ~ z for small z.
        """
        v_norm = v.norm(p=2, dim=-1, keepdim=True)
        c_norm_prod = c.sqrt() * v_norm
        res_unclipped = tanh(c_norm_prod) / c_norm_prod * v
        if not torch.all(torch.isfinite(res_unclipped)):
            print(f"expmap_0: ZeroDivisionError")
            traceback.print_stack(limit=-1)

            # Stable case 1 - norm clamping
            # v_norm = v.norm(p=2, dim=-1, keepdim=True).clamp_min(self.min_enorm)
            # c_norm_prod = c.sqrt() * v_norm

            ## Stable case 2 - cnorm clamping
            v_norm = v.norm(p=2, dim=-1, keepdim=True)
            c_norm_prod = (c.sqrt() * v_norm).clamp_min(self.min_enorm)

            res_unclipped = tanh(c_norm_prod) / c_norm_prod * v
        res = self.proj(res_unclipped, c)
        if torch.not_equal(res, res_unclipped).all():
            ...
            # print(f"expmap_0: Norm clipping applied")
        return res

    def retraction(self, v: torch.Tensor, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        First-order approximation of the exponential map for vector(s) v at PoincareBall point(s) x.
        [Retraction map]

        Parameters
        ----------
        v : torch.Tensor
            vector(s) in the tangent space(s) of x
        x : torch.Tensor
            PoincareBall point(s)
        c : torch.Tensor
            magnitude of sectional curvature

        Returns
        -------
        res : torch.Tensor
            The resulting PoincareBall point(s) after approximate mapping v to the PoincareBall

        References
        ----------
        Gary Bécigneul and Octavian Ganea. "Riemannian adaptive optimization methods."
            International Conference on Learning Representations (2019).
        """
        # always assume v is scaled properly
        approx = x + v
        res = self.proj(approx, c=c)
        return res

    def logmap(self, y: torch.Tensor, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Map PoincareBall point(s) y to the tangent space(s) of PoincareBall point(s) x.
        [Logarithmic map - Mobius version]

        Parameters
        ----------
        y : torch.Tensor
            PoincareBall point(s)
        x : torch.Tensor
            PoincareBall point(s)
        c : torch.Tensor
            magnitude of sectional curvature

        Returns
        -------
        res : torch.Tensor
            The resulting tangent vector(s) after mapping y to the tangent space(s) of x

        References
        ----------
        Ganea, Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic neural networks."
            Advances in neural information processing systems 31 (2018).

        Stability
        ---------
        TODO: check which clamping works better
        logmap converges towards the identity map as the norm of vector(s) y-x approaches zero,
            since artanh(z) ~ z for small z.
        self._lambda() is roughly bounded from above by 1/(c.sqrt()*self.max_enorm_eps)
        ...addition...
        """
        sub = self.addition(-x, y, c)
        sub_norm = sub.norm(p=2, dim=-1, keepdim=True)
        c_norm_prod = c.sqrt() * sub_norm
        res = 2 * artanh(c_norm_prod) / (c_norm_prod * self._lambda(x, c)) * sub
        if not torch.all(torch.isfinite(res)):
            print(f"logmap: ZeroDivisionError")
            traceback.print_stack(limit=-1)

            # Stable case 1 - norm clamping
            # sub_norm = sub.norm(p=2, dim=-1, keepdim=True).clamp_min(self.min_enorm)
            # c_norm_prod = c.sqrt() * sub_norm

            # Stable case 2 - cnorm clamping
            sub_norm = sub.norm(p=2, dim=-1, keepdim=True)
            c_norm_prod = (c.sqrt() * sub_norm).clamp_min(self.min_enorm)

            res = 2 * artanh(c_norm_prod) / (c_norm_prod * self._lambda(x, c)) * sub
        return res

    def logmap_0(self, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Map PoincareBall point(s) y to the tangent space of the PoincareBall origin.
        [Logarithmic map]

        Parameters
        ----------
        y : torch.Tensor
            PoincareBall point(s)
        c : torch.Tensor
            magnitude of sectional curvature

        Returns
        -------
        res : torch.Tensor
            The resulting tangent vector(s) after mapping y to the tangent space of the origin

        References
        ----------
        Ganea, Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic neural networks."
            Advances in neural information processing systems 31 (2018).

        Stability
        ---------
        TODO: check which clamping works better
        logmap_0 converges towards the identity map as the norm of vector(s) y approaches zero,
            since artanh(z) ~ z for small z.
        """
        y_norm = y.norm(p=2, dim=-1, keepdim=True)
        c_norm_prod = c.sqrt() * y_norm
        res = artanh(c_norm_prod) / c_norm_prod * y
        if not torch.all(torch.isfinite(res)):
            print(f"logmap_0: ZeroDivisionError")
            traceback.print_stack(limit=-1)

            # Stable case 1 - norm clamping
            # y_norm = y.norm(p=2, dim=-1, keepdim=True).clamp_min(self.min_enorm)
            # c_norm_prod = c.sqrt() * y_norm

            # Stable case 2 - cnorm clamping
            y_norm = y.norm(p=2, dim=-1, keepdim=True)
            c_norm_prod = (c.sqrt() * y_norm).clamp_min(self.min_enorm)

            res = artanh(c_norm_prod) / c_norm_prod * y
        return res

    def ptransp(self, v: torch.Tensor, x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Parallel transport tangent vector(s) v from the tangent space(s) of PoincareBall point(s) x
        to the tangent space(s) of PoincareBall point(s) y.
        [Mobius version]

        Parameters
        ----------
        v : torch.Tensor
            vector(s) in the tangent space(s) of x
        x : torch.Tensor
            PoincareBall point(s)
        y : torch.Tensor
            PoincareBall point(s)
        c : torch.Tensor
            magnitude of sectional curvature

        Returns
        -------
        res : torch.Tensor
            The parallel transported tangent vector(s)

        References
        ----------
        Ganea, Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic neural networks."
            Advances in neural information processing systems 31 (2018).

        Stability
        ---------
        self._lambda() is roughly bounded from above by 1/(c.sqrt()*self.max_enorm_eps)
        TODO: ...gyr... stability
        """
        conformal_frac = self._lambda(x, c) / self._lambda(y, c)
        res = conformal_frac * self._gyration(y, -x, v, c)
        return res

    def ptransp_0(self, v: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Parallel transport tangent vector(s) v from the tangent space of the PoincareBall origin
        to the tangent space(s) of PoincareBall point(s) y.

        Parameters
        ----------
        v : torch.Tensor
            vector(s) in the tangent space of the PoincareBall origin
        y : torch.Tensor
            PoincareBall point(s)
        c : torch.Tensor
            magnitude of sectional curvature

        Returns
        -------
        res : torch.Tensor
            The parallel transported tangent vector(s)

        References
        ----------
        Ganea, Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic neural networks."
            Advances in neural information processing systems 31 (2018).

        Stability
        ---------
        self._lambda() is roughly bounded from above by 1/(c.sqrt()*self.max_enorm_eps)
        """
        conformal_frac = 2 / self._lambda(y, c)
        res = conformal_frac * v
        return res

    def tangent_inner(self, u: torch.Tensor, v: torch.Tensor, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Compute the inner product(s) between tangent vectors u and v of the tangent space(s) at PoincareBall point(s) x
        with respect to the Riemannian metric of the PoincareBall.

        Parameters
        ----------
        u : torch.Tensor
            vector(s) in the tangent space(s) of x
        v : torch.Tensor
            vector(s) in the tangent space(s) of x
        x : torch.Tensor
            PoincareBall point(s)
        c : torch.Tensor
            magnitude of sectional curvature

        Returns
        -------
        res : torch.Tensor
            The tangent inner product(s) of u and v

        References
        ----------
        Ganea, Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic neural networks."
            Advances in neural information processing systems 31 (2018).

        Stability
        ---------
        self._lambda() is roughly bounded from above by 1/(c.sqrt()*self.max_enorm_eps)
        """
        if v is u:
            res = u.pow(2)
        else:
            res = u * v
        res = res.sum(dim=-1, keepdim=True) * self._lambda(x, c) ** 2
        return res

    def tangent_norm(self, v: torch.Tensor, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Compute the norm(s) of tangent vector(s) v of the tangent space(s) at PoincareBall point(s) x
        with respect to the Riemannian metric of the PoincareBall.

        Parameters
        ----------
        v : torch.Tensor
            vector(s) in the tangent space(s) of x
        x : torch.Tensor
            PoincareBall point(s)
        c : torch.Tensor
            magnitude of sectional curvature

        Returns
        -------
        res : torch.Tensor
            The tangent norm(s) of v

        References
        ----------
        Ganea, Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic neural networks."
            Advances in neural information processing systems 31 (2018).

        Stability
        ---------
        self._lambda() is roughly bounded from above by 1/(c.sqrt()*self.max_enorm_eps)
        """
        v_norm = v.norm(p=2, dim=-1, keepdim=True)
        res = self._lambda(x, c) * v_norm
        return res

    def egrad2rgrad(self, grad: torch.Tensor, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Compute the Riemannian gradient(s) at PoincareBall point(s) x from the Euclidean gradient(s).

        Parameters
        ----------
        grad : torch.Tensor
            Euclidean gradient(s)
        x : torch.Tensor
            PoincareBall point(s)
        c : torch.Tensor
            magnitude of sectional curvature

        Returns
        -------
        res : torch.Tensor
            The Riemannian gradient(s) at x

        References
        ----------
        Bonnabel, Silvere. "Stochastic gradient descent on Riemannian manifolds."
            IEEE Transactions on Automatic Control 58.9 (2013): 2217-2229.

        Stability
        ---------
        self._lambda() is roughly bounded from above by 1/(c.sqrt()*self.max_enorm_eps)
        """
        res = grad / self._lambda(x, c) ** 2
        return res

    def proj(self, x: torch.Tensor, c: torch.Tensor):
        """
        Project point(s) x onto the clipped PoincareBall by restricting the Euclidean norm(s) to 1/c.sqrt()-self.max_enorm_eps.

        Parameters
        ----------
        x : torch.Tensor
            point(s)
        c : torch.Tensor
            magnitude of sectional curvature

        Returns
        -------
        res : torch.Tensor
            The projected PoincareBall point(s)

        References
        ----------
        Nickel, Maximillian, and Douwe Kiela. "Poincaré embeddings for learning hierarchical representations."
            Advances in neural information processing systems 30 (2017).

        Stability
        ---------
        TODO:
        Precision depends on c
        """
        # BUG: Must clamp like them to get their results, can't use enorm here
        if x.dtype == torch.float32:
            eps = 4e-3
        else:
            eps = self.max_enorm_eps
        max_enorm = (1 / c.sqrt()).to(x.dtype) - eps
        assert max_enorm < (1 / c.sqrt()).to(x.dtype)
        x_norm = x.norm(p=2, dim=-1, keepdim=True)
        proj_x = (max_enorm / x_norm) * x
        condition = x_norm > max_enorm
        res = torch.where(condition, proj_x, x)
        return res

    def is_in_manifold(self, x: torch.Tensor, c: torch.Tensor) -> bool:
        """
        Check if point(s) x lie on the PoincareBall.

        Parameters
        ----------
        x : torch.Tensor
            PoincareBall point(s)
        c : torch.Tensor
            magnitude of sectional curvature

        Returns
        -------
        res : bool
            True if all points x lie in the PoincareBall, False otherwise
        """
        x2 = x.pow(2).sum(dim=-1, keepdim=True)
        r2 = torch.ones_like(x2) / c
        res = torch.all(x2 < r2)
        return res

    ################
    ## Miscalleneous (might be useful) - Geoopt implementation available

    # def _proj_to_hyperboloid(self, x, c):
    #     K = 1.0 / c
    #     sqrtK = K**0.5
    #     sqnorm = torch.norm(x, p=2, dim=1, keepdim=True) ** 2
    #     return sqrtK * torch.cat([K + sqnorm, 2 * sqrtK * x], dim=1) / (K - sqnorm)

    # mobius_fn
    # mobius_pointwise_mul
    # geodesic_unit
