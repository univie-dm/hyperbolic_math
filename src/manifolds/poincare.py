"""PoincareBall manifold."""

import torch
import traceback

from .manifold import Manifold
from ..utils.math_utils import arcosh, artanh, tanh, arsinh


class PoincareBall(Manifold):
    """
    PoincareBall manifold class.
    Convention: x0^2 + x1^2 + ... + xd^2 < 1/c  with c > 0 and sectional curvature -c.
    """

    def __init__(self, c: torch.Tensor = 1.0, trainable_c: bool=False):
        super().__init__(c, trainable_c)
        self.name = "PoincareBall"
        self.min_enorm = 1e-15
        self.max_enorm_eps = 5e-15

    def _lambda(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the conformal factor(s) at the PoincareBall point(s) x.

        Parameters
        ----------
        x : torch.Tensor
            PoincareBall point(s)

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
        denom = (1.0 - self.c * x2).clamp_min(2 * self.c.sqrt() * self.max_enorm_eps - self.c * self.max_enorm_eps ** 2)
        res = 2 / denom
        return res

    def _gyration(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
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

        Returns
        -------
        res : torch.Tensor
            The gyration gyr[x,y]z

        References
        ----------
        Ungar, Abraham. A gyrovector space approach to hyperbolic geometry. Springer Nature, 2022.

        Stability
        ---------
        Denominator is zero iff x and y are linearly dependent and c=-1/(||x||*||y||), but c > 0.
        """
        c2 = self.c**2
        x2 = x.pow(2).sum(dim=-1, keepdim=True)
        y2 = y.pow(2).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        xz = (x * z).sum(dim=-1, keepdim=True)
        yz = (y * z).sum(dim=-1, keepdim=True)
        a = -c2 * xz * y2 + self.c * yz + 2 * c2 * xy * yz
        b = -c2 * yz * x2 - self.c * xz
        num = 2 * (a * x + b * y)
        denom = 1 + 2 * self.c * xy + c2 * x2 * y2
        res = z + num / denom
        return res

    def addition(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Add PoincareBall point(s) y to PoincareBall point(s) x using mobius gyrovector addition.
        Non-commutative and non-associative!

        Parameters
        ----------
        x : torch.Tensor
            PoincareBall point(s)
        y : torch.Tensor
            PoincareBall point(s)

        Returns
        -------
        res : torch.Tensor
            The sum(s) of x and y

        References
        ----------
        Ungar, Abraham. A gyrovector space approach to hyperbolic geometry. Springer Nature, 2022.

        Stability
        ---------
        Denominator is zero iff x and y are linearly dependent and c=-1/(||x||*||y||), but c > 0.
        """
        x2 = x.pow(2).sum(dim=-1, keepdim=True)
        y2 = y.pow(2).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        num = (1 + 2 * self.c * xy + self.c * y2) * x + (1 - self.c * x2) * y
        denom = 1 + 2 * self.c * xy + self.c**2 * x2 * y2
        res = num / denom
        return res

    def scalar_mul(self, r: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Multiply PoincareBall point(s) x with scalar(s) r.

        Parameters
        ----------
        r : torch.Tensor
            scalar factor(s)
        x : torch.Tensor
            PoincareBall point(s)

        Returns
        -------
        res : torch.Tensor
            The clipped product(s) of r and x

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
        c_norm_prod = self.c.sqrt() * x_norm
        res = tanh(r * artanh(c_norm_prod)) / c_norm_prod * x
        if not torch.all(torch.isfinite(res)):
            print(f"scalar_mul: ZeroDivisionError")
            traceback.print_stack(limit=-1)
            # Stable case
            x_norm = x.norm(p=2, dim=-1, keepdim=True).clamp_min(self.min_enorm)
            c_norm_prod = self.c.sqrt() * x_norm
            res = tanh(r * artanh(c_norm_prod)) / c_norm_prod * x

        res = self.proj(res)
        return res

    def matvec_mul(self, m: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Multiply PoincareBall point(s) x with (Euclidean) matrix m from the left.

        Parameters
        ----------
        m : torch.Tensor
            (Euclidean) matrix
        x : torch.Tensor
            PoincareBall point(s)

        Returns
        -------
        res : torch.Tensor
            The clipped product(s) of m and x defined as expmap_0(m * logmap_0(x))

        References
        ----------
        Ganea, Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic neural networks."
            Advances in neural information processing systems 31 (2018).

        Stability
        ---------
        TODO: We have sigma_min*||x|| <= ||Mx|| <= ||M||*||x|| <= sigma_max*||x||
        ||x||-> 0 implies that ||Mx|| -> 0 for reasonably bounded M
        ||Mx|| -> 0 may cause problems if ||x|| is very large. How to deal with this?
        Backprojection via self.proj() is applied if the result would be rounded to the boundary.
        """
        sqrt_c = self.c.sqrt()
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
        
        res = self.proj(res)
        return res

    def hyperplane_forward(self, x: torch.Tensor, m: torch.Tensor, p: torch.Tensor,
                           signed: bool = False, scaled: bool = False) -> torch.Tensor:
        """
        #TODO
        """
        sqrt_c = self.c.sqrt()
        m_norm = m.norm(p=2, dim=0, keepdim=True).clamp_min(self.min_enorm)
        sub = self.addition(-p, x)
        msub = sub @ m
        if not signed:
            msub = msub.abs()
        num = 2.0 * sqrt_c * msub
        sub_norm2 = sub.pow(2).sum(dim=-1, keepdim=True)
        denom = m_norm * (1 - self.c * sub_norm2).clamp_min(2 * sqrt_c * self.max_enorm_eps - self.c * self.max_enorm_eps ** 2)
        res = arsinh(num / denom) / sqrt_c
        if scaled:
            res = res * m_norm
        return res
    
    def hyperplane_forward_correct(self, x: torch.Tensor, m: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        #TODO
        """
        # Perform matrix-vector multiplication in the tangent space
        # The row vectors of m are the hyperplane normals
        sub = self.addition(-p, x)
        msub = sub @ m
        # Determine which side of the hyperplanes the point(s) on
        orientation = torch.sign(msub)
        # Get the lengths of the hyperplane normals
        m_norm = self.tangent_norm(m.T, p, self.c).T
        # Compute the geodesic distance(s) of the point(s) to the hyperplane
        sqrt_c = self.c.sqrt()
        denom = m.norm(p=2, dim=0, keepdim=True).clamp_min(self.min_enorm)
        dist2hyp = arsinh(self._lambda(sub, self.c) * sqrt_c * msub.abs() / denom) / sqrt_c
        res = orientation * dist2hyp * m_norm
        return res

    def hyperplane_forward_pp(self, x: torch.Tensor, m: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        #TODO
        """
        # z_norm = z.norm(dim=-2, keepdim=True, p=2)
        # z_unit = z / z_norm.clamp_min(1e-15)

        # x2 = x.pow(2).sum(dim=-1, keepdim=True)

        # distance = (
        #     arsin_k(
        #         2 / (1 + k * p.pow(2).sum(dim=-2, keepdim=True)).clamp_min(1e-15) * (
        #             torch.matmul(x, z_unit)
        #             - (1 + 2 * k * torch.matmul(x, p) - k * x2) 
        #             / (1 + k * x2).clamp_min(1e-15)
        #                 * (p * z_unit).sum(dim=-2, keepdim=True)
        #         ), 
        #         k
        #         )
        # )
        #return 2 * distance * z_norm
        raise NotImplementedError

    def dist(self, x: torch.Tensor, y: torch.Tensor, version: str="mobius") -> torch.Tensor:
        """
        Compute the geodesic distance(s) between PoincareBall points x and y.

        Parameters
        ----------
        x : torch.Tensor
            PoincareBall point(s)
        y : torch.Tensor
            PoincareBall point(s)
        version : str
            version of the geodesic distance to compute (default: "mobius")
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
        Metric-tensor-induced-dist is 75% faster than Mobius-dist, but unstable for boundary points.
        """
        #return dist_compiled(x, y, self.c, version)
        if version == "mobius":
            # Mobius-dist
            sqrt_c = self.c.sqrt()
            dist_c = artanh(sqrt_c * self.addition(-x, y).norm(p=2, dim=-1, keepdim=True))
            res = 2 * dist_c / sqrt_c
        elif version == "mobius_symmetric":
            # TODO check if this is algebraically allowed, numerically OK
            # Symmetrized mobius-dist
            sqrt_c = self.c.sqrt()
            dist_c_1 = artanh(sqrt_c * self.addition(-x, y).norm(p=2, dim=-1, keepdim=True))
            dist_c_2 = artanh(sqrt_c * self.addition(-y, x).norm(p=2, dim=-1, keepdim=True))
            res = (dist_c_1 + dist_c_2) / sqrt_c
        elif version == "metric_tensor":
            # Metric-tensor-induced-dist
            x_sqnorm = x.pow(2).sum(dim=-1, keepdim=True)
            y_sqnorm = y.pow(2).sum(dim=-1, keepdim=True)
            xy_diff_sqnorm = (x - y).pow(2).sum(dim=-1, keepdim=True)
            res = 1 + 2 * self.c * xy_diff_sqnorm / ((1 - self.c * x_sqnorm) * (1 - self.c * y_sqnorm))
            condition = res < 1 + self.min_enorm
            res = torch.where(condition, torch.zeros_like(res), arcosh(res) / self.c.sqrt())
        else:
            raise ValueError(f"Unknown version: {version}")
        return res

    def dist_0(self, x: torch.Tensor, version: str="mobius") -> torch.Tensor:
        """
        Compute the geodesic distance(s) of PoincareBall point(s) x from/to the PoincareBall origin.

        Parameters
        ----------
        x : torch.Tensor
            PoincareBall point(s)
        version : str
            version of the geodesic distance to compute (default: "mobius")
            ['mobius': Mobius-dist, 'mobius_symmetric': Symmetrized mobius-dist, metric_tensor: Metric-tensor-induced-dist]

        Returns
        -------
        res : torch.Tensor
            The geodesic distance(s) of x from/to the PoincareBall origin

        References
        ----------
        Ganea, Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic neural networks."
            Advances in neural information processing systems 31 (2018).

        Stability
        ---------
        Metric-tensor-induced-dist is 75% faster than Mobius-dist, but unstable for boundary points.
        """
        #return dist_0_compiled(x, self.c, version)
        if version in ["mobius", "mobius_symmetric"]:
            # Mobius-dist/Symmetrized mobius-dist
            sqrt_c = self.c.sqrt()
            dist_c = artanh(sqrt_c * x.norm(p=2, dim=-1, keepdim=True))
            res = 2 * dist_c / sqrt_c
        elif version == "metric_tensor":
            # Metric-tensor-induced-dist
            x_sqnorm = x.pow(2).sum(dim=-1, keepdim=True)
            res = 1 + 2 * self.c * x_sqnorm / (1 - self.c * x_sqnorm)
            condition = res < 1 + self.min_enorm
            res = torch.where(condition, torch.zeros_like(res), arcosh(res) / self.c.sqrt())
        else:
            raise ValueError(f"Unknown version: {version}")
        return res

    def expmap(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Map tangent vector(s) v at PoincareBall point(s) x to the clipped PoincareBall.
        [Exponential map]

        Parameters
        ----------
        v : torch.Tensor
            vector(s) in the tangent space(s) of x
        x : torch.Tensor
            PoincareBall point(s)

        Returns
        -------
        res : torch.Tensor
            The resulting PoincareBall point(s) after mapping v to the clipped PoincareBall

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
        
        Backprojection via self.proj() is applied if the result would be rounded to the boundary.
        """
        v_norm = v.norm(p=2, dim=-1, keepdim=True)
        c_norm_prod = self.c.sqrt() * v_norm
        second_term = tanh(c_norm_prod * self._lambda(x) / 2) / c_norm_prod * v
        if not torch.all(torch.isfinite(second_term)):
            print(f"expmap: ZeroDivisionError")
            traceback.print_stack(limit=-1)

            # Stable case 1 - norm clamping
            # v_norm = v.norm(p=2, dim=-1, keepdim=True).clamp_min(self.min_enorm)
            # c_norm_prod = self.c.sqrt() * v_norm
            # second_term = tanh(c_norm_prod * self._lambda(x) / 2) / c_norm_prod * v

            # Stable case 2 - cnorm clamping
            v_norm = v.norm(p=2, dim=-1, keepdim=True)
            c_norm_prod = (self.c.sqrt() * v_norm).clamp_min(self.min_enorm)
            second_term = tanh(c_norm_prod * self._lambda(x) / 2) / c_norm_prod * v

            # Stable case 3 - denom clamping
            # v_norm = v.norm(p=2, dim=-1, keepdim=True)
            # c_norm_prod = self.c.sqrt() * v_norm
            # second_term = tanh(c_norm_prod * self._lambda(x) / 2) / (c_norm_prod).clamp_min(self.min_enorm) * v

        # Apply backprojection if the second term of the addition is rounded to the boundary
        second_term = self.proj(second_term)
        res = self.addition(x, second_term)
        return res

    def expmap_0(self, v: torch.Tensor) -> torch.Tensor:
        """
        Map tangent vector(s) v at the PoincareBall origin to the clipped PoincareBall.
        [Exponential map]

        Parameters
        ----------
        v : torch.Tensor
            vector(s) in the tangent space of the PoincareBall origin

        Returns
        -------
        res : torch.Tensor
            The resulting PoincareBall point(s) after mapping v to the clipped PoincareBall

        References
        ----------
        Ganea, Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic neural networks."
            Advances in neural information processing systems 31 (2018).

        Stability
        ---------
        TODO: check which clamping works better
        
        expmap_0 converges towards the identity map as the norm of vector(s) v approaches zero,
        since tanh(z) ~ z for small z.
        
        Backprojection via self.proj() is applied if the result would be rounded to the boundary.
        """
        v_norm = v.norm(p=2, dim=-1, keepdim=True)
        c_norm_prod = self.c.sqrt() * v_norm
        res = tanh(c_norm_prod) / c_norm_prod * v
        if not torch.all(torch.isfinite(res)):
            print(f"expmap_0: ZeroDivisionError")
            traceback.print_stack(limit=-1)

            # Stable case 1 - norm clamping
            # v_norm = v.norm(p=2, dim=-1, keepdim=True).clamp_min(self.min_enorm)
            # c_norm_prod = self.c.sqrt() * v_norm

            ## Stable case 2 - cnorm clamping
            v_norm = v.norm(p=2, dim=-1, keepdim=True)
            c_norm_prod = (self.c.sqrt() * v_norm).clamp_min(self.min_enorm)

            res = tanh(c_norm_prod) / c_norm_prod * v
        
        res = self.proj(res)
        return res

    def retraction(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        First-order approximation of the exponential map for vector(s) v at PoincareBall point(s) x.
        [Retraction map]

        Parameters
        ----------
        v : torch.Tensor
            vector(s) in the tangent space(s) of x
        x : torch.Tensor
            PoincareBall point(s)

        Returns
        -------
        res : torch.Tensor
            The resulting PoincareBall point(s) after approximate mapping v to the clipped PoincareBall

        References
        ----------
        Gary Bécigneul and Octavian Ganea. "Riemannian adaptive optimization methods."
            International Conference on Learning Representations (2019).
        """
        linear_expmap_approx = x + v
        res = self.proj(linear_expmap_approx)
        return res

    def logmap(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Map PoincareBall point(s) y to the tangent space(s) of PoincareBall point(s) x.
        [Logarithmic map]

        Parameters
        ----------
        y : torch.Tensor
            PoincareBall point(s)
        x : torch.Tensor
            PoincareBall point(s)

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
        """
        sub = self.addition(-x, y)
        sub_norm = sub.norm(p=2, dim=-1, keepdim=True)
        c_norm_prod = self.c.sqrt() * sub_norm
        res = 2 * artanh(c_norm_prod) / (c_norm_prod * self._lambda(x)) * sub
        if not torch.all(torch.isfinite(res)):
            print(f"logmap: ZeroDivisionError")
            traceback.print_stack(limit=-1)

            # Stable case 1 - norm clamping
            # sub_norm = sub.norm(p=2, dim=-1, keepdim=True).clamp_min(self.min_enorm)
            # c_norm_prod = self.c.sqrt() * sub_norm

            # Stable case 2 - cnorm clamping
            sub_norm = sub.norm(p=2, dim=-1, keepdim=True)
            c_norm_prod = (self.c.sqrt() * sub_norm).clamp_min(self.min_enorm)

            res = 2 * artanh(c_norm_prod) / (c_norm_prod * self._lambda(x)) * sub
        return res

    def logmap_0(self, y: torch.Tensor) -> torch.Tensor:
        """
        Map PoincareBall point(s) y to the tangent space of the PoincareBall origin.
        [Logarithmic map]

        Parameters
        ----------
        y : torch.Tensor
            PoincareBall point(s)

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
        c_norm_prod = self.c.sqrt() * y_norm
        res = artanh(c_norm_prod) / c_norm_prod * y
        if not torch.all(torch.isfinite(res)):
            print(f"logmap_0: ZeroDivisionError")
            traceback.print_stack(limit=-1)

            # Stable case 1 - norm clamping
            # y_norm = y.norm(p=2, dim=-1, keepdim=True).clamp_min(self.min_enorm)
            # c_norm_prod = self.c.sqrt() * y_norm

            # Stable case 2 - cnorm clamping
            y_norm = y.norm(p=2, dim=-1, keepdim=True)
            c_norm_prod = (self.c.sqrt() * y_norm).clamp_min(self.min_enorm)

            res = artanh(c_norm_prod) / c_norm_prod * y
        return res

    def ptransp(self, v: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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
        conformal_frac = self._lambda(x) / self._lambda(y)
        res = conformal_frac * self._gyration(y, -x, v)
        return res

    def ptransp_0(self, v: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Parallel transport tangent vector(s) v from the tangent space of the PoincareBall origin
        to the tangent space(s) of PoincareBall point(s) y.

        Parameters
        ----------
        v : torch.Tensor
            vector(s) in the tangent space of the PoincareBall origin
        y : torch.Tensor
            PoincareBall point(s)

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
        conformal_frac = 2 / self._lambda(y)
        res = conformal_frac * v
        return res

    def tangent_inner(self, u: torch.Tensor, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
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
        res = res.sum(dim=-1, keepdim=True) * self._lambda(x) ** 2
        return res

    def tangent_norm(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the norm(s) of tangent vector(s) v of the tangent space(s) at PoincareBall point(s) x
        with respect to the Riemannian metric of the PoincareBall.

        Parameters
        ----------
        v : torch.Tensor
            vector(s) in the tangent space(s) of x
        x : torch.Tensor
            PoincareBall point(s)

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
        res = self._lambda(x) * v_norm
        return res

    def egrad2rgrad(self, grad: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the Riemannian gradient(s) at PoincareBall point(s) x from the Euclidean gradient(s).

        Parameters
        ----------
        grad : torch.Tensor
            Euclidean gradient(s)
        x : torch.Tensor
            PoincareBall point(s)

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
        res = grad / self._lambda(x) ** 2
        return res

    def proj(self, x: torch.Tensor):
        """
        Project point(s) x onto the clipped PoincareBall by restricting the Euclidean norm(s) to 1/c.sqrt()-self.max_enorm_eps.

        Parameters
        ----------
        x : torch.Tensor
            point(s)

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
        max_enorm = (1 / self.c.sqrt()).to(x.dtype) - eps
        assert max_enorm < (1 / self.c.sqrt()).to(x.dtype)
        x_norm = x.norm(p=2, dim=-1, keepdim=True)
        proj_x = (max_enorm / x_norm) * x
        condition = x_norm > max_enorm
        res = torch.where(condition, proj_x, x)
        return res

    def is_in_manifold(self, x: torch.Tensor) -> bool:
        """
        Check if point(s) x lie in the PoincareBall.

        Parameters
        ----------
        x : torch.Tensor
            PoincareBall point(s)

        Returns
        -------
        res : bool
            True if all points x lie in the PoincareBall, False otherwise
        """
        x2 = x.pow(2).sum(dim=-1, keepdim=True)
        r2 = torch.ones_like(x2) / self.c
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




#TODO faulty self ref etc.

# @torch.jit.script
# def dist_compiled(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, version: str) -> torch.Tensor:
#     """
#     Script compiled version of the dist method.
#     """
#     manifold = PoincareBall(c)
#     if version == "mobius":
#         # Mobius-dist
#         sqrt_c = manifold.c.sqrt()
#         dist_c = artanh(sqrt_c * manifold.addition(-x, y).norm(p=2, dim=-1, keepdim=True))
#         res = 2 * dist_c / sqrt_c
#     elif version == "mobius_symmetric":
#         # TODO check if this is algebraically allowed, numerically OK
#         # Symmetrized mobius-dist
#         sqrt_c = manifold.c.sqrt()
#         dist_c_1 = artanh(sqrt_c * manifold.addition(-x, y).norm(p=2, dim=-1, keepdim=True))
#         dist_c_2 = artanh(sqrt_c * manifold.addition(-y, x).norm(p=2, dim=-1, keepdim=True))
#         res = (dist_c_1 + dist_c_2) / sqrt_c
#     elif version == "metric_tensor":
#         # Metric-tensor-induced-dist
#         x_sqnorm = x.pow(2).sum(dim=-1, keepdim=True)
#         y_sqnorm = y.pow(2).sum(dim=-1, keepdim=True)
#         xy_diff_sqnorm = (x - y).pow(2).sum(dim=-1, keepdim=True)
#         res = 1 + 2 * manifold.c * xy_diff_sqnorm / ((1 - manifold.c * x_sqnorm) * (1 - manifold.c * y_sqnorm))
#         condition = res < 1 + manifold.min_enorm
#         res = torch.where(condition, torch.zeros_like(res), arcosh(res) / manifold.c.sqrt())
#     else:
#         raise ValueError(f"Unknown version: {version}")
#     return res

# @torch.jit.script
# def dist_0_compiled(self, x: torch.Tensor, c: torch.Tensor, version: str) -> torch.Tensor:
#     """
#     Script compiled version of the dist_0 method.
#     """
#     manifold = PoincareBall(c)
#     if version in ["mobius", "mobius_symmetric"]:
#         # Mobius-dist/Symmetrized mobius-dist
#         sqrt_c = manifold.c.sqrt()
#         dist_c = artanh(sqrt_c * x.norm(p=2, dim=-1, keepdim=True))
#         res = 2 * dist_c / sqrt_c
#     elif version == "metric_tensor":
#         # Metric-tensor-induced-dist
#         x_sqnorm = x.pow(2).sum(dim=-1, keepdim=True)
#         res = 1 + 2 * manifold.c * x_sqnorm / (1 - manifold.c * x_sqnorm)
#         condition = res < 1 + self.min_enorm
#         res = torch.where(condition, torch.zeros_like(res), arcosh(res) / manifold.c.sqrt())
#     else:
#         raise ValueError(f"Unknown version: {version}")
#     return res
