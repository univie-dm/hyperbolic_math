import logging
import torch
import traceback

from typing import List
from .manifold import Manifold
from ..utils.math_utils import arcosh, artanh, tanh, arsinh, cosh, sinh

class PoincareBall(Manifold):
    """
    PoincareBall manifold class.
    Convention: x0^2 + x1^2 + ... + xd^2 < 1/c  with c > 0 and sectional curvature -c.
    """

    def __init__(
        self,
        c: torch.Tensor=torch.tensor([1.]),
        trainable_c: bool=False,
        dtype: torch.dtype=torch.float32,
    ):
        super().__init__(c, trainable_c)
        self.name = "PoincareBall"
        self.dtype = dtype
        if torch.finfo(c.dtype).eps < torch.finfo(dtype).eps:
            print(f"Warning: self.c.dtype is {c.dtype}, but self.dtype is {self.dtype}."
                  f"All manifold operations will be performed in precision {c.dtype}!")
            self.dtype = c.dtype

        if self.dtype == torch.float16:
            # TODO: Unverified clamps w.r.t unittests
            self.min_enorm = 1e-15
            self.max_enorm_eps = 5e-2
        elif self.dtype == torch.float32:
            # TODO: Unverified clamps w.r.t unittests
            self.min_enorm = 1e-15
            # HRL: Max-clamp with 4e-3 to reproduce their results (likely not the case anymore)
            self.max_enorm_eps = 4e-3
        elif self.dtype == torch.float64:
            # Numerical Stable Unittests for 1e-15 < max_enorm_eps < 1e-07
            self.min_enorm = 1e-15
            self.max_enorm_eps = 5e-15
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")

    def _2manifold_dtype(self, xs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Convert the list of tensor(s) xs to the PoincareBall's dtype.

        Parameters
        ----------
        xs : List[torch.Tensor]
            List of tensor(s)

        Returns
        -------
        res : List[torch.Tensor]
            The list of tensor(s) converted to the PoincareBall's dtype
        """
        res = []
        for x in xs:
            res.append(x.to(self.dtype))
        return res

    def _lambda(self, x: torch.Tensor, dim: int=-1) -> torch.Tensor:
        """
        Compute the conformal factor(s) at the PoincareBall point(s) x.

        Parameters
        ----------
        x : torch.Tensor
            PoincareBall point(s)
        dim : int
            Dimension along which to compute the conformal factor (default: -1)

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
        x, = self._2manifold_dtype([x])
        x2 = x.pow(2).sum(dim=dim, keepdim=True)
        denom = (1.0 - self.c * x2).clamp_min(2 * self.c.sqrt() * self.max_enorm_eps - self.c * self.max_enorm_eps ** 2)
        res = 2 / denom
        return res

    def _gyration(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, dim: int=-1) -> torch.Tensor:
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
        dim : int
            Dimension along which to compute the gyration (default: -1)

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
        x, y, z = self._2manifold_dtype([x, y, z])
        c2 = self.c**2
        x2 = x.pow(2).sum(dim=dim, keepdim=True)
        y2 = y.pow(2).sum(dim=dim, keepdim=True)
        xy = (x * y).sum(dim=dim, keepdim=True)
        xz = (x * z).sum(dim=dim, keepdim=True)
        yz = (y * z).sum(dim=dim, keepdim=True)
        a = -c2 * xz * y2 + self.c * yz + 2 * c2 * xy * yz
        b = -c2 * yz * x2 - self.c * xz
        num = 2 * (a * x + b * y)
        denom = 1 + 2 * self.c * xy + c2 * x2 * y2
        res = z + num / denom
        return res

    def addition(self, x: torch.Tensor, y: torch.Tensor, dim: int=-1, backproject: bool=True) -> torch.Tensor:
        """
        Add PoincareBall point(s) y to PoincareBall point(s) x using mobius gyrovector addition.
        Non-commutative and non-associative!

        Parameters
        ----------
        x : torch.Tensor
            PoincareBall point(s)
        y : torch.Tensor
            PoincareBall point(s)
        dim : int
            Dimension along which to compute the addition (default: -1)
        backproject : bool
            Whether to project results back to the PoincareBall (default: True)

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
        x, y = self._2manifold_dtype([x, y])
        x2 = x.pow(2).sum(dim=dim, keepdim=True)
        y2 = y.pow(2).sum(dim=dim, keepdim=True)
        xy = (x * y).sum(dim=dim, keepdim=True)
        num = (1 + 2 * self.c * xy + self.c * y2) * x + (1 - self.c * x2) * y
        denom = 1 + 2 * self.c * xy + self.c**2 * x2 * y2
        res = num / denom
        if backproject:
            res = self.proj(res, dim=dim)
        return res

    def scalar_mul(self, r: torch.Tensor, x: torch.Tensor, dim: int=-1, backproject: bool=True) -> torch.Tensor:
        """
        Multiply PoincareBall point(s) x with scalar(s) r.

        Parameters
        ----------
        r : torch.Tensor
            Scalar factor(s)
        x : torch.Tensor
            PoincareBall point(s)
        dim : int
            Dimension along which to compute the scalar multiplication (default: -1)
        backproject : bool
            Whether to project results back to the PoincareBall (default: True)

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
        """
        r, x = self._2manifold_dtype([r, x])
        x_norm = x.norm(p=2, dim=dim, keepdim=True)
        c_norm_prod = self.c.sqrt() * x_norm
        res = tanh(r * artanh(c_norm_prod)) / c_norm_prod * x
        if not torch.all(torch.isfinite(res)):
            logging.debug("scalar_mul: ZeroDivisionError")
            stack_trace = ''.join(traceback.format_stack(limit=-1))
            logging.debug(stack_trace)
            # Stable case
            x_norm = x.norm(p=2, dim=dim, keepdim=True).clamp_min(self.min_enorm)
            c_norm_prod = self.c.sqrt() * x_norm
            res = tanh(r * artanh(c_norm_prod)) / c_norm_prod * x
        if backproject:
            res = self.proj(res, dim=dim)
        return res

    def dist2hyperplane(self, x: torch.Tensor, a: torch.Tensor, p: torch.Tensor,
                        dim: int=-1, backproject: bool=True) -> torch.Tensor:
        """
        Computes the geodesic distance(s) of point(s) x to the hyperplane(s) given by a and p.
        [Geoopt implementation]

        Parameters
        ----------
        x : torch.Tensor (B, in_dim)
            PoincareBall point(s)
        a : torch.Tensor (out_dim, in_dim)
            Hyperplane tangent normal(s) in the tangent space at p
        p : torch.Tensor (out_dim, in_dim)
            Hyperplane translation(s)
        backproject : bool
            Whether to project self.addition() results back to the PoincareBall (default: True)

        Returns
        -------
        res : torch.Tensor (B, out_dim)
            The clipped product(s) of m and x defined as expmap_0(m * logmap_0(x))

        References
        ----------
        Ganea, Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic neural networks."
            Advances in neural information processing systems 31 (2018).
        """
        ### GEOOPT (k = -self.c, signed=True, scaled=True) ###
        # diff = _mobius_add(-p, x, k, dim=-1)
        # diff_norm2 = diff.pow(2).sum(dim=-1, keepdim=keepdim).clamp_min(1e-15)
        # sc_diff_a = (diff * a).sum(dim=-1, keepdim=keepdim)
        # a_norm = a.norm(dim=-1, keepdim=keepdim, p=2)
        # num = 2.0 * sc_diff_a
        # denom =  torch.abs((1 + k * diff_norm2) * a_norm) + 1e-15
        # distance = arsin_k(num / denom, k)  # geoopt uses arsinh
        # distance = distance * a_norm
        # return distance
        ######################################################
        x, a, p = self._2manifold_dtype([x, a, p])
        sqrt_c = self.c.sqrt()
        diff = self.addition(-p, x, dim=-1, backproject=backproject) # (B, 1, out_dim, in_dim)
        diff_norm2 = diff.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-15) # (B, 1, out_dim, 1)
        sc_diff_a = (diff * a).sum(dim=-1, keepdim=True) # (B, 1, out_dim, 1)
        a_norm = a.norm(dim=-1, keepdim=True, p=2) # (out_dim, 1)
        num = 2.0 * sc_diff_a # (B, 1, out_dim, 1)
        denom = torch.abs((1 - self.c * diff_norm2) * a_norm) + 1e-15 # (B, 1, out_dim, 1)
        signed_distance = arsinh(sqrt_c * num / denom) / sqrt_c # (B, 1, out_dim, 1)
        res = signed_distance * a_norm # (B, 1, out_dim, 1)
        return res

    def FC_forward(self, x: torch.Tensor, m: torch.Tensor, b: torch.Tensor,
                     backproject: bool=True) -> torch.Tensor:
        pass

    def MLR_forward(self, x: torch.Tensor, m: torch.Tensor, b: torch.Tensor,
                    version="MLR_forward", backproject: bool=True) -> torch.Tensor:
        pass

    def HRL_forward(self, x: torch.Tensor, a: torch.Tensor, p: torch.Tensor,
                    version="HRL_forward", backproject: bool=True) -> torch.Tensor:
        """
        Hyperbolic Reinforcement Learning multinomial linear regressions.
        Versions as implemented in the paper - uses dist2hyperplane from geoopt.
        x: (B, in_dim)
        a: (out_dim, in_dim)
        p: (out_dim, in_dim)
        """
        out_dim, in_dim = a.shape # out_dim, in_dim
        input_batch_dims = x.size()[:-1] # B
        input = x.view(-1, 1, in_dim) # (B, num_spaces=1, dimensions_per_space=in_dim)
        input_p = input.unsqueeze(-3) # (B, 1, num_spaces=1, dim_per_space=in_dim)
        if version == "HRL_forward":
            signed_distance = self.dist2hyperplane(input_p, a, p, backproject=backproject) # (B, 1, out_dim, 1)
            signed_distance = signed_distance #* self.logits_multiplier # logits_multiplier==1
        elif version == "HRL_forward_rs":
            conformal_factor = 1 - self.c * p.pow(2).sum(dim=-1) # (out_dim,) # not really the conformal factor
            signed_distance = self.dist2hyperplane(input_p, a*conformal_factor.unsqueeze(-1), p, backproject=backproject) # (B, 1, out_dim, 1)
            signed_distance = signed_distance * 2 / conformal_factor # (B, 1, out_dim, 1)
        else:
            raise ValueError(f"Unknown forward pass version: {version}")
        signed_distance = signed_distance.sum(-1) # (B, 1, out_dim)
        res = signed_distance.view(*input_batch_dims, out_dim) # (B, num_planes=out_dim)
        return res

    def HNN_MLR(self, x: torch.Tensor, a: torch.Tensor, p: torch.Tensor,
                backproject: bool=True) -> torch.Tensor:
        """
        Multinomial linear regression as described in Ganea's HNN paper.
        x: (B, in_dim)
        a: (out_dim, in_dim)
        p: (out_dim, in_dim)
        """
        sqrt_c = self.c.sqrt()
        sub = self.addition(-p.T.unsqueeze(0), x.unsqueeze(-1), dim=1, backproject=backproject) # (B, in_dim, out_dim)
        suba = (sub * a.T).sum(dim=1) # (B, out_dim)
        a_norm = a.norm(p=2, dim=-1, keepdim=True).clamp_min(self.min_enorm).T # (1, out_dim)
        signed_dist2hyp = arsinh(sqrt_c * self._lambda(sub, dim=1).squeeze(1) * suba / a_norm) / sqrt_c # (B, out_dim)
        res = self._lambda(p).T * a_norm * signed_dist2hyp # (B, out_dim)
        return res

    def hyperplane_forward_pp(self, x: torch.Tensor, z: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        Hyperplane forward as described in the HNN++ paper.
        """
        sqrt_c = self.c.sqrt()
        sqrt_c2r = 2 * sqrt_c * r.T # (out_dim, 1)
        z_norm = z.norm(p=2, dim=-1, keepdim=True).clamp_min(self.min_enorm) # (out_dim, 1)
        lambda_x = self._lambda(x) # (B, 1)
        z_unitx = (x.unsqueeze(-1) * (z / z_norm).T).sum(dim=1) # (B, out_dim)
        arsinh_arg = lambda_x * sqrt_c * z_unitx * cosh(sqrt_c2r) - (lambda_x-1) * sinh(sqrt_c2r)
        #arsinh_arg = (1-lambda_x) * sinh(sqrt_c2r) + sqrt_c * lambda_x * cosh(sqrt_c2r) * z_unitx # (B, out_dim)
        signed_dist2hyp = arsinh(arsinh_arg) / sqrt_c # (B, out_dim)
        res = 2 * z_norm.T * signed_dist2hyp # (B, out_dim)
        return res

    def HNNpp_forward(self, x: torch.Tensor, z: torch.Tensor, r: torch.Tensor,
                      version="HNNpp_MLR", backproject: bool=True) -> torch.Tensor:
        """
        Hyperbolic Neural Networks ++ forward pass & multinomial linear regression.
        x: (B, in_dim)
        z: (out_dim, in_dim)
        r: (output_dim, 1)
        """
        sqrt_c = self.c.sqrt()
        sqrt_c2r = 2 * sqrt_c * r.T # (out_dim, 1)
        z_norm = z.norm(p=2, dim=-1, keepdim=True).clamp_min(self.min_enorm) # (out_dim, 1)
        lambda_x = self._lambda(x) # (B, 1)
        z_unitx = (x.unsqueeze(-1) * (z / z_norm).T).sum(dim=1) # (B, out_dim)
        arsinh_arg = (1-lambda_x) * sinh(sqrt_c2r) + sqrt_c * lambda_x * cosh(sqrt_c2r) * z_unitx # (B, out_dim)
        signed_dist2hyp = arsinh(arsinh_arg) / sqrt_c # (B, out_dim)
        v = 2 * z_norm.T * signed_dist2hyp # (B, out_dim)
        if version == "HNNpp_MLR":
            res = v
        elif version == "HNNpp_linear":
            w = sinh(sqrt_c * v) / sqrt_c # (B, out_dim)
            w2 = w.pow(2).sum(dim=-1, keepdim=True) # (B, 1)
            denom = 1 + (1 + self.c * w2).sqrt() # (B, 1)
            res = w / denom # (B, out_dim)
            if backproject:
                res = self.proj(res)
        else:
            raise ValueError(f"Unknown forward pass version: {version}")
        return res

    def fully_linear_pp(self, x: torch.Tensor, m: torch.Tensor,
                        r: torch.Tensor, backproject: bool=True) -> torch.Tensor:
        """
        Hyperplane FC as described in the HNN++ paper.
        """
        sqrt_c = self.c.sqrt()
        v = self.hyperplane_forward_pp(x, m, r)
        w = sinh(sqrt_c * v) / sqrt_c
        w2 = w.pow(2).sum(dim=-1, keepdim=True)
        denom = 1 + (1 + self.c * w2).sqrt()
        res = w / denom

        if backproject:
            res = self.proj(res)
        return res

    def hyperplane_forward_pp_ours(self, x: torch.Tensor, m: torch.Tensor,
                                   p: torch.Tensor, backproject: bool=True) -> torch.Tensor:
        sqrt_c = self.c.sqrt()
        x = x.unsqueeze(1)
        p = p.unsqueeze(0)
        # Determine on which side of the hyperplanes the point(s) are
        sub = self.addition(-p, x, backproject=backproject)
        msub = (sub * m.unsqueeze(0)).sum(dim=-1)
        orientation = torch.sign(msub)
        # Get the lengths of the hyperplane normals
        m_norm = self.tangent_norm(m, p.squeeze(0)).T
        # Compute the geodesic distance(s) of the point(s) to the hyperplane
        xp = (x * p).sum(dim=-1)
        xp_norm_prod = (x.norm(p=2, dim=-1) * p.norm(p=2, dim=-1)).clamp(min=1e-16)
        cos_alpha = xp / xp_norm_prod
        ## Hyperbolic law of cosines & sines
        dist_x = self.dist_0(x.squeeze(1))
        dist_p = self.dist_0(p.squeeze(0)).T
        dist_px = self.dist(-p, x).squeeze(-1)

        # Test 1: Law of sines
        sin_alpha = (1 - cos_alpha ** 2).clamp(min=0).sqrt()
        sinh_cx = sinh(sqrt_c * dist_x)
        sinh_cp = sinh(sqrt_c * dist_p)
        sinh_cpx = sinh(sqrt_c * dist_px)
        sin_beta = (sin_alpha * sinh_cx) / sinh_cpx
        beta1 = torch.asin(sin_beta)

        # Test 2: Law of cosines
        cosh_cx = cosh(sqrt_c * dist_x)
        cosh_cp = cosh(sqrt_c * dist_p)
        cosh_cpx = cosh(sqrt_c * dist_px)
        cos_beta = (cosh_cp * cosh_cpx - cosh_cx) / (sinh_cp * sinh_cpx)
        beta2 = torch.acos(cos_beta)

        # Test 3: Inner product
        psub = (p * sub).sum(dim=-1)
        psub_norm_prod = (p.norm(p=2, dim=-1) * sub.norm(p=2, dim=-1)).clamp(min=1e-16)
        cos_beta = psub / psub_norm_prod
        beta3 = torch.acos(cos_beta)

        #print(f"beta1-beta2: {abs(beta1-beta2).max()}")
        #print(f"beta1-beta3: {abs(beta1-beta3).max()}")
        #print(f"beta2-beta3: {abs(beta2-beta3).max()}")

        # Shortest dist. to hyperplane
        #dist2hyp = arsinh(sin_beta * sinh_cpx) / sqrt_c
        ## Dist. of x from p
        dist2hyp = arcosh(cosh_cx * cosh_cp - sinh_cx * sinh_cp * cos_alpha) / sqrt_c

        res = orientation * dist2hyp * m_norm
        return res

    def dist(self, x: torch.Tensor, y: torch.Tensor, dim: int=-1,
             version: str="mobius", backproject: bool=True) -> torch.Tensor:
        """
        Compute the geodesic distance(s) between PoincareBall point(s) x and y.

        Parameters
        ----------
        x : torch.Tensor
            PoincareBall point(s)
        y : torch.Tensor
            PoincareBall point(s)
        dim : int
            Dimension along which to compute the geodesic distance (default: -1)
        version : str
            Version of the geodesic distance to compute (default: "mobius")
            ['mobius_direct': Symmetric Mobius distance that doesn't compute self.addition(),
             'mobius': Mobius distance,
             'metric_tensor': Metric-tensor induced distance]
        backproject : bool
            Whether to project results back to the PoincareBall (default: True)

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
        'mobius_direct' avoids the asymmetric mobius addition, but casually fails to comply
            with the tangent norm unit testing since it also uses the mobius addition.
        'mobius' is faster than 'metric_tensor', but not symmetric.
        'metric_tensor' is much faster than Mobius-dist, but unstable for boundary points.
        """
        x, y = self._2manifold_dtype([x, y])
        if version == "mobius_direct":
            # Symmetric Mobius distance that doesn't need self.addition()
            sqrt_c = self.c.sqrt()
            x2 = x.pow(2).sum(dim=dim, keepdim=True)
            y2 = y.pow(2).sum(dim=dim, keepdim=True)
            xy = (-x * y).sum(dim=dim, keepdim=True)
            num = (-x + y).pow(2).sum(dim=dim, keepdim=True)
            denom = 1 + 2 * self.c * xy + self.c**2 * x2 * y2
            xysum_norm = (num / denom).sqrt()
            dist_c = artanh(sqrt_c * xysum_norm)
            res = 2 * dist_c / sqrt_c
        elif version == "mobius":
            # Mobius distance
            sqrt_c = self.c.sqrt()
            dist_c = artanh(sqrt_c * self.addition(-x, y, dim=dim, backproject=backproject).norm(p=2, dim=dim, keepdim=True))
            res = 2 * dist_c / sqrt_c
        elif version == "metric_tensor":
            # Metric-tensor induced distance
            x_sqnorm = x.pow(2).sum(dim=dim, keepdim=True)
            y_sqnorm = y.pow(2).sum(dim=dim, keepdim=True)
            xy_diff_sqnorm = (x - y).pow(2).sum(dim=dim, keepdim=True)
            res = 1 + 2 * self.c * xy_diff_sqnorm / ((1 - self.c * x_sqnorm) * (1 - self.c * y_sqnorm))
            condition = res < 1 + self.min_enorm
            res = torch.where(condition, torch.zeros_like(res), arcosh(res) / self.c.sqrt())
        else:
            raise ValueError(f"Unknown version: {version}")
        return res

    def dist_0(self, x: torch.Tensor, dim: int=-1, version: str="mobius") -> torch.Tensor:
        """
        Compute the geodesic distance(s) of PoincareBall point(s) x from/to the PoincareBall origin.

        Parameters
        ----------
        x : torch.Tensor
            PoincareBall point(s)
        dim : int
            Dimension along which to compute the geodesic distance (default: -1)
        version : str
            Version of the geodesic distance to compute (default: "mobius")
            ['mobius_direct': Symmetric Mobius distance that doesn't compute self.addition(),
             'mobius': Mobius distance,
             'metric_tensor': Metric-tensor induced distance]

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
        'metric_tensor' is much faster than Mobius-dist, but unstable for boundary points.
        """
        x, = self._2manifold_dtype([x])
        if version in ["mobius_direct", "mobius"]:
            # (Direct) Mobius distance
            sqrt_c = self.c.sqrt()
            dist_c = artanh(sqrt_c * x.norm(p=2, dim=dim, keepdim=True))
            res = 2 * dist_c / sqrt_c
        elif version == "metric_tensor":
            # Metric-tensor induced distance
            x_sqnorm = x.pow(2).sum(dim=dim, keepdim=True)
            res = 1 + 2 * self.c * x_sqnorm / (1 - self.c * x_sqnorm)
            condition = res < 1 + self.min_enorm
            res = torch.where(condition, torch.zeros_like(res), arcosh(res) / self.c.sqrt())
        else:
            raise ValueError(f"Unknown version: {version}")
        return res

    def expmap(self, v: torch.Tensor, x: torch.Tensor, dim: int=-1, backproject: bool=True) -> torch.Tensor:
        """
        Map tangent vector(s) v at PoincareBall point(s) x to the clipped PoincareBall.
        [Exponential map]

        Parameters
        ----------
        v : torch.Tensor
            Vector(s) in the tangent space(s) of x
        x : torch.Tensor
            PoincareBall point(s)
        dim : int
            Dimension along which to compute the exponential map (default: -1)
        backproject : bool
            Whether to project results back to the PoincareBall (default: True)

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
        """
        v, x = self._2manifold_dtype([v, x])
        v_norm = v.norm(p=2, dim=dim, keepdim=True)
        c_norm_prod = self.c.sqrt() * v_norm
        second_term = tanh(c_norm_prod * self._lambda(x, dim=dim) / 2) / c_norm_prod * v
        if not torch.all(torch.isfinite(second_term)):
            logging.debug("expmap: ZeroDivisionError")
            stack_trace = ''.join(traceback.format_stack(limit=-1))
            logging.debug(stack_trace)

            # Stable case 1 - norm clamping
            # v_norm = v.norm(p=2, dim=dim, keepdim=True).clamp_min(self.min_enorm)
            # c_norm_prod = self.c.sqrt() * v_norm
            # second_term = tanh(c_norm_prod * self._lambda(x, dim=dim) / 2) / c_norm_prod * v

            # Stable case 2 - cnorm clamping
            v_norm = v.norm(p=2, dim=dim, keepdim=True)
            c_norm_prod = (self.c.sqrt() * v_norm).clamp_min(self.min_enorm)
            second_term = tanh(c_norm_prod * self._lambda(x, dim=dim) / 2) / c_norm_prod * v

            # Stable case 3 - denom clamping
            # v_norm = v.norm(p=2, dim=dim, keepdim=True)
            # c_norm_prod = self.c.sqrt() * v_norm
            # second_term = tanh(c_norm_prod * self._lambda(x, dim=dim) / 2) / (c_norm_prod).clamp_min(self.min_enorm) * v

        if backproject:
            second_term = self.proj(second_term, dim=dim)
        res = self.addition(x, second_term, dim=dim, backproject=backproject)
        return res

    def expmap_0(self, v: torch.Tensor, dim: int=-1, backproject: bool=True) -> torch.Tensor:
        """
        Map tangent vector(s) v at the PoincareBall origin to the clipped PoincareBall.
        [Exponential map]

        Parameters
        ----------
        v : torch.Tensor
            Vector(s) in the tangent space of the PoincareBall origin
        dim : int
            Dimension along which to compute the exponential map (default: -1)
        backproject : bool
            Whether to project results back to the PoincareBall (default: True)

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
        """
        v, = self._2manifold_dtype([v])
        v_norm = v.norm(p=2, dim=dim, keepdim=True)
        c_norm_prod = self.c.sqrt() * v_norm
        res = tanh(c_norm_prod) / c_norm_prod * v
        if not torch.all(torch.isfinite(res)):
            logging.debug("expmap_0: ZeroDivisionError")
            stack_trace = ''.join(traceback.format_stack(limit=-1))
            logging.debug(stack_trace)

            # Stable case 1 - norm clamping
            # v_norm = v.norm(p=2, dim=dim, keepdim=True).clamp_min(self.min_enorm)
            # c_norm_prod = self.c.sqrt() * v_norm

            ## Stable case 2 - cnorm clamping
            v_norm = v.norm(p=2, dim=dim, keepdim=True)
            c_norm_prod = (self.c.sqrt() * v_norm).clamp_min(self.min_enorm)

            res = tanh(c_norm_prod) / c_norm_prod * v
        if backproject:
            res = self.proj(res, dim=dim)
        return res

    def retraction(self, v: torch.Tensor, x: torch.Tensor, dim: int=-1, backproject: bool=True) -> torch.Tensor:
        """
        First-order approximation of the exponential map for vector(s) v at PoincareBall point(s) x.
        [Retraction map]

        Parameters
        ----------
        v : torch.Tensor
            Vector(s) in the tangent space(s) of x
        x : torch.Tensor
            PoincareBall point(s)
        dim : int
            Dimension along which to compute the backprojection of the retraction (default: -1)
        backproject : bool
            Whether to project results back to the PoincareBall (default: True)

        Returns
        -------
        res : torch.Tensor
            The resulting PoincareBall point(s) after approximate mapping v to the clipped PoincareBall

        References
        ----------
        Gary Bécigneul and Octavian Ganea. "Riemannian adaptive optimization methods."
            International Conference on Learning Representations (2019).
        """
        v, x = self._2manifold_dtype([v, x])
        res = x + v
        if backproject:
            res = self.proj(res, dim=dim)
        return res

    def logmap(self, y: torch.Tensor, x: torch.Tensor, dim: int=-1) -> torch.Tensor:
        """
        Map PoincareBall point(s) y to the tangent space(s) of PoincareBall point(s) x.
        [Logarithmic map]

        Parameters
        ----------
        y : torch.Tensor
            PoincareBall point(s)
        x : torch.Tensor
            PoincareBall point(s)
        dim : int
            Dimension along which to compute the logarithmic map (default: -1)

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
        y, x = self._2manifold_dtype([y, x])
        sub = self.addition(-x, y, dim=dim)
        sub_norm = sub.norm(p=2, dim=dim, keepdim=True)
        c_norm_prod = self.c.sqrt() * sub_norm
        res = 2 * artanh(c_norm_prod) / (c_norm_prod * self._lambda(x, dim=dim)) * sub
        if not torch.all(torch.isfinite(res)):
            logging.debug("logmap: ZeroDivisionError")
            stack_trace = ''.join(traceback.format_stack(limit=-1))
            logging.debug(stack_trace)

            # Stable case 1 - norm clamping
            # sub_norm = sub.norm(p=2, dim=dim, keepdim=True).clamp_min(self.min_enorm)
            # c_norm_prod = self.c.sqrt() * sub_norm

            # Stable case 2 - cnorm clamping
            sub_norm = sub.norm(p=2, dim=dim, keepdim=True)
            c_norm_prod = (self.c.sqrt() * sub_norm).clamp_min(self.min_enorm)

            res = 2 * artanh(c_norm_prod) / (c_norm_prod * self._lambda(x, dim=dim)) * sub
        return res

    def logmap_0(self, y: torch.Tensor, dim: int=-1) -> torch.Tensor:
        """
        Map PoincareBall point(s) y to the tangent space of the PoincareBall origin.
        [Logarithmic map]

        Parameters
        ----------
        y : torch.Tensor
            PoincareBall point(s)
        dim : int
            Dimension along which to compute the logarithmic map (default: -1)

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
        y, = self._2manifold_dtype([y])
        y_norm = y.norm(p=2, dim=dim, keepdim=True)
        c_norm_prod = self.c.sqrt() * y_norm
        res = artanh(c_norm_prod) / c_norm_prod * y
        if not torch.all(torch.isfinite(res)):
            logging.debug("logmap_0: ZeroDivisionError")
            stack_trace = ''.join(traceback.format_stack(limit=-1))
            logging.debug(stack_trace)

            # Stable case 1 - norm clamping
            # y_norm = y.norm(p=2, dim=dim, keepdim=True).clamp_min(self.min_enorm)
            # c_norm_prod = self.c.sqrt() * y_norm

            # Stable case 2 - cnorm clamping
            y_norm = y.norm(p=2, dim=dim, keepdim=True)
            c_norm_prod = (self.c.sqrt() * y_norm).clamp_min(self.min_enorm)

            res = artanh(c_norm_prod) / c_norm_prod * y
        return res

    def ptransp(self, v: torch.Tensor, x: torch.Tensor, y: torch.Tensor, dim: int=-1) -> torch.Tensor:
        """
        Parallel transport tangent vector(s) v from the tangent space(s) of PoincareBall point(s) x
        to the tangent space(s) of PoincareBall point(s) y.
        [Mobius version]

        Parameters
        ----------
        v : torch.Tensor
            Vector(s) in the tangent space(s) of x
        x : torch.Tensor
            PoincareBall point(s)
        y : torch.Tensor
            PoincareBall point(s)
        dim : int
            Dimension along which to compute the parallel transport (default: -1)

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
        v, x, y = self._2manifold_dtype([v, x, y])
        conformal_frac = self._lambda(x, dim=dim) / self._lambda(y, dim=dim)
        res = conformal_frac * self._gyration(y, -x, v, dim=dim)
        return res

    def ptransp_0(self, v: torch.Tensor, y: torch.Tensor, dim: int=-1) -> torch.Tensor:
        """
        Parallel transport tangent vector(s) v from the tangent space of the PoincareBall origin
        to the tangent space(s) of PoincareBall point(s) y.

        Parameters
        ----------
        v : torch.Tensor
            Vector(s) in the tangent space of the PoincareBall origin
        y : torch.Tensor
            PoincareBall point(s)
        dim : int
            Dimension along which to compute the parallel transport (default: -1)

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
        v, y = self._2manifold_dtype([v, y])
        conformal_frac = 2 / self._lambda(y, dim=dim)
        res = conformal_frac * v
        return res

    def tangent_inner(self, u: torch.Tensor, v: torch.Tensor, x: torch.Tensor, dim: int=-1) -> torch.Tensor:
        """
        Compute the inner product(s) between tangent vectors u and v of the tangent space(s) at PoincareBall point(s) x
        with respect to the Riemannian metric of the PoincareBall.

        Parameters
        ----------
        u : torch.Tensor
            Vector(s) in the tangent space(s) of x
        v : torch.Tensor
            Vector(s) in the tangent space(s) of x
        x : torch.Tensor
            PoincareBall point(s)
        dim : int
            Dimension along which to compute the tangent inner product (default: -1)

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
        u, v, x = self._2manifold_dtype([u, v, x])
        res = (u * v).sum(dim=dim, keepdim=True) * self._lambda(x, dim=dim) ** 2
        return res

    def tangent_norm(self, v: torch.Tensor, x: torch.Tensor, dim: int=-1) -> torch.Tensor:
        """
        Compute the norm(s) of tangent vector(s) v of the tangent space(s) at PoincareBall point(s) x
        with respect to the Riemannian metric of the PoincareBall.

        Parameters
        ----------
        v : torch.Tensor
            Vector(s) in the tangent space(s) of x
        x : torch.Tensor
            PoincareBall point(s)
        dim : int
            Dimension along which to compute the tangent norm (default: -1)

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
        v, x = self._2manifold_dtype([v, x])
        res = self._lambda(x, dim=dim) * v.norm(p=2, dim=dim, keepdim=True)
        return res

    def egrad2rgrad(self, grad: torch.Tensor, x: torch.Tensor, dim: int=-1) -> torch.Tensor:
        """
        Compute the Riemannian gradient(s) at PoincareBall point(s) x from the Euclidean gradient(s).

        Parameters
        ----------
        grad : torch.Tensor
            Euclidean gradient(s)
        x : torch.Tensor
            PoincareBall point(s)
        dim : int
            Dimension along which to compute the Riemannian gradient (default: -1)

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
        # Compute the conformal factor in the manifold's precision and cast it to the gradient's precision
        x, = self._2manifold_dtype([x])
        conformal_scale = (self._lambda(x, dim=dim) ** 2).to(grad.dtype)
        res = grad / conformal_scale
        return res

    def proj(self, x: torch.Tensor, dim: int=-1) -> torch.Tensor:
        """
        Project point(s) x onto the clipped PoincareBall by restricting
        the Euclidean norm(s) to 1/c.sqrt()-self.max_enorm_eps.

        Parameters
        ----------
        x : torch.Tensor
            Point(s)
        dim : int
            Dimension along which to compute the projection (default: -1)

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
        x, sqrt_c_recipr = self._2manifold_dtype([x, 1 / self.c.sqrt()])
        # Check if max_enorm can be numerically represented for the given c and eps
        max_enorm = sqrt_c_recipr - self.max_enorm_eps
        assert max_enorm < sqrt_c_recipr
        x_norm = x.norm(p=2, dim=dim, keepdim=True)
        proj_x = (max_enorm / x_norm) * x
        res = torch.where(x_norm > max_enorm, proj_x, x)
        return res

    def is_in_manifold(self, x: torch.Tensor, dim: int=-1) -> bool:
        """
        Check if point(s) x lie in the PoincareBall.

        Parameters
        ----------
        x : torch.Tensor
            PoincareBall point(s)
        dim : int
            Dimension along which to check if x lies in the PoincareBall (default: -1)

        Returns
        -------
        res : bool
            True if all points x lie in the PoincareBall, False otherwise
        """
        x, = self._2manifold_dtype([x])
        x2 = x.pow(2).sum(dim=dim, keepdim=True)
        r2 = torch.ones_like(x2) / self.c
        res = torch.all(x2 < r2)
        return res

    def is_in_tangent_space(self, v: torch.Tensor, x: torch.Tensor, dim: int=-1) -> bool:
        """
        Check if vector(s) v belong to the tangent space(s) at PoincareBall point(s) x.

        Parameters
        ----------
        v : torch.Tensor
            Vector(s)
        x : torch.Tensor
            PoincareBall point(s)
        dim : int
            Dimension along which to check if v belong to the tangent space (default: -1)
        Note: The dimension is not used in the PoincareBall, but it is included for consistency with other manifolds.

        Returns
        -------
        res : bool
            True if all vectors v belong to their tangent spaces, False otherwise
        """
        res = True
        return res


    ################
    ## Miscalleneous (might be useful) - Geoopt implementation available

    # def _proj_to_hyperboloid(self, x, c):
    #     K = 1.0 / c
    #     sqrtK = K**0.5
    #     sqnorm = torch.norm(x, p=2, dim=1, keepdim=True) ** 2
    #     return sqrtK * torch.cat([K + sqnorm, 2 * sqrtK * x], dim=1) / (K - sqnorm)

