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
        c: torch.Tensor = torch.tensor([1.]),
        trainable_c: bool = False,
        dtype: str = "float32",
    ):
        super().__init__(c, trainable_c)
        self.name = "PoincareBall"

        if dtype == "float16":
            self.dtype = torch.float16
            # TODO: Unverified clamps w.r.t unittests
            self.min_enorm = 1e-15
            self.max_enorm_eps = 5e-2
        elif dtype == "float32":
            self.dtype = torch.float32
            # TODO: Unverified clamps w.r.t unittests
            self.min_enorm = 1e-15
            # HRL: Max-clamp with 4e-3 to reproduce their results (likely not the case anymore)
            self.max_enorm_eps = 4e-3
        elif dtype == "float64":
            self.dtype = torch.float64
            # Numerical Stable Unittests for 1e-15 < max_enorm_eps < 1e-07
            self.min_enorm = 1e-15
            self.max_enorm_eps = 5e-15
        else:
            raise ValueError(f"Unsupported dtype: {dtype}. Supported dtypes are float16, float32, and float64.")

        if torch.finfo(c.dtype).eps < torch.finfo(self.dtype).eps:
            print(f"Warning: self.c.dtype is {c.dtype}, but self.dtype is {self.dtype}."
                  f"All manifold operations will be performed in precision {c.dtype}!")
            self.dtype = c.dtype

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

    def _lambda(self, x: torch.Tensor, axis: int=-1) -> torch.Tensor:
        """
        Compute the conformal factor(s) at the PoincareBall point(s) x.

        Parameters
        ----------
        x : torch.Tensor
            PoincareBall point(s)
        axis : int
            Axis along which to compute the conformal factor (default: -1)

        Returns
        -------
        res : torch.Tensor
            The conformal factor(s)

        References
        ----------
        Ganea Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic neural networks."
            Advances in neural information processing systems 31 (2018).

        Stability
        ---------
        Roughly bounded from above by 1/(c.sqrt()*self.max_enorm_eps)
        """
        x, = self._2manifold_dtype([x])
        x2 = x.pow(2).sum(dim=axis, keepdim=True)
        denom = (1.0 - self.c * x2).clamp_min(2 * self.c.sqrt() * self.max_enorm_eps - self.c * self.max_enorm_eps ** 2)
        res = 2 / denom
        return res

    def _gyration(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, axis: int=-1) -> torch.Tensor:
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
        axis : int
            Axis along which to compute the gyration (default: -1)

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
        x2 = x.pow(2).sum(dim=axis, keepdim=True)
        y2 = y.pow(2).sum(dim=axis, keepdim=True)
        xy = (x * y).sum(dim=axis, keepdim=True)
        xz = (x * z).sum(dim=axis, keepdim=True)
        yz = (y * z).sum(dim=axis, keepdim=True)
        a = -c2 * xz * y2 + self.c * yz + 2 * c2 * xy * yz
        b = -c2 * yz * x2 - self.c * xz
        num = 2 * (a * x + b * y)
        denom = 1 + 2 * self.c * xy + c2 * x2 * y2
        res = z + num / denom
        return res

    def addition(self, x: torch.Tensor, y: torch.Tensor, axis: int=-1, backproject: bool=True) -> torch.Tensor:
        """
        Add PoincareBall point(s) y to PoincareBall point(s) x using mobius gyrovector addition.
        Non-commutative and non-associative!

        Parameters
        ----------
        x : torch.Tensor
            PoincareBall point(s)
        y : torch.Tensor
            PoincareBall point(s)
        axis : int
            Axis along which to compute the addition (default: -1)
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
        x2 = x.pow(2).sum(dim=axis, keepdim=True)
        y2 = y.pow(2).sum(dim=axis, keepdim=True)
        xy = (x * y).sum(dim=axis, keepdim=True)
        num = (1 + 2 * self.c * xy + self.c * y2) * x + (1 - self.c * x2) * y
        denom = 1 + 2 * self.c * xy + self.c**2 * x2 * y2
        res = num / denom
        if backproject:
            res = self.proj(res, axis=axis)
        return res

    def scalar_mul(self, r: torch.Tensor, x: torch.Tensor, axis: int=-1, backproject: bool=True) -> torch.Tensor:
        """
        Multiply PoincareBall point(s) x with scalar(s) r.

        Parameters
        ----------
        r : torch.Tensor
            Scalar factor(s)
        x : torch.Tensor
            PoincareBall point(s)
        axis : int
            Axis along which to compute the scalar multiplication (default: -1)
        backproject : bool
            Whether to project results back to the PoincareBall (default: True)

        Returns
        -------
        res : torch.Tensor
            The clipped product(s) of r and x

        References
        ----------
        Ganea Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic neural networks."
            Advances in neural information processing systems 31 (2018).

        Stability
        ---------
        The PoincareBall multiplication converges towards the tangent space multiplication
        as the norm of vector(s) x approaches zero, since tanh(z) ~ artanh(z) ~ z for small z.
        """
        r, x = self._2manifold_dtype([r, x])
        x_norm = x.norm(p=2, dim=axis, keepdim=True)
        c_norm_prod = self.c.sqrt() * x_norm
        res = tanh(r * artanh(c_norm_prod)) / c_norm_prod * x
        if not torch.all(torch.isfinite(res)):
            logging.debug("scalar_mul: ZeroDivisionError")
            stack_trace = ''.join(traceback.format_stack(limit=-1))
            logging.debug(stack_trace)
            # Stable case
            x_norm = x.norm(p=2, dim=axis, keepdim=True).clamp_min(self.min_enorm)
            c_norm_prod = self.c.sqrt() * x_norm
            res = tanh(r * artanh(c_norm_prod)) / c_norm_prod * x
        if backproject:
            res = self.proj(res, axis=axis)
        return res

    def dist2hyperplane(self, x: torch.Tensor, a: torch.Tensor, p: torch.Tensor,
                        backproject: bool=True) -> torch.Tensor:
        """
        Computes the geodesic distance(s) of point(s) x to the hyperplane(s) defined by a and p.
        [Geoopt implementation]

        Parameters
        ----------
        x : torch.Tensor (B, in_dim)
            PoincareBall point(s)
        a : torch.Tensor (out_dim, in_dim)
            Hyperplane tangent normal(s) in the tangent space at p
        p : torch.Tensor (out_dim, in_dim)
            Hyperplane PoincareBall translation(s)
        backproject : bool
            Whether to project self.addition() results back to the PoincareBall (default: True)

        Returns
        -------
        res : torch.Tensor (B, out_dim)
            The scaled signed geodesic distance(s) of x to the hyperplane(s) defined by a and p.

        References
        ----------
        Max Kochurov, Rasul Karimov and Serge Kozlukov. "Geoopt: Riemannian Optimization in PyTorch."
            arXiv (2020).
        Ganea Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic neural networks."
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

    def HRL_forward(self, x: torch.Tensor, a: torch.Tensor, p: torch.Tensor,
                    version="HRL_forward", backproject: bool=True) -> torch.Tensor:
        """
        Hyperbolic Reinforcement Learning (scaled) multinomial linear regressions score functions.
        [Paper implementation with dist2hyperplane from geoopt]

        Parameters
        ----------
        x : torch.Tensor (B, in_dim)
            PoincareBall point(s)
        a : torch.Tensor (out_dim, in_dim)
            Hyperplane tangent normal(s)
        p : torch.Tensor (out_dim, in_dim)
            Hyperplane PoincareBall translation(s)
        version : str
            Version of the forward pass to compute (default: "HRL_forward")
            ['HRL_forward': scaled multinomial linear regression score function,
             'HRL_forward_rs': multinomial linear regression with parallel transported a]
        backproject : bool
            Whether to project self.addition() results back to the PoincareBall (default: True)

        Returns
        -------
        res : torch.Tensor (B, out_dim)
            The (scaled) multinomial linear regression score(s) of x with respect to the linear model(s) defined by a and p.

        References
        ----------
        Edoardo Cetin, Benjamin Chamberlain, Michael Bronstein and Jonathan J Hunt. "Hyperbolic deep reinforcement learning."
            arXiv (2022)
        Max Kochurov, Rasul Karimov and Serge Kozlukov. "Geoopt: Riemannian Optimization in PyTorch."
            arXiv (2020).
        """
        x, a, p = self._2manifold_dtype([x, a, p])
        out_dim, in_dim = a.shape # out_dim, in_dim
        input_batch_dims = x.size()[:-1] # B if x is of shape (B, in_dim)
        input = x.view(-1, 1, in_dim) # (B, num_spaces=1, dimensions_per_space=in_dim)
        input_p = input.unsqueeze(-3) # (B, 1, num_spaces=1, dim_per_space=in_dim)
        if version == "HRL_forward":
            # Compute the scaled signed distance to the hyperplane. Scale=Euclidean norm instead of the tangent norm of a
            signed_distance = self.dist2hyperplane(input_p, a, p, backproject=backproject) # (B, 1, out_dim, 1)
            signed_distance = signed_distance #* self.logits_multiplier # logits_multiplier==1
        elif version == "HRL_forward_rs":
            # Parallel transport a to the tangent space at p and return the signed distance to the hyperplane (no scaling)
            conformal_factor = 1 - self.c * p.pow(2).sum(dim=-1, keepdim=True) # (out_dim, 1) # not really the conformal factor
            signed_distance = self.dist2hyperplane(input_p, a*conformal_factor, p, backproject=backproject) # (B, 1, out_dim, 1)
            signed_distance = signed_distance * 2 / conformal_factor.view(1, 1, out_dim, 1) # (B, 1, out_dim, 1)
        else:
            raise ValueError(f"Unknown HRL forward pass version: {version}")
        signed_distance = signed_distance.sum(-1) # (B, 1, out_dim)
        res = signed_distance.view(*input_batch_dims, out_dim) # (B, num_planes=out_dim)
        return res

    def HNN_MLR(self, x: torch.Tensor, a: torch.Tensor, p: torch.Tensor,
                backproject: bool=True) -> torch.Tensor:
        """
        Hyperbolic Neural Networks multinomial linear regressions score function.

        Parameters
        ----------
        x : torch.Tensor (B, in_dim)
            PoincareBall point(s)
        a : torch.Tensor (out_dim, in_dim)
            Hyperplane tangent normal(s) in the tangent space at p
        p : torch.Tensor (out_dim, in_dim)
            Hyperplane PoincareBall translation(s)
        backproject : bool
            Whether to project self.addition() results back to the PoincareBall (default: True)

        Returns
        -------
        res : torch.Tensor (B, out_dim)
            The multinomial linear regression score(s) of x with respect to the linear model(s) defined by a and p.

        References
        ----------
        Ganea Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic neural networks."
            Advances in neural information processing systems 31 (2018).
        """
        x, a, p = self._2manifold_dtype([x, a, p])
        sqrt_c = self.c.sqrt()
        sub = self.addition(-p.T.unsqueeze(0), x.unsqueeze(-1), dim=1, backproject=backproject) # (B, in_dim, out_dim)
        suba = (sub * a.T).sum(dim=1, keepdim=True) # (B, 1, out_dim)
        a_norm = a.norm(p=2, dim=-1, keepdim=True).clamp_min(self.min_enorm).T # (1, out_dim)
        signed_dist2hyp = arsinh(sqrt_c * self._lambda(sub, dim=1) * suba / a_norm) / sqrt_c # (B, 1, out_dim)
        res = self._lambda(p, dim=-1).T * a_norm * signed_dist2hyp.squeeze(1) # (B, out_dim)
        return res

    def HNNpp_forward(self, x: torch.Tensor, z: torch.Tensor, r: torch.Tensor,
                      version="HNNpp_FC", backproject: bool=True) -> torch.Tensor:
        """
        Hyperbolic Neural Networks ++ fully connected and multinomial linear regressions score function.

        Parameters
        ----------
        x : torch.Tensor (B, in_dim)
            PoincareBall point(s)
        z : torch.Tensor (out_dim, in_dim)
            Hyperplane tangent normal(s) in the tangent space at the origin
        r : torch.Tensor (out_dim, 1)
            Hyperplane PoincareBall translation(s) defined by the scalar r and a
        version : str
            Version of the forward pass to compute (default: "HNNpp_FC")
            ['HNNpp_FC': fully connected forward pass,
             'HNNpp_MLR': multinomial linear regression forward pass]
        backproject : bool
            Whether to project the FC result back to the PoincareBall (default: True)

        Returns
        -------
        res : torch.Tensor (B, out_dim)
            The fully connected result or the multinomial linear regression score(s) of x with respect to the linear model(s) defined by z and r.

        References
        ----------
        Shimizu Ryohei, Yusuke Mukuta, and Tatsuya Harada. "Hyperbolic neural networks++."
            arXiv preprint arXiv:2006.08210 (2020).
        """
        sqrt_c = self.c.sqrt()
        sqrt_c2r = 2 * sqrt_c * r.T # (out_dim, 1)
        z_norm = z.norm(p=2, dim=-1, keepdim=True).clamp_min(self.min_enorm) # (out_dim, 1)
        lambda_x = self._lambda(x, dim=-1) # (B, 1)
        z_unitx = (x.unsqueeze(-1) * (z / z_norm).T).sum(dim=1) # (B, out_dim)
        arsinh_arg = (1-lambda_x) * sinh(sqrt_c2r) + sqrt_c * lambda_x * cosh(sqrt_c2r) * z_unitx # (B, out_dim)
        signed_dist2hyp = arsinh(arsinh_arg) / sqrt_c # (B, out_dim)
        v = 2 * z_norm.T * signed_dist2hyp # (B, out_dim)
        if version == "HNNpp_FC":
            w = sinh(sqrt_c * v) / sqrt_c # (B, out_dim)
            w2 = w.pow(2).sum(dim=-1, keepdim=True) # (B, 1)
            denom = 1 + (1 + self.c * w2).sqrt() # (B, 1)
            res = w / denom # (B, out_dim)
            if backproject:
                res = self.proj(res, dim=-1)
        elif version == "HNNpp_MLR":
            res = v
        else:
            raise ValueError(f"Unknown HNNpp forward pass version: {version}")
        return res

    def dist(self, x: torch.Tensor, y: torch.Tensor, axis: int=-1,
             version: str="mobius", backproject: bool=True) -> torch.Tensor:
        """
        Compute the geodesic distance(s) between PoincareBall point(s) x and y.

        Parameters
        ----------
        x : torch.Tensor
            PoincareBall point(s)
        y : torch.Tensor
            PoincareBall point(s)
        axis : int
            Axis along which to compute the geodesic distance (default: -1)
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
        Ganea Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic neural networks."
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
            x2 = x.pow(2).sum(dim=axis, keepdim=True)
            y2 = y.pow(2).sum(dim=axis, keepdim=True)
            xy = (-x * y).sum(dim=axis, keepdim=True)
            num = (-x + y).pow(2).sum(dim=axis, keepdim=True)
            denom = 1 + 2 * self.c * xy + self.c**2 * x2 * y2
            xysum_norm = (num / denom).sqrt()
            dist_c = artanh(sqrt_c * xysum_norm)
            res = 2 * dist_c / sqrt_c
        elif version == "mobius":
            # Mobius distance
            sqrt_c = self.c.sqrt()
            dist_c = artanh(sqrt_c * self.addition(-x, y, axis=axis, backproject=backproject).norm(p=2, dim=axis, keepdim=True))
            res = 2 * dist_c / sqrt_c
        elif version == "metric_tensor":
            # Metric-tensor induced distance
            x_sqnorm = x.pow(2).sum(dim=axis, keepdim=True)
            y_sqnorm = y.pow(2).sum(dim=axis, keepdim=True)
            xy_diff_sqnorm = (x - y).pow(2).sum(dim=axis, keepdim=True)
            res = 1 + 2 * self.c * xy_diff_sqnorm / ((1 - self.c * x_sqnorm) * (1 - self.c * y_sqnorm))
            condition = res < 1 + self.min_enorm
            res = torch.where(condition, torch.zeros_like(res), arcosh(res) / self.c.sqrt())
        else:
            raise ValueError(f"Unknown version: {version}")
        return res

    def dist_0(self, x: torch.Tensor, axis: int=-1, version: str="mobius") -> torch.Tensor:
        """
        Compute the geodesic distance(s) of PoincareBall point(s) x from/to the PoincareBall origin.

        Parameters
        ----------
        x : torch.Tensor
            PoincareBall point(s)
        axis : int
            Axis along which to compute the geodesic distance (default: -1)
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
        Ganea Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic neural networks."
            Advances in neural information processing systems 31 (2018).

        Stability
        ---------
        'metric_tensor' is much faster than Mobius-dist, but unstable for boundary points.
        """
        x, = self._2manifold_dtype([x])
        if version in ["mobius_direct", "mobius"]:
            # (Direct) Mobius distance
            sqrt_c = self.c.sqrt()
            dist_c = artanh(sqrt_c * x.norm(p=2, dim=axis, keepdim=True))
            res = 2 * dist_c / sqrt_c
        elif version == "metric_tensor":
            # Metric-tensor induced distance
            x_sqnorm = x.pow(2).sum(dim=axis, keepdim=True)
            res = 1 + 2 * self.c * x_sqnorm / (1 - self.c * x_sqnorm)
            condition = res < 1 + self.min_enorm
            res = torch.where(condition, torch.zeros_like(res), arcosh(res) / self.c.sqrt())
        else:
            raise ValueError(f"Unknown version: {version}")
        return res

    def expmap(self, v: torch.Tensor, x: torch.Tensor, axis: int=-1, backproject: bool=True) -> torch.Tensor:
        """
        Map tangent vector(s) v at PoincareBall point(s) x to the clipped PoincareBall.
        [Exponential map]

        Parameters
        ----------
        v : torch.Tensor
            Vector(s) in the tangent space(s) of x
        x : torch.Tensor
            PoincareBall point(s)
        axis : int
            Axis along which to compute the exponential map (default: -1)
        backproject : bool
            Whether to project results back to the PoincareBall (default: True)

        Returns
        -------
        res : torch.Tensor
            The resulting PoincareBall point(s) after mapping v to the clipped PoincareBall

        References
        ----------
        Ganea Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic neural networks."
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
        v_norm = v.norm(p=2, dim=axis, keepdim=True)
        c_norm_prod = self.c.sqrt() * v_norm
        second_term = tanh(c_norm_prod * self._lambda(x, axis=axis) / 2) / c_norm_prod * v
        if not torch.all(torch.isfinite(second_term)):
            logging.debug("expmap: ZeroDivisionError")
            stack_trace = ''.join(traceback.format_stack(limit=-1))
            logging.debug(stack_trace)

            # Stable case 1 - norm clamping
            # v_norm = v.norm(p=2, dim=axis, keepdim=True).clamp_min(self.min_enorm)
            # c_norm_prod = self.c.sqrt() * v_norm
            # second_term = tanh(c_norm_prod * self._lambda(x, axis=axis) / 2) / c_norm_prod * v

            # Stable case 2 - cnorm clamping
            v_norm = v.norm(p=2, dim=axis, keepdim=True)
            c_norm_prod = (self.c.sqrt() * v_norm).clamp_min(self.min_enorm)
            second_term = tanh(c_norm_prod * self._lambda(x, axis=axis) / 2) / c_norm_prod * v

            # Stable case 3 - denom clamping
            # v_norm = v.norm(p=2, dim=axis, keepdim=True)
            # c_norm_prod = self.c.sqrt() * v_norm
            # second_term = tanh(c_norm_prod * self._lambda(x, axis=axis) / 2) / (c_norm_prod).clamp_min(self.min_enorm) * v

        if backproject:
            second_term = self.proj(second_term, axis=axis)
        res = self.addition(x, second_term, axis=axis, backproject=backproject)
        return res

    def expmap_0(self, v: torch.Tensor, axis: int=-1, backproject: bool=True) -> torch.Tensor:
        """
        Map tangent vector(s) v at the PoincareBall origin to the clipped PoincareBall.
        [Exponential map]

        Parameters
        ----------
        v : torch.Tensor
            Vector(s) in the tangent space of the PoincareBall origin
        axis : int
            Axis along which to compute the exponential map (default: -1)
        backproject : bool
            Whether to project results back to the PoincareBall (default: True)

        Returns
        -------
        res : torch.Tensor
            The resulting PoincareBall point(s) after mapping v to the clipped PoincareBall

        References
        ----------
        Ganea Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic neural networks."
            Advances in neural information processing systems 31 (2018).

        Stability
        ---------
        TODO: check which clamping works better

        expmap_0 converges towards the identity map as the norm of vector(s) v approaches zero,
        since tanh(z) ~ z for small z.
        """
        v, = self._2manifold_dtype([v])
        v_norm = v.norm(p=2, dim=axis, keepdim=True)
        c_norm_prod = self.c.sqrt() * v_norm
        res = tanh(c_norm_prod) / c_norm_prod * v
        if not torch.all(torch.isfinite(res)):
            logging.debug("expmap_0: ZeroDivisionError")
            stack_trace = ''.join(traceback.format_stack(limit=-1))
            logging.debug(stack_trace)

            # Stable case 1 - norm clamping
            # v_norm = v.norm(p=2, dim=axis, keepdim=True).clamp_min(self.min_enorm)
            # c_norm_prod = self.c.sqrt() * v_norm

            ## Stable case 2 - cnorm clamping
            v_norm = v.norm(p=2, dim=axis, keepdim=True)
            c_norm_prod = (self.c.sqrt() * v_norm).clamp_min(self.min_enorm)

            res = tanh(c_norm_prod) / c_norm_prod * v
        if backproject:
            res = self.proj(res, axis=axis)
        return res

    def retraction(self, v: torch.Tensor, x: torch.Tensor, axis: int=-1, backproject: bool=True) -> torch.Tensor:
        """
        First-order approximation of the exponential map for vector(s) v at PoincareBall point(s) x.
        [Retraction map]

        Parameters
        ----------
        v : torch.Tensor
            Vector(s) in the tangent space(s) of x
        x : torch.Tensor
            PoincareBall point(s)
        axis : int
            Axis along which to compute the backprojection of the retraction (default: -1)
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
            res = self.proj(res, axis=axis)
        return res

    def logmap(self, y: torch.Tensor, x: torch.Tensor, axis: int=-1) -> torch.Tensor:
        """
        Map PoincareBall point(s) y to the tangent space(s) of PoincareBall point(s) x.
        [Logarithmic map]

        Parameters
        ----------
        y : torch.Tensor
            PoincareBall point(s)
        x : torch.Tensor
            PoincareBall point(s)
        axis : int
            Axis along which to compute the logarithmic map (default: -1)

        Returns
        -------
        res : torch.Tensor
            The resulting tangent vector(s) after mapping y to the tangent space(s) of x

        References
        ----------
        Ganea Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic neural networks."
            Advances in neural information processing systems 31 (2018).

        Stability
        ---------
        TODO: check which clamping works better

        logmap converges towards the identity map as the norm of vector(s) y-x approaches zero,
        since artanh(z) ~ z for small z.

        self._lambda() is roughly bounded from above by 1/(c.sqrt()*self.max_enorm_eps)
        """
        y, x = self._2manifold_dtype([y, x])
        sub = self.addition(-x, y, axis=axis)
        sub_norm = sub.norm(p=2, dim=axis, keepdim=True)
        c_norm_prod = self.c.sqrt() * sub_norm
        res = 2 * artanh(c_norm_prod) / (c_norm_prod * self._lambda(x, axis=axis)) * sub
        if not torch.all(torch.isfinite(res)):
            logging.debug("logmap: ZeroDivisionError")
            stack_trace = ''.join(traceback.format_stack(limit=-1))
            logging.debug(stack_trace)

            # Stable case 1 - norm clamping
            # sub_norm = sub.norm(p=2, dim=axis, keepdim=True).clamp_min(self.min_enorm)
            # c_norm_prod = self.c.sqrt() * sub_norm

            # Stable case 2 - cnorm clamping
            sub_norm = sub.norm(p=2, dim=axis, keepdim=True)
            c_norm_prod = (self.c.sqrt() * sub_norm).clamp_min(self.min_enorm)

            res = 2 * artanh(c_norm_prod) / (c_norm_prod * self._lambda(x, axis=axis)) * sub
        return res

    def logmap_0(self, y: torch.Tensor, axis: int=-1) -> torch.Tensor:
        """
        Map PoincareBall point(s) y to the tangent space of the PoincareBall origin.
        [Logarithmic map]

        Parameters
        ----------
        y : torch.Tensor
            PoincareBall point(s)
        axis : int
            Axis along which to compute the logarithmic map (default: -1)

        Returns
        -------
        res : torch.Tensor
            The resulting tangent vector(s) after mapping y to the tangent space of the origin

        References
        ----------
        Ganea Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic neural networks."
            Advances in neural information processing systems 31 (2018).

        Stability
        ---------
        TODO: check which clamping works better

        logmap_0 converges towards the identity map as the norm of vector(s) y approaches zero,
        since artanh(z) ~ z for small z.
        """
        y, = self._2manifold_dtype([y])
        y_norm = y.norm(p=2, dim=axis, keepdim=True)
        c_norm_prod = self.c.sqrt() * y_norm
        res = artanh(c_norm_prod) / c_norm_prod * y
        if not torch.all(torch.isfinite(res)):
            logging.debug("logmap_0: ZeroDivisionError")
            stack_trace = ''.join(traceback.format_stack(limit=-1))
            logging.debug(stack_trace)

            # Stable case 1 - norm clamping
            # y_norm = y.norm(p=2, dim=axis, keepdim=True).clamp_min(self.min_enorm)
            # c_norm_prod = self.c.sqrt() * y_norm

            # Stable case 2 - cnorm clamping
            y_norm = y.norm(p=2, dim=axis, keepdim=True)
            c_norm_prod = (self.c.sqrt() * y_norm).clamp_min(self.min_enorm)

            res = artanh(c_norm_prod) / c_norm_prod * y
        return res

    def ptransp(self, v: torch.Tensor, x: torch.Tensor, y: torch.Tensor, axis: int=-1) -> torch.Tensor:
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
        axis : int
            Axis along which to compute the parallel transport (default: -1)

        Returns
        -------
        res : torch.Tensor
            The parallel transported tangent vector(s)

        References
        ----------
        Ganea Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic neural networks."
            Advances in neural information processing systems 31 (2018).

        Stability
        ---------
        self._lambda() is roughly bounded from above by 1/(c.sqrt()*self.max_enorm_eps)
        """
        v, x, y = self._2manifold_dtype([v, x, y])
        conformal_frac = self._lambda(x, axis=axis) / self._lambda(y, axis=axis)
        res = conformal_frac * self._gyration(y, -x, v, axis=axis)
        return res

    def ptransp_0(self, v: torch.Tensor, y: torch.Tensor, axis: int=-1) -> torch.Tensor:
        """
        Parallel transport tangent vector(s) v from the tangent space of the PoincareBall origin
        to the tangent space(s) of PoincareBall point(s) y.

        Parameters
        ----------
        v : torch.Tensor
            Vector(s) in the tangent space of the PoincareBall origin
        y : torch.Tensor
            PoincareBall point(s)
        axis : int
            Axis along which to compute the parallel transport (default: -1)

        Returns
        -------
        res : torch.Tensor
            The parallel transported tangent vector(s)

        References
        ----------
        Ganea Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic neural networks."
            Advances in neural information processing systems 31 (2018).

        Stability
        ---------
        self._lambda() is roughly bounded from above by 1/(c.sqrt()*self.max_enorm_eps)
        """
        v, y = self._2manifold_dtype([v, y])
        conformal_frac = 2 / self._lambda(y, axis=axis)
        res = conformal_frac * v
        return res

    def tangent_inner(self, u: torch.Tensor, v: torch.Tensor, x: torch.Tensor, axis: int=-1) -> torch.Tensor:
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
        axis : int
            Axis along which to compute the tangent inner product (default: -1)

        Returns
        -------
        res : torch.Tensor
            The tangent inner product(s) of u and v

        References
        ----------
        Ganea Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic neural networks."
            Advances in neural information processing systems 31 (2018).

        Stability
        ---------
        self._lambda() is roughly bounded from above by 1/(c.sqrt()*self.max_enorm_eps)
        """
        u, v, x = self._2manifold_dtype([u, v, x])
        res = (u * v).sum(dim=axis, keepdim=True) * self._lambda(x, axis=axis) ** 2
        return res

    def tangent_norm(self, v: torch.Tensor, x: torch.Tensor, axis: int=-1) -> torch.Tensor:
        """
        Compute the norm(s) of tangent vector(s) v of the tangent space(s) at PoincareBall point(s) x
        with respect to the Riemannian metric of the PoincareBall.

        Parameters
        ----------
        v : torch.Tensor
            Vector(s) in the tangent space(s) of x
        x : torch.Tensor
            PoincareBall point(s)
        axis : int
            Axis along which to compute the tangent norm (default: -1)

        Returns
        -------
        res : torch.Tensor
            The tangent norm(s) of v

        References
        ----------
        Ganea Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic neural networks."
            Advances in neural information processing systems 31 (2018).

        Stability
        ---------
        self._lambda() is roughly bounded from above by 1/(c.sqrt()*self.max_enorm_eps)
        """
        v, x = self._2manifold_dtype([v, x])
        res = self._lambda(x, axis=axis) * v.norm(p=2, dim=axis, keepdim=True)
        return res

    def egrad2rgrad(self, grad: torch.Tensor, x: torch.Tensor, axis: int=-1) -> torch.Tensor:
        """
        Compute the Riemannian gradient(s) at PoincareBall point(s) x from the Euclidean gradient(s).

        Parameters
        ----------
        grad : torch.Tensor
            Euclidean gradient(s)
        x : torch.Tensor
            PoincareBall point(s)
        axis : int
            Axis along which to compute the Riemannian gradient (default: -1)

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
        conformal_scale = (self._lambda(x, axis=axis) ** 2).to(grad.dtype)
        res = grad / conformal_scale
        return res

    def proj(self, x: torch.Tensor, axis: int=-1) -> torch.Tensor:
        """
        Project point(s) x onto the clipped PoincareBall by restricting
        the Euclidean norm(s) to 1/c.sqrt()-self.max_enorm_eps.

        Parameters
        ----------
        x : torch.Tensor
            Point(s)
        axis : int
            Axis along which to compute the projection (default: -1)

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
        x_norm = x.norm(p=2, dim=axis, keepdim=True)
        proj_x = (max_enorm / x_norm) * x
        res = torch.where(x_norm > max_enorm, proj_x, x)
        return res

    def is_in_manifold(self, x: torch.Tensor, axis: int=-1) -> bool:
        """
        Check if point(s) x lie in the PoincareBall.

        Parameters
        ----------
        x : torch.Tensor
            PoincareBall point(s)
        axis : int
            Axis along which to check if x lies in the PoincareBall (default: -1)

        Returns
        -------
        res : bool
            True if all points x lie in the PoincareBall, False otherwise
        """
        x, = self._2manifold_dtype([x])
        x2 = x.pow(2).sum(dim=axis, keepdim=True)
        r2 = torch.ones_like(x2) / self.c
        res = torch.all(x2 < r2)
        return res

    def is_in_tangent_space(self, v: torch.Tensor, x: torch.Tensor, axis: int=-1) -> bool:
        """
        Check if vector(s) v belong to the tangent space(s) at PoincareBall point(s) x.

        Parameters
        ----------
        v : torch.Tensor
            Vector(s)
        x : torch.Tensor
            PoincareBall point(s)
        axis : int
            Axis along which to check if v belong to the tangent space (default: -1)
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

