"""Core library for the replication of the numerical study in

    T.J. Boonen, W. Jiang,
    "Distributionally robust insurance under the Wasserstein distance,"
    Insurance: Mathematics and Economics 120 (2025) 61-78.

This module implements the GlueVaR distortion, the exponential benchmark loss,
and the solver for the worst-case survival function S_{P*} under either the
2-Wasserstein or the L2 distance constraint (Sections 4.2 / 4.3 of the paper).

Only numpy and scipy are used.  Equation numbers in the docstrings refer to the
paper / the replication SPEC.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# 1. GlueVaR distortion  g_{alpha,gamma}^{r1,r2}      (eq. 4.1, with r2 = 1)
# ---------------------------------------------------------------------------
@dataclass
class GlueVaR:
    """The GlueVaR distortion function g and its companions (eq. 4.1).

    A distortion g: [0,1] -> [0,1] is increasing with g(0)=0, g(1)=1.  The
    GlueVaR family is a piecewise-linear, concave distortion with two interior
    kinks at u = 1-gamma and u = 1-alpha.  With r2 = 1 (the value used in
    Section 4) it reads, writing a = 1-gamma and b = 1-alpha:

        g(u) = (r1/(1-gamma)) * u                       for 0     <= u < 1-gamma
             = r1 + ((1-r1)/(gamma-alpha))*(u-(1-gamma)) for 1-gamma <= u < 1-alpha
             = 1                                         for 1-alpha <= u <= 1

    Plain language: g is a "concave envelope" that distorts survival
    probabilities, putting extra weight on the tail.  The first slope
    r1/(1-gamma) is steep, the middle slope (1-r1)/(gamma-alpha) is shallower
    (concavity), and beyond u = 1-alpha the distortion saturates at 1.
    """

    r1: float
    gamma: float
    alpha: float
    r2: float = 1.0

    def g(self, u):
        """Distortion g(u), eq. 4.1.  Vectorized; correct on scalars/arrays.

        Implemented with np.where over the three linear pieces.  The kinks at
        u = 1-gamma and u = 1-alpha are handled by clipping each piece to its
        own validity range so that no piece blows up outside its segment.
        """
        u = np.asarray(u, dtype=float)
        a = 1.0 - self.gamma          # first kink, 1-gamma
        b = 1.0 - self.alpha          # second kink, 1-alpha
        slope1 = self.r1 / (1.0 - self.gamma)
        slope2 = (self.r2 - self.r1) / (self.gamma - self.alpha)

        piece1 = slope1 * u
        piece2 = self.r1 + slope2 * (u - a)
        piece3 = np.ones_like(u) * self.r2   # = 1 (with r2 = 1)

        out = np.where(u < a, piece1, np.where(u < b, piece2, piece3))
        # Clip into [0,1] to be numerically safe at the kinks / endpoints.
        out = np.clip(out, 0.0, 1.0)
        return out if out.ndim else float(out)

    def g_inv(self, v):
        """Inverse distortion g^{-1}(v) for v in [0,1].

        Because g is piecewise linear and strictly increasing on [0, 1-alpha]
        (then flat), its inverse on the range [0,1] is

            g^{-1}(v) = ((1-gamma)/r1) * v                       0   <= v <= r1
                      = (1-gamma) + ((gamma-alpha)/(1-r1))*(v-r1) r1  <  v <= 1

        so g^{-1}(1) = 1-alpha.  Values v in (r1, 1] map into (1-gamma, 1-alpha].
        """
        v = np.asarray(v, dtype=float)
        inv_slope1 = (1.0 - self.gamma) / self.r1
        inv_slope2 = (self.gamma - self.alpha) / (self.r2 - self.r1)

        piece1 = inv_slope1 * v
        piece2 = (1.0 - self.gamma) + inv_slope2 * (v - self.r1)

        out = np.where(v <= self.r1, piece1, piece2)
        out = np.clip(out, 0.0, 1.0)
        return out if out.ndim else float(out)

    def g_prime(self, u):
        """Right-derivative g'(u) (the slope of the active linear piece).

        g'(u) = r1/(1-gamma)        for 0      <= u < 1-gamma
              = (1-r1)/(gamma-alpha) for 1-gamma <= u < 1-alpha
              = 0                    for 1-alpha <= u <= 1

        Right-continuity: at a kink we take the slope of the piece starting
        there, which is what the `<` comparisons below produce.
        """
        u = np.asarray(u, dtype=float)
        a = 1.0 - self.gamma
        b = 1.0 - self.alpha
        slope1 = self.r1 / (1.0 - self.gamma)
        slope2 = (self.r2 - self.r1) / (self.gamma - self.alpha)

        out = np.where(u < a, slope1, np.where(u < b, slope2, 0.0))
        return out if out.ndim else float(out)


# ---------------------------------------------------------------------------
# 2. Exponential benchmark loss  (B = Q),  Section 4 / SPEC.md §1
# ---------------------------------------------------------------------------
@dataclass
class ExponentialLoss:
    """Exponential-type benchmark loss X with CDF F(x) = 1 - exp(-x/scale).

    The loss is truncated at the right endpoint M: F(x) = 1 for x > M.
    Throughout Section 4 the benchmark measure B equals the insurer's measure Q.
    """

    scale: float = 100.0
    M: float = 5000.0

    def S(self, x):
        """Survival function S_Q(x) = exp(-x/scale) on [0, M].

        Defined as 1 for x < 0 and 0 for x > M (the truncation point).
        """
        x = np.asarray(x, dtype=float)
        out = np.exp(-x / self.scale)
        out = np.where(x < 0.0, 1.0, out)
        out = np.where(x > self.M, 0.0, out)
        return out if out.ndim else float(out)

    def F_inv(self, u):
        """Quantile F_Q^{-1}(u) = -scale * ln(1-u) for u in [0,1)."""
        u = np.asarray(u, dtype=float)
        out = -self.scale * np.log1p(-u)   # -scale*ln(1-u), accurate near 0
        return out if out.ndim else float(out)

    def x_at_survival(self, s):
        """x such that S_Q(x) = s, i.e. F_Q^{-1}(1-s) = -scale*ln(s).

        The result is clipped to [0, M] so it can be used directly as an
        integration limit.
        """
        s = np.asarray(s, dtype=float)
        # avoid log(0): clip s away from 0 from below
        s_safe = np.clip(s, np.finfo(float).tiny, 1.0)
        out = -self.scale * np.log(s_safe)
        out = np.clip(out, 0.0, self.M)
        return out if out.ndim else float(out)

    def tail_integral(self, a):
        """Tail integral  int_a^M S_Q(y) dy = scale*(exp(-a/scale)-exp(-M/scale)).

        (Closed form of the exponential survival integral on [a, M].)
        """
        a = np.asarray(a, dtype=float)
        out = self.scale * (np.exp(-a / self.scale) - np.exp(-self.M / self.scale))
        return out if out.ndim else float(out)

    def J(self, s):
        """J(s) = int_0^M [s ^ S_Q(y)] dy   (eq. used in 3.8-3.9, SPEC §3a).

        For the exponential benchmark the closed form is

            J(s) = -scale * s * ln(s) + scale * (s - exp(-M/scale)),   J(0) = 0.

        Derivation: s ^ S_Q(y) equals s while S_Q(y) >= s (i.e. y <= x_s where
        x_s = -scale*ln s) and equals S_Q(y) afterwards.  Hence
            J(s) = s*x_s + int_{x_s}^M S_Q(y) dy,
        and substituting x_s and the tail integral gives the closed form above.
        This lets us replace the O(n^2) double integral int int S_P ^ S_B by a
        1-D quadrature int_0^M J(S_P(x)) dx.
        """
        s = np.asarray(s, dtype=float)
        out = (
            -self.scale * s * np.log(np.where(s > 0.0, s, 1.0))
            + self.scale * (s - np.exp(-self.M / self.scale))
        )
        # J(0) = 0 exactly (the s*ln s term -> 0, the scale*(0 - exp(-M/scale))
        # term is the residual tail mass which is tiny; set the limit explicitly).
        out = np.where(s <= 0.0, 0.0, out)
        return out if out.ndim else float(out)


# ---------------------------------------------------------------------------
# 3. Result container
# ---------------------------------------------------------------------------
@dataclass
class WorstCaseResult:
    """Full output of solve_worst_case (see SPEC §6)."""

    x: np.ndarray
    S_star: np.ndarray
    S_Q: np.ndarray
    net_price: np.ndarray
    x0: float
    x1: float
    beta: float
    binding: bool
    distance: float   # achieved W2 or L2 distance (NOT squared)
    mean: float       # int_0^M S_star dx


# ---------------------------------------------------------------------------
# 4. Solver
# ---------------------------------------------------------------------------
def _slack_form(loss: ExponentialLoss, glue: GlueVaR, theta, x, x0, x1):
    """The beta = 0 worst-case survival S~* (Theorem 3.1, eq. 3.12).

        S~*(x) = (t0 v S_Q(x)) * 1_{[0,x0)}            with t0 = g^{-1}(1) = 1-alpha
               + g^{-1}((1+theta) S_Q(x)) * 1_{A}      A  = [x0, x1)
               + S_Q(x) * 1_{Ac}                       Ac = [x1, M]

    Plain language: on [0,x0) the distorted price is capped, so the worst-case
    survival is pushed up to the plateau t0 = 1-alpha (or stays at S_Q if S_Q is
    already above it).  On the active region A it is the level whose distortion
    matches the price ceiling (1+theta)S_Q.  On Ac the loss is ceded and the
    worst case coincides with S_Q.
    """
    S_Q = loss.S(x)
    t0 = glue.g_inv(1.0)                      # = 1-alpha
    region0 = x < x0
    regionA = (x >= x0) & (x < x1)
    # Ac is the remainder.

    S = np.array(S_Q, dtype=float)            # default = S_Q (covers Ac)
    S[region0] = np.maximum(t0, S_Q[region0])
    # On A: g^{-1}((1+theta) S_Q), clipped because the argument can exceed 1.
    arg = np.clip((1.0 + theta) * S_Q[regionA], 0.0, 1.0)
    S[regionA] = glue.g_inv(arg)
    # Worst case must dominate the benchmark: S >= S_Q everywhere.
    S = np.maximum(S, S_Q)
    return S


def _S_hat(loss: ExponentialLoss, glue: GlueVaR, x, beta):
    """Auxiliary survival  Shat(x;beta)  (eq. 4.2).

    With the two interior horizontal shifts
        d2 = (1-r1)/(beta*(gamma-alpha)),   d1 = r1/(beta*(1-gamma)),
    and breakpoints
        x~1 = F_Q^{-1}(alpha),
        x~2 = min(F_Q^{-1}(alpha)+d2, M),
        x~3 = min(F_Q^{-1}(gamma)+d2, M),
        x~4 = min(F_Q^{-1}(gamma)+d1, M),
    the function is

        Shat = S_Q(x)        x in [0 , x~1)
             = 1-alpha       x in [x~1, x~2)
             = S_Q(x-d2)     x in [x~2, x~3)
             = 1-gamma       x in [x~3, x~4)
             = S_Q(x-d1)     x in [x~4, M].

    Plain language: starting from S_Q, two plateaus (at heights 1-alpha and
    1-gamma) are inserted and the survival curve is shifted to the right by d2
    then d1.  As beta -> 0 the shifts blow up and the plateaus widen to the
    eq. 3.12 form; as beta -> infinity the shifts vanish and Shat -> S_Q.
    """
    gamma, alpha, r1 = glue.gamma, glue.alpha, glue.r1
    d2 = (1.0 - r1) / (beta * (gamma - alpha))
    d1 = r1 / (beta * (1.0 - gamma))

    xt1 = loss.F_inv(alpha)
    xt2 = min(loss.F_inv(alpha) + d2, loss.M)
    xt3 = min(loss.F_inv(gamma) + d2, loss.M)
    xt4 = min(loss.F_inv(gamma) + d1, loss.M)

    S = loss.S(x)                              # default piece [0, x~1)
    seg2 = (x >= xt1) & (x < xt2)
    seg3 = (x >= xt2) & (x < xt3)
    seg4 = (x >= xt3) & (x < xt4)
    seg5 = x >= xt4

    out = np.array(S, dtype=float)
    out[seg2] = 1.0 - alpha
    out[seg3] = loss.S(x[seg3] - d2)
    out[seg4] = 1.0 - gamma
    out[seg5] = loss.S(x[seg5] - d1)
    return out


def _binding_form(loss: ExponentialLoss, glue: GlueVaR, theta, x, x0, x1, beta):
    """Binding worst-case survival S_{P*}(x;beta)  (eq. 3.14).

        S_{P*} = Shat(x;beta) * 1_{[0,x0)}
               + (Shat(x;beta) ^ g^{-1}((1+theta)S_Q(x))) * 1_A
               + S_Q(x) * 1_{Ac}.

    The element on A is capped by the price-matching level g^{-1}((1+theta)S_Q),
    exactly as in the slack form, while Shat replaces the plateau elsewhere.
    """
    S_Q = loss.S(x)
    S_hat = _S_hat(loss, glue, x, beta)

    region0 = x < x0
    regionA = (x >= x0) & (x < x1)

    S = np.array(S_Q, dtype=float)             # default = S_Q (Ac)
    S[region0] = S_hat[region0]
    arg = np.clip((1.0 + theta) * S_Q[regionA], 0.0, 1.0)
    cap = glue.g_inv(arg)
    S[regionA] = np.minimum(S_hat[regionA], cap)
    S = np.maximum(S, S_Q)                     # enforce S_P >= S_Q
    return S


def _w2_squared(loss: ExponentialLoss, x, S, int_x_SQ):
    """Achieved squared 2-Wasserstein distance of survival S (SPEC §3a).

        W2^2 = 2 * ( int x S dx  -  int J(S) dx  +  int x S_Q dx ).

    The cross term int int S_P ^ S_B is computed as int_0^M J(S(x)) dx, avoiding
    the O(n^2) double integral.  `int_x_SQ` = int x S_Q dx is passed in
    pre-computed since it does not depend on S.
    """
    int_x_S = np.trapz(x * S, x)
    int_J = np.trapz(loss.J(S), x)
    return 2.0 * (int_x_S - int_J + int_x_SQ)


def solve_worst_case(loss, glue, theta, radius, metric="wasserstein", n=200001):
    """Solve the distributionally robust problem (3.3) for the worst-case S_{P*}.

    Steps (SPEC §2-§3):
      1. Build a uniform grid on [0, M] with n points.
      2. Threshold x0 = F_Q^{-1}(theta/(1+theta))   (eq. 3.11; S_Q(x0)=1/(1+theta)).
      3. x1 from the closed form (Case 1, p.69) and cross-check against the
         largest numeric root of (1+theta)S_Q(x) - g(S_Q(x)) on [x0, M].
      4. Build the beta=0 slack form S~* (eq. 3.12).
      5. metric == "wasserstein":
           - if W2(S~*) <= radius -> slack, beta=0, S* = S~*;
           - else binding: root-find beta>0 with brentq so W2(beta)=radius,
             using the eq. 4.2 / 3.14 construction (W2 is decreasing in beta).
         metric == "L2":
           - constraint int (S_P-S_Q)^2 dx <= radius^2.  For all Section-4
             params S~* is slack; if it ever binds we raise NotImplementedError.
      6. net_price = (1+theta)S_Q - g(S*);  mean = int S* dx;
         distance = achieved W2 (or L2), NOT squared.
    """
    M = loss.M
    x = np.linspace(0.0, M, n)
    S_Q = loss.S(x)

    # --- x0 (eq. 3.11) -----------------------------------------------------
    x0 = loss.F_inv(theta / (1.0 + theta))

    # --- x1 closed form (Case 1, p.69) ------------------------------------
    gamma, alpha, r1 = glue.gamma, glue.alpha, glue.r1
    u_x1 = (alpha * r1 - alpha + theta * gamma - theta * alpha) / (
        (gamma - alpha) * (1.0 + theta) - (1.0 - r1)
    )
    x1 = loss.F_inv(u_x1)

    # Cross-check x1 against the largest numeric root of
    #   phi(x) = (1+theta) S_Q(x) - g(S_Q(x))   on [x0, M].
    def phi(xx):
        sq = loss.S(xx)
        return (1.0 + theta) * sq - glue.g(sq)

    xs = np.linspace(x0, M, 20001)
    vals = phi(xs)
    sign_change = np.where(np.diff(np.sign(vals)) != 0.0)[0]
    if len(sign_change) > 0:
        i = sign_change[-1]                      # largest root
        x1_num = brentq(phi, xs[i], xs[i + 1])
        assert abs(x1_num - x1) < 1e-3, (
            f"x1 closed form {x1} disagrees with numeric root {x1_num}"
        )

    # --- beta = 0 slack form S~* (eq. 3.12) -------------------------------
    S_tilde = _slack_form(loss, glue, theta, x, x0, x1)

    # Pre-compute int x S_Q dx (does not depend on the candidate survival).
    int_x_SQ = np.trapz(x * S_Q, x)

    beta = 0.0
    binding = False

    if metric == "wasserstein":
        eps = radius ** 2
        w2sq_tilde = _w2_squared(loss, x, S_tilde, int_x_SQ)

        if w2sq_tilde <= eps:
            # Constraint slack: the unconstrained beta=0 form is optimal.
            S_star = S_tilde
        else:
            # Constraint binds: find beta>0 with W2^2(beta) = eps.
            binding = True

            def gap(b):
                S_b = _binding_form(loss, glue, theta, x, x0, x1, b)
                return _w2_squared(loss, x, S_b, int_x_SQ) - eps

            # W2 is decreasing in beta. Bracket by scanning decades:
            # small beta -> large W2 (gap > 0); large beta -> S~S_Q (gap < 0).
            lo, hi = None, None
            for exp in np.linspace(-4, 4, 81):
                b = 10.0 ** exp
                gb = gap(b)
                if gb > 0:
                    lo = b                       # still too large -> increase beta
                elif gb < 0 and hi is None:
                    hi = b                       # first beta where W2 < radius
                    break
            if lo is None or hi is None:
                raise RuntimeError(
                    f"Failed to bracket beta (lo={lo}, hi={hi}); "
                    "widen the search decades."
                )
            beta = brentq(gap, lo, hi, xtol=1e-10, rtol=1e-12)
            S_star = _binding_form(loss, glue, theta, x, x0, x1, beta)

        distance = np.sqrt(max(_w2_squared(loss, x, S_star, int_x_SQ), 0.0))

    elif metric == "L2":
        eps = radius ** 2
        d2_sq_tilde = np.trapz((S_tilde - S_Q) ** 2, x)
        if d2_sq_tilde <= eps:
            S_star = S_tilde
            distance = np.sqrt(max(d2_sq_tilde, 0.0))
        else:
            # Per the spec all Section-4 L2 cases are slack.
            raise NotImplementedError(
                "L2 binding case (Theorem C.3) is not implemented; per the "
                "spec all Section-4 L2 parameter sets are slack."
            )
    else:
        raise ValueError(f"unknown metric {metric!r}; use 'wasserstein' or 'L2'")

    net_price = (1.0 + theta) * S_Q - glue.g(S_star)
    mean = np.trapz(S_star, x)

    return WorstCaseResult(
        x=x,
        S_star=S_star,
        S_Q=S_Q,
        net_price=net_price,
        x0=float(x0),
        x1=float(x1),
        beta=float(beta),
        binding=bool(binding),
        distance=float(distance),
        mean=float(mean),
    )


# ---------------------------------------------------------------------------
# 5. Self-tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- GlueVaR basic identities -------------------------------------------
    glue = GlueVaR(r1=0.6, gamma=0.7, alpha=0.05)
    a = 1.0 - glue.gamma     # 0.30
    b = 1.0 - glue.alpha     # 0.95

    assert abs(glue.g(0.0) - 0.0) < 1e-12
    assert abs(glue.g(a) - glue.r1) < 1e-12
    assert abs(glue.g(b) - 1.0) < 1e-12

    u_grid = np.linspace(1e-6, b, 1000)          # g invertible on [0, 1-alpha]
    assert np.allclose(glue.g_inv(glue.g(u_grid)), u_grid, atol=1e-9)
    print("GlueVaR identities: PASS")

    # --- Worst-case means (2-Wasserstein, r1=0.6, radius=10) ---------------
    loss = ExponentialLoss(scale=100.0, M=5000.0)
    radius = 10.0

    res_01 = solve_worst_case(loss, glue, theta=0.1, radius=radius)
    res_05 = solve_worst_case(loss, glue, theta=0.5, radius=radius)
    res_09 = solve_worst_case(loss, glue, theta=0.9, radius=radius)

    print(f"theta=0.1: mean = {res_01.mean:.4f}  binding = {res_01.binding}")
    print(f"theta=0.5: mean = {res_05.mean:.4f}  binding = {res_05.binding}")
    print(f"theta=0.9: mean = {res_09.mean:.4f}  binding = {res_09.binding}")

    ok_01 = (res_01.binding is False)
    ok_05 = abs(res_05.mean - 106.59) < 0.05
    ok_09 = abs(res_09.mean - 107.78) < 0.05

    print(f"theta=0.1 slack (binding is False): {'PASS' if ok_01 else 'FAIL'}")
    print(f"theta=0.5 mean ~= 106.59: {'PASS' if ok_05 else 'FAIL'}")
    print(f"theta=0.9 mean ~= 107.78: {'PASS' if ok_09 else 'FAIL'}")

    assert ok_01, "theta=0.1 should be slack"
    assert ok_05, f"theta=0.5 mean {res_05.mean} != 106.59"
    assert ok_09, f"theta=0.9 mean {res_09.mean} != 107.78"

    print("ALL SELF-TESTS PASS")
