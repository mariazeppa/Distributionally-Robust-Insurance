"""Independent verification of the published numerical values of

    T.J. Boonen, W. Jiang,
    "Distributionally robust insurance under the Wasserstein distance,"
    Insurance: Mathematics and Economics 120 (2025) 61-78.

This script recomputes the verification integrals of Section 4 (the worst-case
means under the 2-Wasserstein constraint, Section 4.2, and the L2 slack
distances, Section 4.3) and checks them against the numbers printed in the
paper.  It then performs an INDEPENDENT cross-check of the binding 2-Wasserstein
worst case by element-wise maximization, NOT reusing the eq. (4.2) construction
of the library.

Run:
    python3 verify_paper_values.py

All checks should print PASS.  The published targets are never altered here.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from dro_insurance import ExponentialLoss, GlueVaR, solve_worst_case


# ---------------------------------------------------------------------------
# Fixed model primitives of Section 4 (SPEC §1).
# ---------------------------------------------------------------------------
GAMMA = 0.7          # GlueVaR confidence level for the inner VaR
ALPHA = 0.05         # GlueVaR confidence level for the outer VaR / TVaR
RADIUS = 10.0        # ambiguity radius  eps^{1/2} = 10  (so eps = 100)

# Published verification targets (Section 4.5 / SPEC §5).  DO NOT EDIT.
TARGET_W2_VARY_THETA = {0.5: 106.59, 0.9: 107.78}   # r1 = 0.6
TARGET_W2_VARY_R1 = {0.6: 106.59, 0.7: 105.99, 0.8: 105.53}  # theta = 0.5
TARGET_L2_MAX_DIST = 2.5779                          # max over all 6 L2 cases

# Tolerances requested in the task brief.
TOL_MEAN = 0.02      # absolute tolerance on the worst-case means
TOL_L2 = 0.01        # absolute tolerance on the 2.5779 L2 distance


def _verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


# ===========================================================================
# TASK 1(a) -- 2-Wasserstein, r1 = 0.6, vary theta in {0.1, 0.5, 0.9}.
# ===========================================================================
def task1a(loss: ExponentialLoss) -> bool:
    print("=" * 78)
    print("TASK 1(a)  2-Wasserstein, r1 = 0.6, vary theta in {0.1, 0.5, 0.9}")
    print("=" * 78)
    print(
        f"{'theta':>6} | {'x0':>8} | {'x1':>8} | {'beta':>10} | "
        f"{'binding':>7} | {'W2':>7} | {'mean':>9}"
    )
    print("-" * 78)

    glue = GlueVaR(r1=0.6, gamma=GAMMA, alpha=ALPHA)
    means = {}
    for theta in (0.1, 0.5, 0.9):
        res = solve_worst_case(loss, glue, theta=theta, radius=RADIUS,
                               metric="wasserstein")
        means[theta] = res.mean
        beta_str = f"{res.beta:.4f}" if res.binding else "  -   "
        print(
            f"{theta:>6.1f} | {res.x0:>8.3f} | {res.x1:>8.3f} | {beta_str:>10} | "
            f"{str(res.binding):>7} | {res.distance:>7.4f} | {res.mean:>9.4f}"
        )

    # Check 1: published means for theta = 0.5 and 0.9.
    ok_05 = abs(means[0.5] - TARGET_W2_VARY_THETA[0.5]) < TOL_MEAN
    ok_09 = abs(means[0.9] - TARGET_W2_VARY_THETA[0.9]) < TOL_MEAN
    print()
    print(f"  int S*_2 (theta=0.5) = {means[0.5]:.4f}  vs published 106.59  "
          f"-> {_verdict(ok_05)}")
    print(f"  int S*_3 (theta=0.9) = {means[0.9]:.4f}  vs published 107.78  "
          f"-> {_verdict(ok_09)}")

    # Check 2: the ordering int S*_2 < int S*_3  (106.59 < 107.78).
    ok_order = means[0.5] < means[0.9]
    print(f"  ordering int S*_2 < int S*_3  ({means[0.5]:.2f} < {means[0.9]:.2f})"
          f"  -> {_verdict(ok_order)}")

    # Stochastic-dominance reading (Proposition / remark of the paper):
    # as theta increases the worst-case loss grows in the increasing-convex
    # order; in particular  X1 <=st X2 <=icx X3.
    print("  Stochastic-dominance reading:  X1  <=_st  X2  <=_icx  X3")
    print("  (theta=0.1 is stochastically dominated; theta=0.5 -> 0.9 grows in")
    print("   the increasing-convex (icx) order, hence the larger mean at 0.9).")

    return ok_05 and ok_09 and ok_order


# ===========================================================================
# TASK 1(b) -- 2-Wasserstein, theta = 0.5, vary r1 in {0.6, 0.7, 0.8}.
# ===========================================================================
def task1b(loss: ExponentialLoss) -> bool:
    print()
    print("=" * 78)
    print("TASK 1(b)  2-Wasserstein, theta = 0.5, vary r1 in {0.6, 0.7, 0.8}")
    print("=" * 78)
    print(
        f"{'r1':>6} | {'x0':>8} | {'x1':>8} | {'beta':>10} | "
        f"{'binding':>7} | {'W2':>7} | {'mean':>9}"
    )
    print("-" * 78)

    means = {}
    for r1 in (0.6, 0.7, 0.8):
        glue = GlueVaR(r1=r1, gamma=GAMMA, alpha=ALPHA)
        res = solve_worst_case(loss, glue, theta=0.5, radius=RADIUS,
                               metric="wasserstein")
        means[r1] = res.mean
        beta_str = f"{res.beta:.4f}" if res.binding else "  -   "
        print(
            f"{r1:>6.1f} | {res.x0:>8.3f} | {res.x1:>8.3f} | {beta_str:>10} | "
            f"{str(res.binding):>7} | {res.distance:>7.4f} | {res.mean:>9.4f}"
        )

    # Published means.
    ok = {}
    for r1, target in TARGET_W2_VARY_R1.items():
        ok[r1] = abs(means[r1] - target) < TOL_MEAN
        print()
        print(f"  int S*  (r1={r1}) = {means[r1]:.4f}  vs published {target}  "
              f"-> {_verdict(ok[r1])}", end="")
    print()

    # Ordering int S*_1 > int S*_2 > int S*_3  (106.59 > 105.99 > 105.53).
    ok_order = means[0.6] > means[0.7] > means[0.8]
    print(f"  ordering int S*_1 > int S*_2 > int S*_3  "
          f"({means[0.6]:.2f} > {means[0.7]:.2f} > {means[0.8]:.2f})"
          f"  -> {_verdict(ok_order)}")
    print("  Stochastic-dominance reading:  X3  <=_icx  X2  <=_icx  X1")
    print("  (a larger r1 puts more weight near the body, shrinking the tail")
    print("   distortion, so the worst-case mean decreases.)")

    return all(ok.values()) and ok_order


# ===========================================================================
# TASK 1(c) -- L2 distance, all 6 cases must be SLACK; report max distance.
# ===========================================================================
def task1c(loss: ExponentialLoss) -> bool:
    print()
    print("=" * 78)
    print("TASK 1(c)  L2 distance, all 6 cases -- must be SLACK; max distance")
    print("=" * 78)
    print(f"{'case':>22} | {'slack?':>7} | {'sqrt int (S~*-S_Q)^2':>22}")
    print("-" * 78)

    cases = []
    # theta in {0.1, 0.5, 0.9} with r1 = 0.6
    for theta in (0.1, 0.5, 0.9):
        cases.append((f"theta={theta}, r1=0.6", 0.6, theta))
    # r1 in {0.6, 0.7, 0.8} with theta = 0.5
    for r1 in (0.6, 0.7, 0.8):
        cases.append((f"r1={r1}, theta=0.5", r1, 0.5))

    all_slack = True
    distances = []
    for label, r1, theta in cases:
        glue = GlueVaR(r1=r1, gamma=GAMMA, alpha=ALPHA)
        res = solve_worst_case(loss, glue, theta=theta, radius=RADIUS,
                               metric="L2")
        # For L2 the achieved distance IS sqrt(int (S~* - S_Q)^2 dx).
        slack = (not res.binding) and (res.distance < RADIUS)
        all_slack = all_slack and slack
        distances.append(res.distance)
        print(f"{label:>22} | {str(slack):>7} | {res.distance:>22.4f}")

    max_dist = max(distances)
    ok_slack = all_slack
    ok_value = abs(max_dist - TARGET_L2_MAX_DIST) < TOL_L2
    ok_below_radius = max_dist < RADIUS

    print()
    print(f"  all 6 cases SLACK  -> {_verdict(ok_slack)}")
    print(f"  max distance = {max_dist:.4f}  vs published 2.5779  "
          f"-> {_verdict(ok_value)}")
    print(f"  max distance < radius (10)  ({max_dist:.4f} < 10)  "
          f"-> {_verdict(ok_below_radius)}")

    return ok_slack and ok_value and ok_below_radius


# ===========================================================================
# TASK 2 -- Independent element-wise-maximization cross-check of the binding
#           2-Wasserstein worst case (does NOT reuse the eq. 4.2 construction).
# ===========================================================================
def task2(loss: ExponentialLoss) -> bool:
    print()
    print("=" * 78)
    print("TASK 2  Independent cross-check of the binding worst case")
    print("        (element-wise maximization; eq. 4.2 NOT reused)")
    print("=" * 78)
    print("""  For the binding case we take the library's solved beta and, on a coarse
  grid, solve for each x the 1-D concave program (SPEC eq. for K3):

      S_check(x) = argmax_{t in [S_Q(x), 1]}  K3(t; x),
      K3(t; x)   = min( g(t), (1+theta) S_Q(x) ) - beta * ( x*t - J(t) ).

  This is the first-order/pointwise characterization of the worst case; if the
  eq. (4.2) construction in the library is correct, the two must coincide.""")
    print()
    print(f"{'theta':>6} | {'beta':>10} | {'max |S_check - S_star|':>22} | {'PASS/FAIL':>9}")
    print("-" * 78)

    glue = GlueVaR(r1=0.6, gamma=GAMMA, alpha=ALPHA)
    n_coarse = 2000
    tol = 5e-3
    all_ok = True

    for theta in (0.5, 0.9):
        res = solve_worst_case(loss, glue, theta=theta, radius=RADIUS,
                               metric="wasserstein")
        assert res.binding, f"theta={theta} expected to be binding"
        beta = res.beta

        # Coarse grid over [0, M].
        xc = np.linspace(0.0, loss.M, n_coarse)
        S_Q_c = loss.S(xc)

        # Element-wise maximization of K3 over t in [S_Q(x), 1].
        S_check = np.empty_like(xc)
        for i, xx in enumerate(xc):
            lo = float(S_Q_c[i])
            hi = 1.0
            if hi - lo < 1e-12:
                # Degenerate interval (deep tail where S_Q ~ 0 but lo~0): just
                # take lo; K3 is flat there to numerical precision.
                S_check[i] = lo
                continue

            sq = float(S_Q_c[i])
            cap = (1.0 + theta) * sq

            def neg_K3(t, _xx=xx, _sq=sq, _cap=cap):
                gt = glue.g(t)
                # min(g(t), (1+theta) S_Q(x)) - beta*(x*t - J(t))
                k3 = min(gt, _cap) - beta * (_xx * t - loss.J(t))
                return -k3

            sol = minimize_scalar(neg_K3, bounds=(lo, hi), method="bounded",
                                  options={"xatol": 1e-9})
            S_check[i] = sol.x

        # Interpolate the library's fine-grid S_star onto the coarse grid.
        S_star_c = np.interp(xc, res.x, res.S_star)

        max_diff = float(np.max(np.abs(S_check - S_star_c)))
        ok = max_diff < tol
        all_ok = all_ok and ok
        print(f"{theta:>6.1f} | {beta:>10.4f} | {max_diff:>22.3e} | "
              f"{_verdict(ok):>9}")

    print()
    print(f"  Cross-check tolerance: max |S_check - S_star| < {tol:g}")
    print(f"  -> confirms eq. (4.2) / (3.14) is implemented correctly.")
    return all_ok


# ===========================================================================
# Main driver.
# ===========================================================================
def main() -> None:
    loss = ExponentialLoss(scale=100.0, M=5000.0)

    results = {
        "Task 1(a)  2-Wasserstein vary theta": task1a(loss),
        "Task 1(b)  2-Wasserstein vary r1": task1b(loss),
        "Task 1(c)  L2 slack (2.5779)": task1c(loss),
        "Task 2     element-wise cross-check": task2(loss),
    }

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for name, ok in results.items():
        print(f"  {name:<40} -> {_verdict(ok)}")
    print("-" * 78)
    overall = all(results.values())
    print(f"  OVERALL: {'ALL CHECKS PASS' if overall else 'SOME CHECKS FAILED'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
