"""Reproduce Figures 2-5 of

    T.J. Boonen, W. Jiang,
    "Distributionally robust insurance under the Wasserstein distance,"
    Insurance: Mathematics and Economics 120 (2025) 61-78,

Section 4 numerical study.  Each figure is a 1-row x 2-column panel:

    LEFT  : the worst-case survival functions S_{P*}(x) for the three cases,
            overlaid, plus the benchmark S_Q(x) as a thin reference curve.
    RIGHT : the net price of marginal coverage
                NP(x) = (1+theta) S_Q(x) - g(S_{P*}(x))
            for the same three cases.

The figures are:

    figure_2.png : 2-Wasserstein, r1=0.6, theta in {0.1, 0.5, 0.9}.
    figure_3.png : 2-Wasserstein, theta=0.5, r1 in {0.6, 0.7, 0.8}.
    figure_4.png : L2,           r1=0.6, theta in {0.1, 0.5, 0.9}.
    figure_5.png : L2,           theta=0.5, r1 in {0.6, 0.7, 0.8}.

Throughout: radius = eps^{1/2} = 10, alpha = 0.05, gamma = 0.7.  The worst-case
survival functions and net prices are computed over the full support [0, M] by
the core library `dro_insurance`; only the *viewing window* is restricted to
x in [0, 600] (the curves rejoin S_Q in the tail), so we merely slice the
returned arrays for plotting and never re-integrate.
"""

from __future__ import annotations

import os

import matplotlib

# Non-interactive backend: we only write PNG files, never open a window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
import numpy as np  # noqa: E402

from dro_insurance import ExponentialLoss, GlueVaR, solve_worst_case  # noqa: E402


# ---------------------------------------------------------------------------
# Shared configuration (SPEC §4)
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))

RADIUS = 10.0          # = eps^{1/2}; the Wasserstein / L2 ball radius
ALPHA = 0.05
GAMMA = 0.7

X_VIEW_MAX = 400.0     # right edge of the plotting window (curves meet S_Q here)

# Distinct colours / line styles for the three overlaid cases.
COLORS = ["tab:blue", "tab:orange", "tab:green"]
LINESTYLES = ["-", "--", "-."]

LOSS = ExponentialLoss(scale=100.0, M=5000.0)


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------
def make_figure(cases, metric, vary_label, fig_title, out_path):
    """Build one 1x2 figure and save it as a PNG.

    Parameters
    ----------
    cases : list of (label, WorstCaseResult)
        Up to three solved cases, already computed with `solve_worst_case`.
        `label` is the legend text (e.g. "theta=0.5").
    metric : str
        "2-Wasserstein" or "L2", used only in the panel titles.
    vary_label : str
        Human-readable description of the swept parameter (suptitle).
    fig_title : str
        Short tag like "Fig. 2" used in the panel titles.
    out_path : str
        Absolute path of the PNG to write.

    The left panel overlays the three worst-case survival functions S_{P*}(x)
    plus the benchmark S_Q(x); the right panel overlays the three net prices
    NP(x).  Both x-axes are restricted to [0, X_VIEW_MAX] by slicing the arrays
    returned by the solver (no re-integration).
    """
    fig, (ax_surv, ax_np) = plt.subplots(1, 2, figsize=(12, 5))

    # All cases share the same x grid, so derive the viewing mask once.
    x0_grid = cases[0][1].x
    view = x0_grid <= X_VIEW_MAX

    # Benchmark survival S_Q as a thin reference curve (left panel only).
    S_Q = cases[0][1].S_Q
    ax_surv.plot(
        x0_grid[view], S_Q[view],
        color="black", linewidth=0.8, linestyle=":", label="S_Q (benchmark)",
    )

    for (label, res), color, ls in zip(cases, COLORS, LINESTYLES):
        x = res.x[view]

        # Augment the legend label with slack/binding status and the achieved
        # distance, which is informative for the thesis discussion.
        status = "binding" if res.binding else "slack"
        legend = f"{label} ({status}, dist={res.distance:.2f})"

        ax_surv.plot(x, res.S_star[view], color=color, linestyle=ls,
                     linewidth=1.6, label=legend)
        ax_np.plot(x, res.net_price[view], color=color, linestyle=ls,
                   linewidth=1.6, label=legend)

    # --- left panel: worst-case survival functions -------------------------
    ax_surv.set_title(f"({metric}): worst-case survival functions")
    ax_surv.set_xlabel("loss x")
    ax_surv.set_ylabel("survival probability")
    ax_surv.set_xlim(0.0, X_VIEW_MAX)
    ax_surv.set_ylim(0.0, 1.0)
    ax_surv.legend(loc="upper right", fontsize=8)
    ax_surv.grid(True, alpha=0.3)

    # --- right panel: net price of marginal coverage -----------------------
    ax_np.set_title(f"({metric}): net price")
    ax_np.set_xlabel("loss x")
    ax_np.set_ylabel("net price of marginal coverage")
    ax_np.set_xlim(0.0, X_VIEW_MAX)
    ax_np.axhline(0.0, color="grey", linewidth=0.6)   # reference: NP = 0
    ax_np.legend(loc="upper right", fontsize=8)
    ax_np.grid(True, alpha=0.3)

    fig.suptitle(vary_label, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def print_summary(fig_name, cases):
    """Print a one-line summary per case for eyeball verification."""
    print(f"\n{fig_name}:")
    for label, res in cases:
        print(
            f"  {label:>10s} | binding={'T' if res.binding else 'F'} "
            f"| distance={res.distance:8.4f} | mean=int S* = {res.mean:8.4f}"
        )


# ---------------------------------------------------------------------------
# Figure 2 — 2-Wasserstein, vary theta (r1 = 0.6)
# ---------------------------------------------------------------------------
# Shows how the worst-case survival function and net price respond to the
# safety loading theta in {0.1, 0.5, 0.9} with the GlueVaR parameter r1 fixed at
# 0.6.  theta=0.1 is slack (the Wasserstein ball does not bite); theta=0.5 and
# theta=0.9 bind.  A larger theta lifts the worst-case survival in the tail
# (X1 <=st X2 <=icx X3), with published means 106.59 (theta=0.5) and 107.78
# (theta=0.9).  Matches Figure 2 of the paper.
# ---------------------------------------------------------------------------
def figure_2():
    glue = GlueVaR(r1=0.6, gamma=GAMMA, alpha=ALPHA)
    cases = [
        (f"theta={t}", solve_worst_case(LOSS, glue, theta=t, radius=RADIUS,
                                        metric="wasserstein"))
        for t in (0.1, 0.5, 0.9)
    ]
    out = os.path.join(HERE, "figure_2.png")
    make_figure(
        cases,
        metric="2-Wasserstein",
        vary_label="2-Wasserstein, varying safety loading "
                   "theta (r1=0.6, alpha=0.05, gamma=0.7, radius=10)",
        fig_title="Fig. 2",
        out_path=out,
    )
    print_summary("Figure 2 (2-Wasserstein, vary theta, r1=0.6)", cases)
    return out


# ---------------------------------------------------------------------------
# Figure 3 — 2-Wasserstein, vary r1 (theta = 0.5)
# ---------------------------------------------------------------------------
# Shows the effect of the GlueVaR parameter r1 in {0.6, 0.7, 0.8} on the
# worst-case survival / net price with the safety loading fixed at theta=0.5.
# All three cases bind.  A larger r1 (less tail-weighting) lowers the worst-case
# survival, so X3 <=icx X2 <=icx X1, with published means 106.59 (r1=0.6),
# 105.99 (r1=0.7), 105.53 (r1=0.8).  Matches Figure 3 of the paper.
# ---------------------------------------------------------------------------
def figure_3():
    cases = []
    for r1 in (0.6, 0.7, 0.8):
        glue = GlueVaR(r1=r1, gamma=GAMMA, alpha=ALPHA)
        res = solve_worst_case(LOSS, glue, theta=0.5, radius=RADIUS,
                               metric="wasserstein")
        cases.append((f"r1={r1}", res))
    out = os.path.join(HERE, "figure_3.png")
    make_figure(
        cases,
        metric="2-Wasserstein",
        vary_label="2-Wasserstein, varying GlueVaR parameter "
                   "r1 (theta=0.5, alpha=0.05, gamma=0.7, radius=10)",
        fig_title="Fig. 3",
        out_path=out,
    )
    print_summary("Figure 3 (2-Wasserstein, vary r1, theta=0.5)", cases)
    return out


# ---------------------------------------------------------------------------
# Figure 4 — L2, vary theta (r1 = 0.6)
# ---------------------------------------------------------------------------
# The L2 analogue of Figure 2: r1=0.6, theta in {0.1, 0.5, 0.9}.  Under the L2
# distance the ball is slack for every Section-4 parameter set, so the worst
# case is the beta=0 form S~* (eq. 3.12) in all three cases.  Because the L2
# constraint is not binding, the worst-case survival sits HIGHER (riskier) than
# under the corresponding 2-Wasserstein cases of Figure 2.  Matches Figure 4.
# ---------------------------------------------------------------------------
def figure_4():
    glue = GlueVaR(r1=0.6, gamma=GAMMA, alpha=ALPHA)
    cases = [
        (f"theta={t}", solve_worst_case(LOSS, glue, theta=t, radius=RADIUS,
                                        metric="L2"))
        for t in (0.1, 0.5, 0.9)
    ]
    out = os.path.join(HERE, "figure_4.png")
    make_figure(
        cases,
        metric="L2",
        vary_label="L2 distance, varying safety loading "
                   "theta (r1=0.6, alpha=0.05, gamma=0.7, radius=10)",
        fig_title="Fig. 4",
        out_path=out,
    )
    print_summary("Figure 4 (L2, vary theta, r1=0.6)", cases)
    return out


# ---------------------------------------------------------------------------
# Figure 5 — L2, vary r1 (theta = 0.5)
# ---------------------------------------------------------------------------
# The L2 analogue of Figure 3: theta=0.5, r1 in {0.6, 0.7, 0.8}.  Again all
# cases are slack, so S_{P*} = S~* (eq. 3.12).  Matches Figure 5 of the paper.
# ---------------------------------------------------------------------------
def figure_5():
    cases = []
    for r1 in (0.6, 0.7, 0.8):
        glue = GlueVaR(r1=r1, gamma=GAMMA, alpha=ALPHA)
        res = solve_worst_case(LOSS, glue, theta=0.5, radius=RADIUS,
                               metric="L2")
        cases.append((f"r1={r1}", res))
    out = os.path.join(HERE, "figure_5.png")
    make_figure(
        cases,
        metric="L2",
        vary_label="L2 distance, varying GlueVaR parameter "
                   "r1 (theta=0.5, alpha=0.05, gamma=0.7, radius=10)",
        fig_title="Fig. 5",
        out_path=out,
    )
    print_summary("Figure 5 (L2, vary r1, theta=0.5)", cases)
    return out


if __name__ == "__main__":
    paths = [figure_2(), figure_3(), figure_4(), figure_5()]

    print("\nWritten figures:")
    for p in paths:
        size = os.path.getsize(p) if os.path.exists(p) else -1
        print(f"  {p}  ({size} bytes)")
