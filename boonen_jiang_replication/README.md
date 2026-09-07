# Replication: Boonen & Jiang (2025), *Distributionally robust insurance under the Wasserstein distance*

This repository reproduces the numerical study of

> T.J. Boonen, W. Jiang, "Distributionally robust insurance under the
> Wasserstein distance," *Insurance: Mathematics and Economics* **120** (2025)
> 61–78.

specifically **Section 4.2** (worst-case indemnity under the 2-Wasserstein
ambiguity set) and the **L2 comparison of Section 4.3** (Appendix C). The code
regenerates Figures 2–5 and recomputes every published verification integral.

This is a master's-thesis replication: the code is written to be read, not just
run. Equation numbers in the docstrings refer to the published paper.

---

## What is replicated

The insurer faces an exponential benchmark loss and chooses an indemnity to
minimize a distortion-risk-based price, robustly over all loss distributions `P`
within an ambiguity ball of the benchmark `B`. By the paper's minimax reduction
(Lemma 3.1 + the optimal stop-loss indemnity), the problem collapses to finding
the **worst-case survival function** `S_{P*}` that solves problem (3.3):

```
sup_{S_P >= S_Q}  ∫_0^M  g(S_P(x)) ∧ (1+θ) S_Q(x)  dx     s.t.  dist(P, B) <= ε^{1/2}
```

We reproduce, for both ambiguity metrics:

- the worst-case survival functions `S_{P*}(x)` (left panels of Figs 2–5);
- the **net price of marginal coverage** `NP(x) = (1+θ) S_Q(x) − g(S_{P*}(x))`
  (right panels);
- the worst-case means `∫_0^M S_{P*}(x) dx = E_{P*}[X]` (verification integrals).

## Model

| Component | Choice |
|---|---|
| Benchmark loss `X` | exponential, `S_Q(x) = exp(−x/100)` on `[0, M]`, `M = 5000` |
| Benchmark measure | `B = Q` (the insurer's own measure) throughout Section 4 |
| Distortion `g` | **GlueVaR** `g_{α,γ}^{r1,r2}` (eq. 4.1) with `r2 = 1`, `α = 0.05`, `γ = 0.7` |
| Safety loading `θ` | `{0.1, 0.5, 0.9}` |
| GlueVaR slope `r1` | `{0.6, 0.7, 0.8}` |
| Ambiguity metric | **2-Wasserstein** (Sec 4.2) vs **L2** (Sec 4.3 / App. C) |
| Ambiguity radius | `ε^{1/2} = 10`  (so `ε = 100`) |

The GlueVaR is a concave, piecewise-linear distortion with kinks at `u = 1−γ`
and `u = 1−α`; it puts extra weight on the loss tail.

## Method (key equations)

- **Threshold** `x0 = F_Q^{-1}(θ/(1+θ))` so that `S_Q(x0) = 1/(1+θ)` — eq. **(3.11)**.
- **Active region** `A = [x0, x1]`, with `x1` the largest root of
  `(1+θ)S_Q(x) − g(S_Q(x))`; closed form on p. 69. On `Ac = [x1, M]` the net
  price is negative, so the loss is ceded and `S_{P*} = S_Q`. The optimal
  indemnity is the stop-loss `I*(x) = (x − x1)_+`.
- **Slack worst case** `S̃*` (β = 0) — Theorem 3.1, eq. **(3.12)** (identical for
  both metrics, Theorem C.2).
- **Binding 2-Wasserstein worst case** — Theorem 3.2, eq. **(3.14)** built on the
  auxiliary `Ŝ(x; β)` of eq. **(4.2)**; `β > 0` chosen so the achieved
  `W2² = ε`, eq. **(3.16)**. The squared distance uses the closed-form
  `J(s) = ∫_0^M [s ∧ S_Q(y)] dy` to avoid the O(n²) double integral.
- **L2 binding form** — Theorem C.3, eq. **(C.5)** (not needed: all Section-4
  cases are slack).

## File layout

| File | Role |
|---|---|
| `dro_insurance.py` | **Core library.** `GlueVaR`, `ExponentialLoss`, `WorstCaseResult`, and `solve_worst_case(loss, glue, theta, radius, metric, n)`. Has self-tests under `__main__`. |
| `reproduce_figures.py` | Generates `figure_2.png` … `figure_5.png` (1×2 panels each: worst-case survival + net price). |
| `verify_paper_values.py` | **Independent verification.** Recomputes every §5 number with PASS/FAIL, and cross-checks the binding worst case by element-wise maximization of `K3(t;x)` (does not reuse eq. 4.2). |
| `SPEC.md` | Full transcription of the math and the published targets. |

## How to run

```bash
python3 dro_insurance.py        # library self-tests
python3 reproduce_figures.py    # writes figure_2.png ... figure_5.png
python3 verify_paper_values.py  # prints the verification table (all PASS)
```

### Dependencies

`numpy`, `scipy` (core + verification) and `matplotlib` (figures only). No other
packages are required.

## Reproduced numbers

Worst-case mean `= ∫_0^M S_{P*}(x) dx`. Integrals use a fine grid
(`n ≈ 200 001` points on `[0, M]`, trapezoidal rule).

| Setting | Quantity | Published | Reproduced |
|---|---|---|---|
| 2-Wasserstein, `r1=0.6`, `θ=0.5` | `∫ S*` | 106.59 | 106.59 |
| 2-Wasserstein, `r1=0.6`, `θ=0.9` | `∫ S*` | 107.78 | 107.78 |
| 2-Wasserstein, `θ=0.5`, `r1=0.7` | `∫ S*` | 105.99 | 106.00 |
| 2-Wasserstein, `θ=0.5`, `r1=0.8` | `∫ S*` | 105.53 | 105.53 |
| L2, max over all 6 cases | `sqrt ∫ (S̃*−S_Q)²` | 2.5779 (< 10) | 2.5779 |

Orderings (from the stochastic-dominance results of the paper):

- vary `θ` (`r1=0.6`): `∫S*_2 = 106.59 < 107.78 = ∫S*_3`, i.e. `X1 ≤st X2 ≤icx X3`.
- vary `r1` (`θ=0.5`): `∫S*_1 = 106.59 > 105.99 > 105.53`, i.e. `X3 ≤icx X2 ≤icx X1`.

Under L2 every Section-4 case is **slack**, so the worst case equals the closed
form `S̃*` of eq. (3.12) and no `β` root-finding is needed.

### Independent cross-check

`verify_paper_values.py` confirms the binding 2-Wasserstein construction by an
independent pointwise optimization. For the library's solved `β`, it maximizes

```
K3(t; x) = min( g(t), (1+θ) S_Q(x) ) − β ( x·t − J(t) )    over t ∈ [S_Q(x), 1]
```

at each `x` (concave 1-D program, `scipy.optimize.minimize_scalar`). This agrees
with the eq. (4.2)/(3.14) library solution to ~2e-8 for `θ = 0.5` and `θ = 0.9`,
confirming the construction is implemented correctly.
