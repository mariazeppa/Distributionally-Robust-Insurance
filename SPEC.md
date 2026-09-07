# Replication spec — Boonen & Jiang (2025), Section 4 numerical study

Paper: T.J. Boonen, W. Jiang, "Distributionally robust insurance under the
Wasserstein distance," *Insurance: Mathematics and Economics* 120 (2025) 61–78.

Goal: reproduce the **numerical results of Section 4.2** (2-Wasserstein) and the
**L2 comparison of Section 4.3**, i.e. Figures 2, 3, 4, 5, plus the published
verification integrals. Code must be **readable and well commented** (the user
is a maths master's student writing a thesis chapter around this).

All formulas below are transcribed from the paper; you do NOT need the PDF.

------------------------------------------------------------------------
## 1. Model primitives

### Loss distribution (benchmark) — exponential-type, B = Q
CDF:  F_X(x) = 1 - exp(-x/100) for x in [0, 5000];  = 1 for x > 5000.
So with `scale = 100`, `M = 5000`:
- Survival:   S_Q(x) = exp(-x/100)           for x in [0, M], (define 0 for x>M, 1 for x<0)
- Quantile:   F_Q^{-1}(u) = -100 * ln(1 - u)  for u in [0,1)
- "x at survival level s":  x such that S_Q(x)=s  is  -100*ln(s)  = F_Q^{-1}(1-s)
- Tail integral:  ∫_a^M S_Q(y) dy = 100*(exp(-a/100) - exp(-M/100))

Benchmark measure 𝔹 equals the insurer's measure ℚ (B = Q) throughout Section 4.

### GlueVaR distortion  g_{α,γ}^{r1,r2}  (eq. 4.1), with r2 = 1
Parameters used: r1 ∈ {0.6,0.7,0.8}, r2 = 1, α = 0.05, γ = 0.7.
Let a = 1-γ (=0.3), b = 1-α (=0.95).
```
g(u) = (r1/(1-γ)) * u,                              0 <= u < 1-γ
     = r1 + ((r2-r1)/(γ-α)) * (u - (1-γ)),          1-γ <= u < 1-α
     = 1,                                            1-α <= u <= 1
```
With r2=1 the middle slope is (1-r1)/(γ-α). g is concave (since
(1-r1)/(γ-α) < r1/(1-γ) for these params). g(0)=0, g(1-γ)=r1, g(1-α)=1.

Inverse g^{-1}(v) for v in [0,1]:
```
g^{-1}(v) = ((1-γ)/r1) * v,                          0 <= v <= r1
          = (1-γ) + ((γ-α)/(1-r1)) * (v - r1),       r1 <  v <= 1
```
(g^{-1}(1) = 1-α; values v in (r1,1] map into (1-γ, 1-α].)

Right-derivative g'(u):
```
g'(u) = r1/(1-γ)        for 0 <= u < 1-γ
      = (1-r1)/(γ-α)    for 1-γ <= u < 1-α
      = 0               for 1-α <= u <= 1
```

------------------------------------------------------------------------
## 2. The distributionally robust problem (already reduced by the paper)

After applying the minimax theorem + optimal indemnity (Lemma 3.1), the
worst-case survival function S_{P*} solves (problem 3.3):

    sup_{S_P >= S_Q}  ∫_0^M  g(S_P(x)) ∧ (1+θ)S_Q(x)  dx
    s.t. distance(P, B) <= radius

θ is the safety loading. The **net price of marginal coverage** (right-hand
plots in every figure) is
    NP(x) = (1+θ) S_Q(x) - g(S_{P*}(x)).

Key threshold (eq. 3.11):  x0 = F_Q^{-1}(θ/(1+θ))   (so S_Q(x0) = 1/(1+θ)).

Define the region A and its complement (B = Q):
    x1 = sup{ x in [x0, M) : (1+θ)S_Q(x) >= g(S_Q(x)) },     A = [x0, x1],  Ac = [x1, M].
Closed form (second bullet, Case 1, p.69):
    x1 = F_Q^{-1}( (α r1 - α + θγ - θα) / ((γ-α)(1+θ) - (1-r1)) ).
On Ac the net price is negative ⇒ S_{P*} = S_Q there (those losses are ceded).
The optimal indemnity is the stop-loss  I*(x) = (x - x1)_+.

For the parameter sets in Section 4 (θ∈{0.1,0.5,0.9}, r1∈{0.6,0.7,0.8}) we are
always in the regime  (1-γ)/r1 < 1/(1+θ) < 1-α, so A = [x0, x1] as above.
(You may also locate x1 numerically as the largest root of
 (1+θ)S_Q(x) - g(S_Q(x)) on [x0, M]; assert it matches the closed form.)

------------------------------------------------------------------------
## 3a. 2-WASSERSTEIN distance (Section 4.2)

The squared 2-Wasserstein constraint (p=2) reduces (eq. 3.8–3.9) to
    L(S_P) := ∫_0^M x S_P(x) dx  -  ∫_0^M∫_0^M [S_P(x) ∧ S_B(y)] dx dy   <=  ζ,
    ζ = (1/2) ε  -  ∫_0^M x S_B(x) dx,      where ε = radius^2.
Equivalently the achieved squared distance is
    W2^2 = 2 * ( ∫_0^M x S_P dx - ∫_0^M∫_0^M S_P∧S_B + ∫_0^M x S_Q dx ).
Require W2^2 = ε when the constraint binds.

**Avoid the O(n^2) double integral.** Use, for any survival value s in [0,1],
    J(s) := ∫_0^M [s ∧ S_Q(y)] dy = s * x_s + ∫_{x_s}^M S_Q(y) dy,
            where x_s = F_Q^{-1}(1-s) = -100 ln s   (clip to [0,M]).
    For exponential Q:  J(s) = -100 s ln(s) + 100 (s - exp(-M/100)),  with J(0)=0.
Then ∫∫ S_P(x)∧S_B(y) dx dy = ∫_0^M J(S_P(x)) dx  (a 1-D quadrature). 

### Slack case β = 0  (Theorem 3.1, eq. 3.12)
    S̃*(x) = (t0 ∨ S_Q(x)) · 1_{[0,x0)}                 # t0 = g^{-1}(1) = 1-α
          +  g^{-1}((1+θ)S_Q(x)) · 1_{A}(x)             # A = [x0, x1)
          +  S_Q(x) · 1_{Ac}(x)                         # Ac = [x1, M]
If L(S̃*) <= ζ  (equivalently W2(S̃*) <= radius), the constraint is slack and
S_{P*} = S̃*.

### Binding case β > 0  (Theorem 3.2, eq. 3.14 + 4.2)
Define the auxiliary  Ŝ(x;β)  (eq. 4.2), with the two interior shifts
    d2 = (1-r1)/(β(γ-α)),     d1 = r1/(β(1-γ)),
and breakpoints
    x̃1 = F_Q^{-1}(α),
    x̃2 = min(F_Q^{-1}(α) + d2, M),
    x̃3 = min(F_Q^{-1}(γ) + d2, M),
    x̃4 = min(F_Q^{-1}(γ) + d1, M):
```
Ŝ(x;β) = S_Q(x),                  x in [0,  x̃1)
       = 1-α,                     x in [x̃1, x̃2)
       = S_Q(x - d2),             x in [x̃2, x̃3)
       = 1-γ,                     x in [x̃3, x̃4)
       = S_Q(x - d1),             x in [x̃4, M]
```
Then (eq. 3.14):
    S_{P*}(x;β) = Ŝ(x;β) · 1_{[0,x0)}
                + ( Ŝ(x;β) ∧ g^{-1}((1+θ)S_Q(x)) ) · 1_A
                + S_Q(x) · 1_{Ac}.
Choose β>0 by root-finding so the constraint binds: W2^2(β) = ε  (eq. 3.16).
As β→0, Ŝ→ the β=0 plateau form (S̃*); as β→∞, S_{P*}→S_Q. W2(β) is decreasing
in β, so bisect/brentq on β∈(tiny, large).

Independent cross-check (use in the verification script): for a given β the
worst case can also be found by *element-wise maximization* — for each x,
    S_{P*}(x) = argmax_{t in [S_Q(x),1]} K3(t;x),
    K3(t;x) = [ g(t) ∧ (1+θ)S_Q(x) ] - β ( x*t - J(t) ),
since (p-1)(x-y)^{p-2}=1 for p=2 and ∫_0^M t∧S_B(y) dy = J(t). K3 is concave in
t. The argmax (on a coarse x grid) must agree with the eq. (4.2) construction.

------------------------------------------------------------------------
## 3b. L2 distance (Appendix C, Section 4.3)

Constraint:  ∫_0^M (S_P(x) - S_Q(x))^2 dx <= ε,   ε = radius^2,  S_P >= S_Q.
Slack form S̃* is identical to (3.12) above (Theorem C.2).
Binding form (Theorem C.3, eq. C.5): Ŝ(x;β)=inf{t∈[S_Q(x),1]: g'(t) - β·2·(t - S_Q(x)) <= 0}.
BUT for ALL Section-4 parameter sets the L2 constraint is SLACK:
    max over θ∈{0.1,0.5,0.9}, r1∈{0.6,0.7,0.8} of  sqrt(∫_0^M (S̃*(x)-S_Q(x))^2 dx)
       = 2.5779  <  10 = radius.
So under L2 the worst case is S̃* (eq. 3.12) for every case — no β needed.
(The verification script must reproduce the number 2.5779.)

------------------------------------------------------------------------
## 4. Figures to reproduce (each = 1 row, 2 panels: left survival, right net price)

Common: 2-Wasserstein/L2 radius ε^{1/2} = 10  (so ε = 100). x-axis is the loss x.
Plot the worst-case survival functions S_{P*}(x) (left) and net price NP(x)=
(1+θ)S_Q(x)-g(S_{P*}(x)) (right). Also overlay S_Q on the left for context.

- **Fig. 2** (2-Wasserstein, vary θ): r1=0.6, θ ∈ {0.1, 0.5, 0.9}.
    θ=0.1 slack; θ=0.5, 0.9 binding.
- **Fig. 3** (2-Wasserstein, vary r1): θ=0.5, r1 ∈ {0.6, 0.7, 0.8}. All binding.
- **Fig. 4** (L2, vary θ): r1=0.6, θ ∈ {0.1, 0.5, 0.9}. All slack (=S̃*).
- **Fig. 5** (L2, vary r1): θ=0.5, r1 ∈ {0.6, 0.7, 0.8}. All slack (=S̃*).

A useful viewing window is x ∈ [0, ~600] (curves meet S_Q in the tail), but
integrate over the full [0, M].

------------------------------------------------------------------------
## 5. Published numbers to reproduce (verification targets)

Worst-case mean = ∫_0^M S_{P*}(x) dx  ( = E_{P*}[X] ).

2-Wasserstein, vary θ (r1=0.6): writing S*_1,S*_2,S*_3 for θ=0.1,0.5,0.9,
    ∫ S*_2 dx = 106.59  <  107.78 = ∫ S*_3 dx.        (X1 ≤st X2 ≤icx X3)
2-Wasserstein, vary r1 (θ=0.5): writing S*_1,S*_2,S*_3 for r1=0.6,0.7,0.8,
    ∫ S*_1 dx = 106.59 > ∫ S*_2 dx = 105.99 > ∫ S*_3 dx = 105.53.   (X3 ≤icx X2 ≤icx X1)
    (Note ∫S*_1 here, r1=0.6 θ=0.5, equals ∫S*_2 of the θ-table: same case, =106.59.)
L2: max over all 6 cases of sqrt(∫(S̃*-S_Q)^2 dx) = 2.5779 < 10.

Tolerance: reproduce to ~0.01 (two decimals) ⇒ use a fine x-grid for integrals
(e.g. n ≈ 200001 points on [0, M], trapezoidal rule).

------------------------------------------------------------------------
## 6. Deliverable layout (target API — code to this contract)

`dro_insurance.py` — core library (Agent A). Public API:

    @dataclass
    class GlueVaR:
        r1: float; gamma: float; alpha: float; r2: float = 1.0
        def g(self, u):       # vectorized distortion, u array/scalar in [0,1]
        def g_inv(self, v):   # vectorized inverse
        def g_prime(self, u): # right-derivative

    @dataclass
    class ExponentialLoss:
        scale: float = 100.0; M: float = 5000.0
        def S(self, x):              # survival (vectorized)
        def F_inv(self, u):          # quantile
        def x_at_survival(self, s):  # = F_inv(1-s) = -scale*ln(s)
        def tail_integral(self, a):  # ∫_a^M S(y) dy
        def J(self, s):              # ∫_0^M [s ∧ S(y)] dy

    @dataclass
    class WorstCaseResult:
        x: np.ndarray; S_star: np.ndarray; S_Q: np.ndarray; net_price: np.ndarray
        x0: float; x1: float; beta: float; binding: bool
        distance: float    # achieved W2 or L2 distance (not squared)
        mean: float        # ∫_0^M S_star dx

    def solve_worst_case(loss, glue, theta, radius, metric="wasserstein",
                         n=200001) -> WorstCaseResult
        # metric in {"wasserstein", "L2"}; radius = ε^{1/2} (=10 in the paper).
        # builds x grid on [0, M], computes x0, x1, the β=0 form, decides
        # slack vs binding, root-finds β if binding (wasserstein only here),
        # returns the full result.

Include `if __name__ == "__main__":` self-tests in dro_insurance.py asserting:
g(0)=0, g(1-γ)=r1, g(1-α)=1, g_inv(g(u))≈u; and that solve_worst_case for the
2-Wasserstein θ=0.5 and θ=0.9 cases (r1=0.6, radius=10) yields means
≈106.59 and ≈107.78.

`reproduce_figures.py` — Agent B. Imports the API, generates Fig 2–5 as
PNG files (figure_2.png … figure_5.png) in this folder, 1×2 subplots each,
clear legends/labels/titles, and a short comment block per figure.

`verify_paper_values.py` — Agent C. Independently recomputes and prints a table
of all verification numbers in §5 (with PASS/FAIL vs the published values),
including an independent element-wise-maximization cross-check of the binding
worst case, and the L2 slack check (2.5779). Plus a short README.md.
