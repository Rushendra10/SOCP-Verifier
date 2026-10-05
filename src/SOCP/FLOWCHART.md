# SOCP Training / Certification Flowchart with Option-B Gamma

This folder implements the structured SOCP certificate pipeline from the AAAI draft using sparse lifted variables, McCormick envelopes, and SOC-representable 2x2 PSD minors.

The important update is that sparse coupling selection now uses **Option B**:

```text
gamma comes from WK dual_WK or CROWN/dual backward relaxation
```

rather than using interval width alone.

---

## High-level flow

```text
MNIST sample (x, y_true)
        |
        v
Existing MNIST model from src/model.py
        |
        v
Forward logits f_theta(x)
        |
        +--------------------------+
        |                          |
        v                          v
Clean cross entropy          Interval bound propagation
                              using existing collect_bound_records()
                                      |
                                      v
                         Certified bounds for each ReLU:
                         l_s^(k), u_s^(k), l_z^(k), u_z^(k)
                                      |
                                      v
                         Identify unstable ReLUs:
                         l_s < 0 < u_s
                                      |
                                      v
For each target class y_target != y_true:
                                      |
                                      v
Build target-vs-true margin vector:
c = e_target - e_true
                                      |
                                      v
Choose gamma backend:

    gamma_backend = dual_WK   -> Wong--Kolter fixed-alpha LP-dual
    gamma_backend = dual      -> CROWN / LP backward relaxation
    gamma_backend = clean     -> clean autograd gradient only
                                      |
                                      v
Backward relaxation pass
                                      |
                                      v
Extract gamma^(k) at every ReLU preactivation s^(k)
                                      |
                                      v
Node selection, Eq. (26):
node_score_j^(k) = |gamma_j^(k)| (u_s,j^(k) - l_s,j^(k))
                                      |
                                      v
Keep top J_k unstable neurons
                                      |
                                      v
Pair selection:

E pairs, Eq. (27): previous activation x current activation
S pairs, Eq. (28): current activation x current activation
T pairs, Eq. (29): previous activation x previous activation
                                      |
                                      v
Build lifted SOCP relaxation:

1. LP ReLU convex hull
2. diagonal square lifts xi, zeta
3. sparse products eta, sigma, tau
4. McCormick envelopes
5. SOC 2x2 PSD minors
                                      |
                                      v
Solve class-wise SOCP:
phi_SOCP(x, y_true, y_target)
                                      |
                                      v
Worst target margin:
Phi_SOCP = max_target phi_SOCP
                                      |
                                      v
Certified loss:
psi_lse = log(1 + sum_target exp(phi_SOCP))
                                      |
                                      v
Training objective:
CE(f_theta(x), y_true) + lambda * psi_lse
```

---

## What exactly is gamma?

For a target class, the adversarial margin is

```text
m(x) = z_target(x) - z_true(x).
```

Gamma is the backward influence of each ReLU preactivation on this margin under a cheap relaxation.

For the recommended setting:

```yaml
gamma_backend: dual_WK
```

the code runs the Wong--Kolter fixed-alpha backward recursion and records the coefficient that reaches each ReLU preactivation. This coefficient is `gamma^(k)`.

For CROWN:

```yaml
gamma_backend: dual
```

the code runs the CROWN-style sign-dependent backward ReLU relaxation and records the corresponding preactivation coefficient.

The SOCP pair selector uses `abs(gamma)`, so the sign of the backward coefficient is not important for ranking; the magnitude tells us how much that neuron affects the relaxed target margin.

---

## File mapping

```text
socp_influence.py
    Computes gamma from dual_WK, dual/CROWN, or clean gradients.

socp_bounds.py
    Collects certified input, preactivation, activation, and logit bounds.

socp_relaxation.py
    Uses gamma and interval widths to select sparse E, S, T coupling sets.

socp_solver.py
    Builds LP hull constraints, lifted variables, McCormick envelopes,
    and SOC 2x2 PSD minors in CVXPY.

socp_certificate.py
    Runs the full per-target certificate pipeline.

socp_loss.py
    Converts solved class-wise margins into psi_lse.

socp_train.py
    Training/evaluation entry point.
```

---

## Defensible statement for the paper

The sparse SOCP module does not choose lifted variables arbitrarily. For each target margin, it first computes a backward influence signal from a cheap LP certificate, preferably the Wong--Kolter fixed-alpha dual. It then selects the unstable ReLUs and lifted pair constraints with the largest product of relaxation uncertainty and margin influence. Therefore, the added SOC constraints are concentrated on the neurons and pairwise interactions most likely to loosen the LP certificate.
