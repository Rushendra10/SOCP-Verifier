"""
Exact sparse SOCP certificate construction using CVXPY.

This file builds the mathematical relaxation from the paper:

1. Input box and affine layer equations.
2. Scalar ReLU convex-hull constraints.
3. Selected lifted variables.
4. McCormick envelopes for lifted products.
5. SOC-representable 2x2 PSD minors.
6. Class-wise margin maximization.

Important research note
-----------------------
CVXPY solves the certificate, but this module does not yet implement implicit
KKT differentiation through the conic solver.  Therefore it is suitable for
post-hoc certificates and for building the exact formulation.  A publishable
training implementation should add either differentiable unrolled solver steps
or implicit differentiation around this same problem structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from SOCP.socp_bounds import LayerBounds
from SOCP.socp_relaxation import SOCPPattern
from SOCP.utils import DenseAffineLayer

try:
    import cvxpy as cp
except Exception:  # pragma: no cover - allows import before cvxpy installation
    cp = None  # type: ignore


@dataclass
class SOCPResult:
    """Result returned by the SOCP solver."""

    status: str
    value: float
    target_class: int
    true_class: int
    solver: str
    num_variables: int
    num_constraints: int
    metadata: Dict[str, float] = field(default_factory=dict)


def _add_box_constraints(var, lower: np.ndarray, upper: np.ndarray, constraints: list) -> None:
    """Add lower <= var <= upper componentwise constraints."""
    constraints += [var >= lower, var <= upper]


def _add_relu_hull_constraints(s, z, lower: np.ndarray, upper: np.ndarray, constraints: list) -> None:
    """
    Add scalar ReLU convex hull constraints for every neuron.

    Stable active:   z = s
    Stable inactive: z = 0
    Unstable:        z >= 0, z >= s, z <= alpha (s - l)
    """
    eps = 1e-12
    active = lower >= 0.0
    inactive = upper <= 0.0
    unstable = (~active) & (~inactive)

    if np.any(active):
        constraints.append(z[active] == s[active])
    if np.any(inactive):
        constraints.append(z[inactive] == 0.0)
    if np.any(unstable):
        l = lower[unstable]
        u = upper[unstable]
        alpha = u / (u - l + eps)
        constraints += [
            z[unstable] >= 0.0,
            z[unstable] >= s[unstable],
            z[unstable] <= cp.multiply(alpha, s[unstable] - l),
        ]

    # Always include certified boxes for numerical stability.
    constraints += [s >= lower, s <= upper]
    constraints += [z >= np.maximum(lower, 0.0), z <= np.maximum(upper, 0.0)]


def _add_square_lift(name: str, a, lower: float, upper: float, constraints: list):
    """
    Create a lifted diagonal variable d approximately equal to a^2.

    Convex valid constraints over [l,u]:
        d >= a^2
        d <= (l+u) a - l u
        d >= 0

    The upper inequality is the secant line over the interval.  Together these
    form the convex hull of the scalar square on a bounded interval.
    """
    d = cp.Variable(name=name)
    constraints += [d >= 0.0, d >= cp.square(a)]
    constraints.append(d <= (lower + upper) * a - lower * upper)
    return d


def _add_mccormick(name: str, a, b, la: float, ua: float, lb: float, ub: float, constraints: list):
    """
    Create w ~= a*b and add the four McCormick envelope constraints.
    """
    w = cp.Variable(name=name)
    constraints += [
        w >= la * b + lb * a - la * lb,
        w >= ua * b + ub * a - ua * ub,
        w <= ua * b + lb * a - ua * lb,
        w <= la * b + ub * a - la * ub,
    ]
    return w


def _add_soc_minor(offdiag, diag_a, diag_b, constraints: list) -> None:
    """
    Add the 2x2 PSD minor constraint:

        [[diag_a, offdiag], [offdiag, diag_b]] >= 0

    Equivalent SOC form:

        || [2 offdiag, diag_a - diag_b] ||_2 <= diag_a + diag_b
    """
    constraints.append(cp.SOC(diag_a + diag_b, cp.hstack([2.0 * offdiag, diag_a - diag_b])))


class SparseSOCPVerifier:
    """
    Build and solve class-wise sparse SOCP certificates.

    The class-wise objective is

        max f_target(x') - f_true(x')

    over the structured SOCP relaxation.  The returned value is phi_SOCP.
    """

    def __init__(self, solver: str = "CLARABEL", verbose: bool = False, max_iters: int = 500):
        if cp is None:
            raise ImportError("cvxpy is required. Install with: pip install cvxpy clarabel scs")
        self.solver = solver
        self.verbose = verbose
        self.max_iters = max_iters

    def solve_margin(
        self,
        affines: Sequence[DenseAffineLayer],
        input_lower: np.ndarray,
        input_upper: np.ndarray,
        hidden_bounds: Sequence[LayerBounds],
        pattern: SOCPPattern,
        true_class: int,
        target_class: int,
    ) -> SOCPResult:
        """Build and solve one target-vs-true SOCP margin problem."""
        constraints = []
        square_lifts: Dict[Tuple[str, int], object] = {}

        # z_vars[0] is the input x. z_vars[k] for k>=1 are hidden ReLU activations.
        z_vars = []
        z_lowers = []
        z_uppers = []

        x_var = cp.Variable(input_lower.shape[0], name="x")
        _add_box_constraints(x_var, input_lower, input_upper, constraints)
        z_vars.append(x_var)
        z_lowers.append(input_lower)
        z_uppers.append(input_upper)

        s_vars = []

        # Hidden affine + ReLU stages.
        prev = x_var
        for k, hb in enumerate(hidden_bounds):
            A = affines[k].weight
            b = affines[k].bias
            s = cp.Variable(A.shape[0], name=f"s_{k+1}")
            z = cp.Variable(A.shape[0], name=f"z_{k+1}")
            constraints.append(s == A @ prev + b)
            _add_relu_hull_constraints(s, z, hb.s_lower, hb.s_upper, constraints)
            s_vars.append(s)
            z_vars.append(z)
            z_lowers.append(hb.z_lower)
            z_uppers.append(hb.z_upper)
            prev = z

        # Final affine logits, no final ReLU.
        A_final = affines[len(hidden_bounds)].weight
        b_final = affines[len(hidden_bounds)].bias
        logits = A_final @ prev + b_final

        # Add lifted variables, McCormick envelopes, and SOC minors.
        for layer_sets in pattern.per_layer:
            k = layer_sets.layer_index + 1  # z_vars index of current hidden layer
            prev_z = z_vars[k - 1]
            curr_z = z_vars[k]
            prev_l = z_lowers[k - 1]
            prev_u = z_uppers[k - 1]
            curr_l = z_lowers[k]
            curr_u = z_uppers[k]

            def get_square(layer_name: str, var, lower_arr, upper_arr, idx: int):
                key = (layer_name, int(idx))
                if key not in square_lifts:
                    square_lifts[key] = _add_square_lift(
                        name=f"sq_{layer_name}_{idx}",
                        a=var[idx],
                        lower=float(lower_arr[idx]),
                        upper=float(upper_arr[idx]),
                        constraints=constraints,
                    )
                return square_lifts[key]

            # E: cross-layer pairs eta_ij ~= z_prev_i * z_curr_j.
            for i, j in layer_sets.E:
                eta = _add_mccormick(
                    name=f"eta_L{k}_{i}_{j}",
                    a=prev_z[i],
                    b=curr_z[j],
                    la=float(prev_l[i]),
                    ua=float(prev_u[i]),
                    lb=float(curr_l[j]),
                    ub=float(curr_u[j]),
                    constraints=constraints,
                )
                di = get_square(f"z{k-1}", prev_z, prev_l, prev_u, i)
                dj = get_square(f"z{k}", curr_z, curr_l, curr_u, j)
                _add_soc_minor(eta, di, dj, constraints)

            # S: current-layer pairs sigma_jj' ~= z_curr_j * z_curr_j'.
            for j, jp in layer_sets.S:
                sigma = _add_mccormick(
                    name=f"sigma_L{k}_{j}_{jp}",
                    a=curr_z[j],
                    b=curr_z[jp],
                    la=float(curr_l[j]),
                    ua=float(curr_u[j]),
                    lb=float(curr_l[jp]),
                    ub=float(curr_u[jp]),
                    constraints=constraints,
                )
                dj = get_square(f"z{k}", curr_z, curr_l, curr_u, j)
                djp = get_square(f"z{k}", curr_z, curr_l, curr_u, jp)
                _add_soc_minor(sigma, dj, djp, constraints)

            # T: previous-layer pairs tau_pq ~= z_prev_p * z_prev_q.
            for p, q in layer_sets.T:
                tau = _add_mccormick(
                    name=f"tau_L{k}_{p}_{q}",
                    a=prev_z[p],
                    b=prev_z[q],
                    la=float(prev_l[p]),
                    ua=float(prev_u[p]),
                    lb=float(prev_l[q]),
                    ub=float(prev_u[q]),
                    constraints=constraints,
                )
                dp = get_square(f"z{k-1}", prev_z, prev_l, prev_u, p)
                dq = get_square(f"z{k-1}", prev_z, prev_l, prev_u, q)
                _add_soc_minor(tau, dp, dq, constraints)

        objective = cp.Maximize(logits[target_class] - logits[true_class])
        problem = cp.Problem(objective, constraints)

        solve_kwargs = {"verbose": self.verbose}
        if self.solver.upper() in {"SCS"}:
            solve_kwargs["max_iters"] = self.max_iters
        elif self.solver.upper() in {"CLARABEL"}:
            solve_kwargs["max_iter"] = self.max_iters

        # value = problem.solve(solver=self.solver, **solve_kwargs)
        try:
            value = problem.solve(solver=self.solver, **solve_kwargs)
        except Exception as e:
            print(f"[SOCP] Solver {self.solver} failed: {e}")
            value = problem.solve(solver="SCS", max_iters=500, verbose=False)
    
        if value is None or not np.isfinite(value):
            value = float("inf")

        return SOCPResult(
            status=str(problem.status),
            value=float(value),
            target_class=target_class,
            true_class=true_class,
            solver=self.solver,
            num_variables=int(sum(v.size for v in problem.variables())),
            num_constraints=len(problem.constraints),
            metadata={"num_square_lifts": float(len(square_lifts))},
        )
