# Class-Optimized Robust Evaluation (CORE)

This package implements the class-subproblem cascade from Algorithm 1.  CORE is
kept separate from `SOCP/` because it is verifier-agnostic: SOCP is one possible
stage in the cascade, not a dependency of the cascade logic itself.

## Execution rule

For every target class, CORE invokes the ordered verifier stages until one
returns a valid negative upper bound on

```text
f_target(x') - f_reference(x').
```

That target is then certified and CORE proceeds to the next target.  If the
last stage cannot certify a target, CORE immediately returns `uncertified`.
It returns `robust` only after every target class has been certified.

Verifier errors and unaccepted solver statuses are treated conservatively as
unresolved results.  They cause escalation and, at the final stage, an
`uncertified` outcome.  Set `raise_on_verifier_error=True` only for debugging.

## Main files

| File | Purpose |
| --- | --- |
| `cascade.py` | Short, paper-facing implementation of Algorithm 1. |
| `verifier.py` | Verifier interface, problem definition, and per-input cache. |
| `result.py` | Common margin, trace, and final outcome types. |
| `adapters.py` | CROWN, Wong-Kolter, DeepPoly, and sparse SOCP adapters. |

## Example

```python
from CORE import COREProblem, run_core
from CORE.adapters import (
    CROWNMarginVerifier,
    SparseSOCPMarginVerifier,
    SparseSOCPStageConfig,
)

problem = COREProblem(
    model=model,
    x=x,
    reference_class=int(y),
    epsilon=epsilon,
    target_classes=tuple(c for c in range(10) if c != int(y)),
)

stages = [
    CROWNMarginVerifier(),
    SparseSOCPMarginVerifier(
        "base-socp",
        SparseSOCPStageConfig(max_E=8, max_S=0, max_T=0),
    ),
    SparseSOCPMarginVerifier(
        "zz-socp",
        SparseSOCPStageConfig(max_E=8, max_S=6, max_T=0),
    ),
    SparseSOCPMarginVerifier(
        "xx-socp",
        SparseSOCPStageConfig(max_E=8, max_S=6, max_T=6),
    ),
]

outcome = run_core(problem, stages)
```

The per-input `CORECache` reuses dense affine conversion, interval bounds,
target-specific influence scores, and sparse coupling patterns across stages.
The repository does not currently contain an SDP verifier; an SDP adapter can
be added later by implementing the same `MarginVerifier` interface.
