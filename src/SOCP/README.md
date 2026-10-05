# SOCP Verifier

Evaluate pretrained ReLU classifiers on MNIST and CIFAR-10 using sparse
second-order cone programming (SOCP) relaxations. The evaluation path supports
a fully connected MNIST model (`tiny`), a MNIST CNN (`cnn`), and a CIFAR-10 CNN5
(`cifar10`). It solves a target-versus-true logit margin problem for each selected
incorrect class.

This source bundle is incomplete as a standalone project: the model builders,
data loaders, interval-layer helpers, and legacy relaxation selector listed
below were not supplied. No training is required to evaluate an existing
checkpoint.

## Dependencies and setup

Use an environment with PyTorch and the dataset loader dependencies installed.
For standard torchvision-based loaders, the following includes the usual
verification dependencies:

```bash
pip install torch torchvision numpy tqdm cvxpy clarabel scs
```

Match the PyTorch installation to your hardware. The optional training losses
also require `cvxpylayers`; it is not needed by the evaluation entry point.

Place the verifier modules under the project's `src/SOCP/` package and run the
commands below from `src/`. Use the canonical filenames in this bundle: upload
suffixes such as `(1)` or `(3)` are not Python module names used by the imports.
Provide package markers such as `__init__.py` if the project uses regular Python
packages rather than namespace packages. See the import-layout issue below
before treating this bundle as runnable.

## Evaluate a checkpoint

The architecture arguments must match the saved checkpoint. The evaluator
accepts either a state dictionary or a dictionary whose `model` key contains
the state dictionary. Inputs must match the checkpoint's preprocessing and the
verifier's `[0, 1]` image domain. Mean/std-normalized images require corresponding
changes to the input bounds; switching solver files does not change preprocessing.

Example for a MNIST CNN; replace the checkpoint and dimensions as needed:

```bash
python -m SOCP.eval_SOCP_robustness \
  --checkpoint ../checkpoints/mnist_model.pt \
  --model_type cnn --epsilon 0.3 \
  --n1 16 --n2 32 --linear_size 100 \
  --max_E 8 --max_S 4 --max_T 4 \
  --prev_candidate_limit 32 --gamma_backend dual \
  --all_classes --solver SCS --solver_max_iters 200
```

Use `--model_type tiny` for the fully connected MNIST builder. Example for the
CIFAR-10 CNN5 dimensions used in the supplied evaluator's original examples:

```bash
python -m SOCP.eval_SOCP_robustness \
  --checkpoint ../checkpoints/cifar10/eps_0.0088889/cifar10_deeppoly_cnn5_standard.pt \
  --model_type cifar10 --epsilon 0.0088889 \
  --n1 24 --n2 48 --n3 96 --linear_size 192 \
  --max_E 8 --max_S 4 --max_T 4 \
  --prev_candidate_limit 32 --gamma_backend dual \
  --all_classes --solver SCS --solver_max_iters 200
```

These commands illustrate argument usage; the checkpoint files were not
included. Epsilon is an L-infinity budget in the model's input units.

## Evaluation behavior

- The current evaluator processes exactly the first **200 test images** for
  either dataset and raises an error if the loader provides fewer images.
- Cleanly misclassified images count as uncertified and skip the SOCP solver.
- Certification handles one image per solver call. `--test_batch_size` controls
  loading batches, not the number of images in a cone program.
- `--all_classes` checks all nine incorrect classes for these ten-class models.
  Without it, the default is the four incorrect classes with the largest clean
  logits. A result for selected targets is not a full multiclass certificate.
- The current `certified` property checks only whether the largest solved margin
  is negative. It does not require an `optimal` status and can count
  `optimal_inaccurate` solutions. Review solver status and numerical accuracy
  before interpreting the printed value as a validated robustness guarantee.
- No PGD pre-check is present in this uploaded evaluator.

The output reports clean accuracy, the margin-based certified flag rate,
certification rate among cleanly correct images, average worst margin, runtime,
variable/constraint counts, and solver statuses. The average solve time measures
the full `certify_sample` call for cleanly correct images, including affine
conversion, bound collection, coupling selection, and all selected target solves.

## Important evaluation options

| Option | Evaluator default | Purpose |
| --- | --- | --- |
| `--model_type` | `tiny` | `tiny`/`cnn` select MNIST; `cifar10` selects CIFAR-10. |
| `--epsilon` | Required | L-infinity perturbation radius. |
| `--nodes_per_layer` | `8` | Number of high-scoring neuron candidates retained per hidden layer. |
| `--max_E` | `8` | Maximum selected previous/current activation products per layer. |
| `--max_S` | `6` | Maximum selected current/current activation products per layer. |
| `--max_T` | `6` | Maximum selected previous/previous activation products per layer. |
| `--prev_candidate_limit` | `16` | Limit previous-layer candidates before scoring pairs. |
| `--gamma_backend` | `dual` | CROWN-style (`dual`), fixed-alpha Wong--Kolter (`dual_WK`), or clean-gradient (`clean`) influence scores. |
| `--solver` | `SCS` | Conic solver; `CLARABEL` is also handled explicitly. |
| `--solver_max_iters` | `200` | Solver iteration limit, not a time limit. |
| `--solver_verbose` | Off | Show solver diagnostics. |
| `--all_classes` | Off | Check every incorrect class. |
| `--max_targets` | `4` | Number of classes to check when `--all_classes` is absent. |
| `--test_batch_size` | `9` | DataLoader test batch size. |

Couplings are selected separately for each sample and target. Square lifts are
created only when needed by the selected pairs. The input to the first layer is
the perturbed image; in later layers, it is the preceding ReLU activation.

## Slow CIFAR-10 evaluation: use the legacy backend

**If the code is running slowly on CIFAR-10, you can use `socp_solver_old.py`
and `socp_relaxation_old.py` as the legacy fallback.** The same fallback can be
tried for slow MNIST runs.

The evaluator imports the active module names `SOCP.socp_solver` and
`SOCP.socp_relaxation`. Merely placing `_old.py` files beside them does not select
the fallback. When both legacy files are available, back up the current modules
and copy the legacy implementations to the active filenames. Run this from
`src/` in a fresh shell:

```bash
if [ ! -f SOCP/socp_solver_old.py ] || [ ! -f SOCP/socp_relaxation_old.py ]; then
  echo "Both legacy backend files are required before switching." >&2
  exit 1
fi

mkdir -p SOCP/backend_backup
cp -n SOCP/socp_solver.py SOCP/backend_backup/socp_solver.py
cp -n SOCP/socp_relaxation.py SOCP/backend_backup/socp_relaxation.py
cp SOCP/socp_solver_old.py SOCP/socp_solver.py
cp SOCP/socp_relaxation_old.py SOCP/socp_relaxation.py
```

Restart Python and rerun the same evaluation command. To restore the backed-up
modules:

```bash
cp SOCP/backend_backup/socp_solver.py SOCP/socp_solver.py
cp SOCP/backend_backup/socp_relaxation.py SOCP/socp_relaxation.py
```

**Attachment status:** `socp_solver_old.py` was supplied, but
`socp_relaxation_old.py` was not. The supplied `socp_relaxation_new.py` is
byte-for-byte identical to the uploaded `socp_relaxation(1).py`; neither can be
identified as the missing legacy selector from these attachments.

The legacy solver omits the current solver's residual constraints linking
selected E products to activation squares. Switching therefore changes the
relaxation and can weaken bounds; a speedup has not been measured in this review.
Both versions still use dense convolution matrices. `certify_sample` rebuilds
those matrices for each sample, so dense conversion and CVXPY construction can
remain expensive on larger CIFAR-10 CNNs. Record the backend version when
comparing results. Reducing coupling budgets can also reduce work, but checking
fewer targets yields only a partial-target result.

## Files needed for verification

| File | Role |
| --- | --- |
| `eval_SOCP_robustness.py` | Dataset/checkpoint loading, subset evaluation, and metrics. |
| `socp_certificate.py` | Per-sample and per-target orchestration. |
| `socp_bounds.py` | Input, pre-activation, post-ReLU, and logit intervals. |
| `socp_influence.py` | Backward influence scores for selecting couplings. |
| `socp_relaxation.py` | Sparse E/S/T coupling selection. |
| `socp_solver.py` | CVXPY model construction and target-margin optimization. |
| `utils.py` | Device helpers and dense affine conversion. |
| `dual_bounds.py` | Shared interval records and backward bound routines; required by verification despite also supporting training. |

## Optional or redundant supplied files

| File | Needed by the supplied evaluator? | Recommendation |
| --- | --- | --- |
| `socp_cnn_hybrid_loss.py` | No | Training-only differentiable CNN-head loss; omit from a verification-only package. |
| `socp_cvxpy_layer_loss.py` | No | Training-only differentiable fully connected loss; omit from a verification-only package. |
| `socp_relaxation_new.py` | No | Exact duplicate of the supplied active selector; retain only if desired as a backup. |
| `socp_solver_old.py` | Only when activated as the fallback | Keep for the requested legacy backend. |
| `FLOWCHART.md` | No runtime dependency | Optional pipeline documentation. |

No supplied files were deleted. The training loss modules use `cvxpylayers` and
are separate from the non-differentiable post-hoc evaluation solver. Their
presence does not mean a training entry point was supplied.

## Missing files and integration issue

| Missing item | Why it is needed |
| --- | --- |
| `model.py` | The evaluator imports `build_mnist_model`, `build_mnist_tiny_model`, and `build_cnn5_model`. |
| `data.py` | The evaluator imports `get_mnist_loaders` and `get_cifar10_loaders`; needed to check preprocessing as well as load data. |
| `bound_layers.py` | `dual_bounds.py` imports `TensorPair`, `initial_linf_bounds`, `conv2d_interval`, `linear_interval`, and `relu_interval_relaxation`. |
| `socp_relaxation_old.py` | Required for the complete two-file legacy fallback requested above. |
| Model checkpoint(s) | Required for an actual evaluation; architecture arguments must match the saved weights. |

If retaining training, `deeppoly_bounds.py` is additionally needed for the CNN
loss's `bound_method="deeppoly"` branch. The original documentation also referred
to `socp_train.py`, `socp_loss.py`, and training YAML configurations, none of which
were supplied. They are unnecessary for the checkpoint-evaluation path and their
training behavior cannot be reviewed from this bundle.

**Import-layout issue:** `socp_bounds.py` first imports `.dual_bounds`, then tries
top-level `dual_bounds`; `socp_influence.py` tries the same top-level
`dual_bounds` import in both its `try` and `except` branches. The supplied
`dual_bounds.py` itself uses a relative `.bound_layers` import. Loading that
file as a top-level module will fail with an attempted relative import error.
The complete project's package layout must resolve this mismatch before the
evaluation can run. This review updates comments, help/error wording, and
documentation; import statements and verification logic were preserved.
