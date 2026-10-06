# SOCP Verifier

Evaluate pretrained ReLU classifiers on **MNIST and CIFAR-10** with sparse
second-order cone programming (SOCP) relaxations. The verifier constructs an
input perturbation box, propagates activation bounds, selects sparse lifted
products, and solves a target-versus-true logit margin problem with CVXPY.

The evaluation entry point supports a MNIST MLP, a MNIST CNN, and a CIFAR-10
CNN5. Pretrained checkpoints and experiment configurations are included.
Evaluation uses saved weights; retraining is not required.


## Repository structure

The repository is organized as follows. Paths are relative to the repository
root, and folder names are case-sensitive.

| Folder | Purpose | Contents |
| --- | --- | --- |
| [`checkpoints/`](checkpoints/) | Saved model weights for evaluation. | MNIST models under `eps_0.1/` and `eps_0.3/`; CIFAR-10 models under `cifar10/`. |
| [`configs/`](configs/) | Records of model and training experiment settings. | Nine YAML files for MNIST/CIFAR-10 LP, SOCP, and smoothing experiments. The evaluator takes CLI arguments and does not load these YAML files. |
| [`data/`](data/) | Local dataset storage. | A committed MNIST dataset cache. CIFAR-10 data is not included. This directory does not replace the missing Python loader module. |
| [`scripts/`](scripts/) | Environment setup reference. | `create_env.sh`, containing Conda, PyTorch, utility, and conic-solver installation commands. See the setup note below before using it. |
| [`src/`](src/) | Model definitions and shared bound routines. | `model.py`, `bound_layers.py`, `dual_bounds.py`, `deeppoly_bounds.py`, and the `SOCP/` package. |
| [`src/SOCP/`](src/SOCP/) | Active verifier implementation and documentation. | Evaluation, certificates, bounds, influence scoring, coupling selection, conic solvers, utilities, `__init__.py`, `README.md`, and `FLOWCHART.md`. |
| [`src/SOCP/Old_codes/`](src/SOCP/Old_codes/) | Archived implementations and training experiments. | An older certificate API, affine conversion helpers, and three loss modules. These are separate from the active evaluation path. |

### Checkpoint folders

The epsilon folder names record experiment budgets; evaluation epsilon is set
separately with `--epsilon`. Checkpoint names describe training variants, while
verification uses the same active SOCP pipeline.

| Folder under `checkpoints/` | Contents |
| --- | --- |
| `eps_0.1/` | `IBP_cnn.pt`, `dual_WK_cnn_ignore.pt`, `dual_tiny_gls.pt`, and five `tmp_*.pt` CNN checkpoints covering IBP, CROWN-IBP, DeepPoly, dual-WK, and dual training variants. |
| `eps_0.3/` | Five `*_cnn_standard.pt` MNIST checkpoints: CROWN-IBP, DeepPoly, dual, dual-WK, and IBP. |
| `cifar10/` | CIFAR-10 checkpoint groups for standard, GLS, REG, and IGNORE experiments. |
| `cifar10/eps_0.0088889/` | Five `cifar10_*_cnn5_standard.pt` checkpoints: CROWN-IBP, DeepPoly, dual, dual-WK, and IBP. |
| `cifar10/GLS/` | Gaussian/local smoothing experiment checkpoints, grouped by epsilon. |
| `cifar10/GLS/eps_0.0088889/` | Six `cifar10_*_cnn5_gls*.pt` files, including dual and `*_gls_test.pt` variants. |
| `cifar10/REG/` | Regularizer experiment checkpoints, including a nested `GLS/` group. |
| `cifar10/REG/eps_0.0088889/` | `cifar10_dual_cnn5_socp_standard_new.pt`. |
| `cifar10/REG/GLS/` | Combined regularizer and smoothing checkpoint group. |
| `cifar10/REG/GLS/eps_0.0088889/` | `cifar10_dual_cnn5_socp_gls_bad.pt`, `cifar10_dual_cnn5_socp_gls_test.pt`, and `cifar10_dual_wk_cnn5_socp_gls_test.pt`. |
| `cifar10/IGNORE/` | Additional experiment snapshots, with standard and `GLS/` subgroups. The folder name does not define a verifier mode. |
| `cifar10/IGNORE/eps_0.0088889/` | `cifar10_dual_cnn5_standard.pt`. |
| `cifar10/IGNORE/GLS/` | Additional smoothing snapshot group. |
| `cifar10/IGNORE/GLS/eps_0.0088889/` | `cifar10_dual_cnn5_gls.pt`. |

Checkpoint labels such as `tmp`, `test`, and `bad` are retained experiment
filenames, not measured performance guarantees. Choose the checkpoint you want
to evaluate and match its architecture parameters.

### Configuration files

| File(s) in `configs/` | Purpose |
| --- | --- |
| `mnist_lp.yaml` | MNIST CNN LP/dual training settings, regularizer settings, and smoothing options. |
| `mnist_lp_new.yaml`, `mnist_lp_old.yaml` | Alternative MNIST LP experiment settings. |
| `mnist_socp.yaml`, `mnist_socp_old.yaml` | SOCP-oriented training/loss settings, including model dimensions, solver/coupling budgets, and loss weights. |
| `cifar_lp.yaml` | CIFAR-10 CNN5 training settings, including architecture, epsilon schedule, and bound method. |
| `cifar_lp_crown.yaml` | CIFAR-10 CROWN-IBP training variant. |
| `cifar_lp_dual_wk.yaml` | CIFAR-10 dual-WK training variant. |
| `cifar_rgs_lp.yaml` | CIFAR-10 configuration containing randomized Gaussian smoothing options; inspect `use_rgs` for whether they are enabled. |

Training entry points are not included in this repository. These configurations
provide experiment context; a filename does not automatically select the
evaluator's model type or gamma backend.

### Dataset folders

| Folder | Contents |
| --- | --- |
| `data/MNIST/` | MNIST dataset cache. |
| `data/MNIST/raw/` | Training/test images and labels in IDX format, plus their compressed `.gz` copies: `train-images-idx3-ubyte`, `train-labels-idx1-ubyte`, `t10k-images-idx3-ubyte`, and `t10k-labels-idx1-ubyte`. |

### Source files

| File | Role |
| --- | --- |
| [`src/model.py`](src/model.py) | Builds the MNIST CNN, MNIST tiny MLP, and pooling-free CIFAR-10 CNN5. |
| [`src/bound_layers.py`](src/bound_layers.py) | Input-box bounds and interval propagation through convolution, linear, and ReLU layers. Required by verification. |
| [`src/dual_bounds.py`](src/dual_bounds.py) | Bound records and CROWN-style/Wong–Kolter backward routines reused by the verifier. Required even though these routines also support training. |
| [`src/deeppoly_bounds.py`](src/deeppoly_bounds.py) | Additional DeepPoly bound routines. The active SOCP evaluator does not import this file. |

| File in `src/SOCP/` | Role |
| --- | --- |
| `eval_SOCP_robustness.py` | Loads a dataset/checkpoint, evaluates the test subset, and reports metrics. |
| `socp_certificate.py` | Builds each sample's certificate and orchestrates target-class solves. |
| `socp_bounds.py` | Collects input, pre-activation, ReLU activation, and final-logit intervals. |
| `socp_influence.py` | Computes CROWN-style, dual-WK, or clean-gradient influence scores. |
| `socp_relaxation.py` | Selects sparse cross-layer (E), current-layer (S), and previous-layer (T) products. |
| `socp_solver.py` | Constructs the CVXPY SOCP relaxation and maximizes a target-versus-true margin. |
| `socp_solver_old.py` | Legacy solver available as a fallback for slow runs. |
| `socp_relaxation_old.py` | Legacy coupling-selector companion; currently identical to `socp_relaxation.py`. |
| `utils.py` | Device/seed helpers and conversion of supported model layers to dense affine matrices. |
| `__init__.py` | Python package marker. |
| `README.md` | Module-level notes from the earlier source-bundle review. Its attachment-era missing-file inventory is outdated; use this root README for the current repository inventory. |
| [`FLOWCHART.md`](src/SOCP/FLOWCHART.md) | Additional verifier pipeline and coupling-selection documentation. |

| File in `src/SOCP/Old_codes/` | Role |
| --- | --- |
| `socp_certificate_old.py` | Archived certificate orchestration. |
| `conv_to_dense_affines.py` | Older convolution-to-affine and linear-bound helpers. |
| `socp_cvxpy_layer_loss_cnn.py` | Differentiable CNN-head loss experiment using `cvxpylayers`. |
| `socp_diff_loss.py` | Unrolled adversarial-margin training surrogate; not a certified SOCP bound. |
| `socp_loss.py` | Training/logging loss wrapper around detached CVXPY certificate values. |

## Models and input domain

| CLI model type | Dataset/input shape | Builder |
| --- | --- | --- |
| `tiny` | MNIST, `1 × 28 × 28` | `build_mnist_tiny_model(n1, n2, linear_size)`: three linear layers with two hidden ReLUs; `linear_size` is unused. |
| `cnn` | MNIST, `1 × 28 × 28` | `build_mnist_model(n1, n2, linear_size)`: two convolution blocks and a hidden linear layer. |
| `cifar10` | CIFAR-10, `3 × 32 × 32` | `build_cnn5_model(c1, c2, c3, linear_size)`: five convolutions and two linear layers. CLI `n1/n2/n3` map to `c1/c2/c3`. |

The active conversion/bound routines support sequential models built from
`Conv2d`, `Linear`, `ReLU`, and flatten layers. They do not support arbitrary
PyTorch architectures or pooling layers.

The perturbation domain is an L-infinity ball clipped to **[0, 1]**:
`clamp(x - epsilon, 0, 1) <= x' <= clamp(x + epsilon, 0, 1)`.
Inputs and checkpoint preprocessing must match this domain. Mean/std-normalized
inputs require corresponding changes to the bounds. The missing loader prevents
verification of the current dataset preprocessing.

## Environment setup

Python 3.11 matches the environment specified in the supplied setup reference.
Install the evaluation dependencies in an environment suitable for your hardware:

```bash
conda create -n socp-verifier python=3.11 -y
conda activate socp-verifier
python -m pip install torch torchvision numpy tqdm cvxpy scs clarabel
```

Use a PyTorch build compatible with your hardware if you need CUDA. The evaluator
uses CUDA for PyTorch operations when available, while SCS/Clarabel perform the
conic solves on the CPU.

**Setup script status:** `scripts/create_env.sh` contains a malformed shebang
and uncommented explanatory text, including `Recommended:` and `GPU Version:`.
Treat it as an installation reference until those lines are corrected. The
archived CNN loss additionally needs `cvxpylayers`; evaluation does not.

## Integration requirements

The following must be resolved for a fresh checkout to run:

1. **Restore `src/data.py`.** The evaluator imports `get_mnist_loaders` and
   `get_cifar10_loaders`. Each must return `(train_loader, test_loader)` and
   accept the data directory plus `batch_size` and `test_batch_size`. Ensure
   deterministic test ordering and preprocessing consistent with the saved
   checkpoint and the [0, 1] input domain.
2. **Align shared-module imports with the launch layout.** The examples run
   from `src/`, where the evaluator imports `model`, `data`, and
   `dual_bounds` as top-level modules. However, `dual_bounds.py` uses
   `from .bound_layers import ...`, which fails in that layout.
   `socp_bounds.py` first tries the absent `SOCP.dual_bounds`, then falls back
   to top-level `dual_bounds`; both branches in `socp_influence.py` also
   import top-level `dual_bounds`. A consistent `src/` launch layout should
   use top-level `bound_layers` and `dual_bounds` imports in those modules.
   Alternatively, convert the entry point and its dependencies together to a
   consistent `src` package layout.
3. **Supply CIFAR-10 data through the loader.** Only MNIST raw files are
   committed. Configure the restored loader to download or find CIFAR-10
   under the chosen `--data_dir`.

Model builders, interval helpers, both legacy backend files, and pretrained
checkpoints are present. They are not missing from this checkout.

## Evaluate a checkpoint

After resolving the integration requirements, run evaluation from `src/`:

```bash
cd src
```

Architecture dimensions must match the saved state dictionary. The evaluator
accepts either a state dictionary directly or a dictionary containing it under
the `model` key. The examples use checkpoint paths present in the repository
and dimensions recorded in the supplied examples/configurations; checkpoint
loading has not been runtime-tested here.

### MNIST CNN

```bash
python -m SOCP.eval_SOCP_robustness \
  --checkpoint ../checkpoints/eps_0.3/ibp_cnn_standard.pt \
  --data_dir ../data --model_type cnn \
  --n1 16 --n2 32 --linear_size 100 --epsilon 0.3 \
  --max_E 8 --max_S 4 --max_T 4 --prev_candidate_limit 32 \
  --gamma_backend dual --all_classes \
  --solver SCS --solver_max_iters 200
```

For the tiny MNIST checkpoint, use `--model_type tiny` and set `--n1` and
`--n2` to its hidden widths. The CLI defaults differ from the tiny builder's
defaults, so pass the widths explicitly.

### CIFAR-10 CNN5

```bash
python -m SOCP.eval_SOCP_robustness \
  --checkpoint ../checkpoints/cifar10/eps_0.0088889/cifar10_deeppoly_cnn5_standard.pt \
  --data_dir ../data --model_type cifar10 \
  --n1 24 --n2 48 --n3 96 --linear_size 192 --epsilon 0.0088889 \
  --max_E 8 --max_S 4 --max_T 4 --prev_candidate_limit 16 \
  --gamma_backend dual --all_classes \
  --solver SCS --solver_max_iters 200
```

These are Bash commands; in PowerShell, place each evaluation command on one
line or replace the continuation backslashes with PowerShell backticks.

### Evaluation options

| Option | CLI default | Meaning |
| --- | --- | --- |
| `--checkpoint` | Required | Saved weights to evaluate. |
| `--data_dir` | `../data` | Dataset location, relative to the working directory. |
| `--model_type` | `tiny` | `tiny`/`cnn` select MNIST; `cifar10` selects CIFAR-10. |
| `--n1`, `--n2`, `--n3` | `16`, `32`, `96` | Architecture widths; `n3` is used only for CIFAR-10. |
| `--linear_size` | `100` | Hidden classifier width for CNN models. |
| `--epsilon` | Required | L-infinity radius in the model's input units. |
| `--nodes_per_layer` | `8` | Retained neuron-candidate budget per hidden layer. |
| `--max_E`, `--max_S`, `--max_T` | `8`, `6`, `6` | Per-layer coupling budgets for cross-layer, current-layer, and previous-layer products. |
| `--prev_candidate_limit` | `16` | Previous-layer candidates considered when scoring pairs. |
| `--gamma_backend` | `dual` | CROWN-style `dual`, fixed-alpha Wong–Kolter `dual_WK`, or clean-gradient `clean` influence scores. |
| `--solver` | `SCS` | Conic solver; `CLARABEL` is also handled explicitly. |
| `--solver_max_iters` | `200` | Iteration limit, not a wall-clock timeout. |
| `--solver_verbose` | Off | Print solver diagnostics. |
| `--all_classes` | Off | Check all nine incorrect classes for these ten-class models. |
| `--max_targets` | `4` | Top incorrect classes by clean logits when `--all_classes` is absent. |
| `--test_batch_size` | `9` | Loader batch size; each SOCP certificate still handles one image. |

These are evaluator defaults. Direct calls to `CertificateConfig` have
different defaults.

## Evaluation behavior and reported results

The evaluator processes exactly the first **200 test images** for either
dataset. It raises an error if the loader provides fewer images. Some source
comments/help strings still say “100” or “MNIST”; the executed limit is 200 and
`--model_type cifar10` selects CIFAR-10.

Cleanly misclassified images count as uncertified and skip the SOCP solve.
For each cleanly correct image, the verifier maximizes
`f_target(x') - f_true(x')` over the relaxation for each requested target.
A negative, accurately solved upper bound for every incorrect class is the
intended full multiclass robustness criterion.

The printed results include clean accuracy, certified accuracy, certification
rate among cleanly correct images, average worst margin, average certificate
time, variable/constraint counts, and solver statuses. Certificate time covers
the complete `certify_sample` call, including dense conversion and all targets.

**Interpretation of the current implementation:**

- Use `--all_classes` for full target coverage. The default four targets give
  only a partial-target result.
- The `certified` property checks only whether the largest returned margin is
  negative; it does not enforce an `optimal` solver status or full target
  coverage. Inspect numerical convergence and statuses before treating this
  flag as a validated certificate.
- On a solver exception, both solver versions retry with SCS at 500 iterations.
  The result's `solver` field retains the originally requested solver name.
- There is no PGD pre-check in this evaluation entry point.

## Slow CIFAR-10 runs: legacy backend

**If the code is running slowly on CIFAR-10, you can use
`socp_solver_old.py` and `socp_relaxation_old.py`.**

Both files are included under `src/SOCP/`. There is no CLI backend selector:
the certificate API imports the active names `socp_solver.py` and
`socp_relaxation.py`. Back up those active files and copy the legacy pair to
the active names before starting a new Python process.

From the repository root, in Bash:

```bash
mkdir -p .socp_backend_backup
cp -n src/SOCP/socp_solver.py .socp_backend_backup/socp_solver.py
cp -n src/SOCP/socp_relaxation.py .socp_backend_backup/socp_relaxation.py
cp src/SOCP/socp_solver_old.py src/SOCP/socp_solver.py
cp src/SOCP/socp_relaxation_old.py src/SOCP/socp_relaxation.py
```

Restart Python, return to `src/`, and rerun the evaluation command. Restore the
backed-up active implementation from the repository root with:

```bash
cp .socp_backend_backup/socp_solver.py src/SOCP/socp_solver.py
cp .socp_backend_backup/socp_relaxation.py src/SOCP/socp_relaxation.py
```

Keep the backup for the code version you are evaluating; refresh it deliberately
after changing the active implementation. The legacy selector is currently
identical to the active selector. The legacy solver omits the active solver's
residual constraints linking selected E products to activation squares, so this
switch changes the relaxation and may weaken the resulting bounds. A speedup
is not guaranteed or benchmarked here.

Both backends use dense convolution matrices, rebuilt for every sample.
CIFAR-10 can therefore remain expensive in memory, conversion time, and CVXPY
problem construction. Smaller coupling/candidate budgets can reduce work.
Reducing target coverage speeds up exploration but yields partial-target results.
Record the backend, budgets, solver settings, and evaluated subset when comparing
experiments.

## Files optional for a verification-only distribution

| Item | Recommendation |
| --- | --- |
| `src/SOCP/Old_codes/` | Archive or omit if only active checkpoint evaluation is required. Its loss modules are training experiments, and some archived imports still refer to their former locations. |
| `configs/` | Keep for experiment provenance; the evaluator does not require training configurations. |
| `src/deeppoly_bounds.py` | Optional for this evaluator, including when evaluating a DeepPoly-trained checkpoint. The checkpoint's training method does not select a DeepPoly verification backend. |
| MNIST training IDX files and compressed copies | The active evaluator uses the test loader. Whether training data can be omitted depends on how the restored loader constructs its returned loaders. Compressed copies are dataset cache material, not Python dependencies. |
| Extra checkpoint variants | Keep the models you intend to evaluate. Filename labels alone are insufficient to decide which experimental weights to discard. |
| `socp_solver_old.py`, `socp_relaxation_old.py` | Keep to preserve the documented legacy fallback. |
| `FLOWCHART.md` and module README | Optional runtime documentation. |

Retain `model.py`, `bound_layers.py`, `dual_bounds.py`, and all active verifier
modules. In particular, the shared bound files are verification dependencies,
even though their comments also discuss training. No files have been deleted.

## Repository scope

This repository contains the verifier, saved checkpoints, configuration
records, and archived loss experiments described above.
