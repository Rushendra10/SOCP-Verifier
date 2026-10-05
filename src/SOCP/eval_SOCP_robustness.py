import argparse
import time
from collections import defaultdict
from typing import Any, Dict

import torch
from tqdm import tqdm

from SOCP.socp_certificate import CertificateConfig, certify_sample
from SOCP.utils import get_device
from data import get_mnist_loaders, get_cifar10_loaders
from model import build_mnist_model, build_mnist_tiny_model, build_cnn5_model

"""
        Full SOCP certified robustness evaluation.
        run code:
            python -m SOCP.eval_SOCP_robustness --checkpoint ../checkpoints/tmp_IBP_tiny.pt --model_type tiny --epsilon 0.1 --max_E 8 --max_S 4 --max_T 4 --prev_candidate_limit 16
            python -m SOCP.eval_SOCP_robustness --checkpoint ../checkpoints/eps_0.3/ibp_cnn_standard.pt --model_type cnn --epsilon 0.3 --max_E 8 --max_S 4 --max_T 4 --prev_candidate_limit 32 --gamma_backend dual --max_targets 8
            
            nohup python -m SOCP.eval_SOCP_robustness --checkpoint ../checkpoints/eps_0.1/tmp_deeppoly_cnn.pt --model_type cnn --epsilon 0.145\
                                                    --linear_size 100 --max_E 16 --max_S 8 --max_T 8\
                                                    --prev_candidate_limit 32 --gamma_backend dual --max_targets 9\
                                                    > MNIST_tmp_deeppoly_cnn_train0.1_Val0.15.out 2>&1 &
            
            
            
            python -m SOCP.eval_SOCP_robustness --checkpoint ../checkpoints/cifar10/eps_0.0088889/cifar10_deeppoly_cnn5_standard.pt --model_type cifar10 --epsilon 0.0088889 --n1 24 --n2 48 --n3 96 --linear_size 192 --max_E 8 --max_S 4 --max_T 4 --prev_candidate_limit 16 --gamma_backend dual --max_targets 6
            
            nohup python -m SOCP.eval_SOCP_robustness --checkpoint ../checkpoints/cifar10/eps_0.0088889/cifar10_deeppoly_cnn5_standard.pt --model_type cifar10 --epsilon 0.0039\
                            --n1 24 --n2 48 --n3 96 --linear_size 192 --max_E 8 --max_S 4 --max_T 4\
                            --prev_candidate_limit 16 --gamma_backend dual --max_targets 6\
                            > cifar10_deeppoly_cnn5_standard_eps_0.0039.out 2>&1 &
        Returns
        -------
        dict containing all paper metrics.
    """

# This evaluator intentionally processes only the first 100 test images.
MAX_TEST_IMAGES = 200


def _safe_average(values) -> float:
    """Return the arithmetic mean, or 0.0 when the list is empty."""
    return float(sum(values) / len(values)) if values else 0.0


def evaluate_socp_full(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    cert_cfg: CertificateConfig,
) -> Dict[str, Any]:
    """
    Evaluate SOCP-certified robustness on exactly the first 100 test images.

    Cleanly misclassified images are counted as not certified and are not sent
    to the SOCP solver. The progress bar advances once per image rather than
    once per DataLoader batch.
    """
    model.eval()

    total = 0
    clean_correct = 0
    certified = 0

    worst_margins = []
    solve_times = []
    num_variables = []
    num_constraints = []
    solver_status = defaultdict(int)

    with tqdm(
        total=MAX_TEST_IMAGES,
        desc="SOCP Robustness Evaluation",
        unit="image",
    ) as pbar:
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            for i in range(x.size(0)):
                # Stop before processing image 101.
                if total >= MAX_TEST_IMAGES:
                    break

                xi = x[i : i + 1]
                yi = int(y[i].item())
                total += 1

                # Clean prediction does not require gradients.
                with torch.inference_mode():
                    logits = model(xi)
                    pred = int(logits.argmax(dim=1).item())

                last_solve_time = 0.0
                targets_solved = 0

                if pred == yi:
                    clean_correct += 1

                    # Keep this call outside torch.no_grad()/inference_mode().
                    # Some gamma backends may require gradient-enabled code.
                    start = time.perf_counter()
                    cert = certify_sample(
                        model=model,
                        x=xi,
                        y=yi,
                        cfg=cert_cfg,
                    )
                    last_solve_time = time.perf_counter() - start

                    solve_times.append(last_solve_time)
                    worst_margins.append(float(cert.worst_margin))
                    targets_solved = len(cert.margins)

                    if cert.certified:
                        certified += 1

                    for result in cert.margins.values():
                        solver_status[str(result.status)] += 1
                        num_variables.append(int(result.num_variables))
                        num_constraints.append(int(result.num_constraints))

                clean_acc = clean_correct / total
                cert_acc = certified / total

                pbar.set_postfix(
                    clean=f"{clean_acc:.4f}",
                    cert=f"{cert_acc:.4f}",
                    targets=targets_solved,
                    last=f"{last_solve_time:.1f}s",
                )
                pbar.update(1)

            if total >= MAX_TEST_IMAGES:
                break

    if total < MAX_TEST_IMAGES:
        raise RuntimeError(
            f"The test loader ended after {total} images; "
            f"{MAX_TEST_IMAGES} images were required."
        )

    results = {
        "total_test_images": total,
        "clean_accuracy": clean_correct / total,
        "certified_accuracy": certified / total,
        "certification_rate": certified / max(clean_correct, 1),
        "avg_worst_margin": _safe_average(worst_margins),
        "avg_solve_time": _safe_average(solve_times),
        "avg_num_variables": _safe_average(num_variables),
        "avg_num_constraints": _safe_average(num_constraints),
        "solver_status": dict(solver_status),
    }

    print()
    print("=" * 60)
    print("SOCP CERTIFIED ROBUSTNESS RESULTS")
    print("=" * 60)
    print(f"Total Test Images        : {results['total_test_images']}")
    print(f"Clean Accuracy           : {results['clean_accuracy']:.4%}")
    print(f"Certified Accuracy       : {results['certified_accuracy']:.4%}")
    print(f"Certification Rate       : {results['certification_rate']:.4%}")
    print(f"Average Worst Margin     : {results['avg_worst_margin']:.6f}")
    print(f"Average Solve Time       : {results['avg_solve_time']:.4f} sec")
    print(f"Average Variables        : {results['avg_num_variables']:.1f}")
    print(f"Average Constraints      : {results['avg_num_constraints']:.1f}")

    print("\nSolver Status Counts")
    if results["solver_status"]:
        for status, count in sorted(results["solver_status"].items()):
            print(f"    {status:<20} {count}")
    else:
        print("    No SOCP problems were solved.")

    print("=" * 60)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SOCP certified robustness on the first 100 MNIST test images."
    )

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="../data")

    # A batch size of 1 gives the clearest per-image progress display.
    # The evaluation loop still enforces a hard limit of 100 images.
    parser.add_argument("--test_batch_size", type=int, default=9)

    parser.add_argument(
        "--model_type",
        choices=["cnn", "tiny", "cifar10"],
        default="tiny",
    )
    parser.add_argument("--n1", type=int, default=16)
    parser.add_argument("--n2", type=int, default=32)
    parser.add_argument("--n3", type=int, default=96)
    parser.add_argument("--linear_size", type=int, default=100)

    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--nodes_per_layer", type=int, default=8)
    parser.add_argument("--max_E", type=int, default=8)
    parser.add_argument("--max_S", type=int, default=6)
    parser.add_argument("--max_T", type=int, default=6)
    parser.add_argument("--prev_candidate_limit", type=int, default=16)

    parser.add_argument("--solver", type=str, default="SCS") #SCS CLARABEL
    parser.add_argument("--solver_max_iters", type=int, default=200)
    parser.add_argument("--solver_verbose", action="store_true")

    parser.add_argument(
        "--all_classes",
        action="store_true",
        help="Verify all nine incorrect MNIST classes.",
    )
    parser.add_argument(
        "--max_targets",
        type=int,
        default=4,
        help="Number of target classes when --all_classes is not used.",
    )
    parser.add_argument(
        "--gamma_backend",
        choices=["dual_WK", "dual", "clean"],
        default="dual",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.test_batch_size < 1:
        raise ValueError("--test_batch_size must be at least 1.")

    if args.max_targets < 1:
        raise ValueError("--max_targets must be at least 1.")

    device = get_device()
    print(f"Using device: {device}")
    print(f"Evaluating exactly the first {MAX_TEST_IMAGES} test images.")

    
    if(args.model_type == "cifar10"):
        _, test_loader = get_cifar10_loaders(
                args.data_dir,
                batch_size=1,
                test_batch_size=args.test_batch_size,
            )
    else:
        _, test_loader = get_mnist_loaders(
                args.data_dir,
                batch_size=1,
                test_batch_size=args.test_batch_size,
            )

    if args.model_type == "tiny":
        model = build_mnist_tiny_model(
            args.n1,
            args.n2,
            args.linear_size,
        )
    elif args.model_type == "cifar10":
        model = build_cnn5_model(
            c1=args.n1,
            c2=args.n2,
            c3=args.n3,
            linear_size=args.linear_size,
            num_classes=10,
        )
    else:
        model = build_mnist_model(
            args.n1,
            args.n2,
            args.linear_size,
        )
    
    model = model.to(device)
    
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )
    state_dict = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )

    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")
    #check clean accuracy
    correct = 0
    seen = 0
    model.eval()

    with torch.no_grad():
        for x, y in test_loader:
            remaining = 200 - seen
            x = x[:remaining]
            y = y[:remaining]

            predictions = model(x.to(device)).argmax(dim=1).cpu()
            correct += (predictions == y).sum().item()
            seen += len(y)

            if seen == 200:
                break

    if seen != 200:
        raise RuntimeError(f"Expected 200 test images, but found {seen}")

    print(f"Clean accuracy on first 200 images: {correct}/200 = {correct / 200:.2%}")
    
    cert_cfg = CertificateConfig(
        epsilon=args.epsilon,
        nodes_per_layer=args.nodes_per_layer,
        max_E=args.max_E,
        max_S=args.max_S,
        max_T=args.max_T,
        prev_candidate_limit=args.prev_candidate_limit,
        solver=args.solver,
        solver_max_iters=args.solver_max_iters,
        solver_verbose=args.solver_verbose,
        all_classes=args.all_classes,
        max_targets=args.max_targets,
        gamma_backend=args.gamma_backend,
    )

    evaluate_socp_full(
        model=model,
        test_loader=test_loader,
        device=device,
        cert_cfg=cert_cfg,
    )


if __name__ == "__main__":
    main()
