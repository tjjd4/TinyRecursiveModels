"""
E1.8 Degradation Rate Analysis

Compares correct_flags from baseline z_raw.npz vs ablation z_raw.npz files
to compute how many originally-correct puzzles degraded under z_L reset conditions.

Both runs must use the same dataset, seed, and z_analysis_max_samples so that
correct_flags[i] refers to the same puzzle in every file.

Usage:
    python scripts/compute_degradation_rate.py \
        --baseline   path/to/baseline/z_raw.npz \
        --ablation_A path/to/reset_per_H_cycle/z_raw.npz \
        --ablation_B path/to/reset_per_step/z_raw.npz

At minimum one of --ablation_A or --ablation_B must be provided.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np


def load_flags(path: str) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    flags = data["correct_flags"].astype(bool)
    print(f"  Loaded {len(flags)} puzzles from {path}")
    return flags


def compute_degradation(baseline: np.ndarray, ablation: np.ndarray, name: str, ablation_path: str = "") -> dict:
    n = min(len(baseline), len(ablation))
    if len(baseline) != len(ablation):
        print(f"  WARNING: size mismatch — baseline={len(baseline)}, {name}={len(ablation)}. Truncating to {n}.")

    b = baseline[:n]
    a = ablation[:n]

    n_baseline_correct   = b.sum()
    n_baseline_incorrect = (~b).sum()

    # Correct → Incorrect (degraded)
    degraded = (b & ~a).sum()
    # Incorrect → Correct (recovered)
    recovered = (~b & a).sum()
    # Correct → Correct (stable correct)
    stable_correct = (b & a).sum()
    # Incorrect → Incorrect (stable incorrect)
    stable_incorrect = (~b & ~a).sum()

    degradation_rate = degraded / n_baseline_correct if n_baseline_correct > 0 else float("nan")
    recovery_rate    = recovered / n_baseline_incorrect if n_baseline_incorrect > 0 else float("nan")

    ablation_accuracy = a.mean()

    return dict(
        name=name,
        checkpoint=ablation_path,
        n_total=n,
        n_baseline_correct=int(n_baseline_correct),
        n_baseline_incorrect=int(n_baseline_incorrect),
        degraded=int(degraded),
        recovered=int(recovered),
        stable_correct=int(stable_correct),
        stable_incorrect=int(stable_incorrect),
        degradation_rate=float(degradation_rate),
        recovery_rate=float(recovery_rate),
        ablation_accuracy=float(ablation_accuracy),
    )


def print_result(r: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  {r['name']}")
    print(f"{'='*60}")
    print(f"  Checkpoint             : {r['checkpoint']}")
    print(f"  Total puzzles compared : {r['n_total']}")
    print(f"  Baseline correct       : {r['n_baseline_correct']}")
    print(f"  Baseline incorrect     : {r['n_baseline_incorrect']}")
    print(f"  Ablation accuracy      : {r['ablation_accuracy']:.4f}")
    print()
    print(f"  [Degradation] Correct → Incorrect : {r['degraded']} / {r['n_baseline_correct']} = {r['degradation_rate']:.2%}")
    print(f"  [Recovery]    Incorrect → Correct : {r['recovered']} / {r['n_baseline_incorrect']} = {r['recovery_rate']:.2%}")
    print(f"  Stable correct                    : {r['stable_correct']}")
    print(f"  Stable incorrect                  : {r['stable_incorrect']}")


def main():
    parser = argparse.ArgumentParser(description="E1.8 degradation rate analysis")
    parser.add_argument("--baseline",   required=True,  help="Path to baseline z_raw.npz")
    parser.add_argument("--ablation_A", default=None,   help="reset per H_cycle z_raw.npz")
    parser.add_argument("--ablation_B", default=None,   help="reset per step z_raw.npz")
    parser.add_argument("--save",       default=None,   help="Optional: save results as .npz to this path")
    args = parser.parse_args()

    if args.ablation_A is None and args.ablation_B is None:
        parser.error("At least one of --ablation_A or --ablation_B must be provided.")

    print("\nLoading files...")
    print(f"  Baseline checkpoint    : {args.baseline}")
    baseline = load_flags(args.baseline)
    abl_A = load_flags(args.ablation_A) if args.ablation_A else None
    abl_B = load_flags(args.ablation_B) if args.ablation_B else None

    results = []

    if abl_A is not None:
        r_A = compute_degradation(baseline, abl_A, "reset per H_cycle", args.ablation_A)
        results.append(r_A)
        print_result(r_A)

    if abl_B is not None:
        r_B = compute_degradation(baseline, abl_B, "reset per step", args.ablation_B)
        results.append(r_B)
        print_result(r_B)

    if len(results) == 2:
        r_A, r_B = results
        print(f"\n{'='*60}")
        print("  Comparison: reset per H_cycle vs reset per step")
        print(f"{'='*60}")
        ratio = r_A["degradation_rate"] / r_B["degradation_rate"] if r_B["degradation_rate"] > 0 else float("nan")
        print(f"  Degradation ratio A/B : {ratio:.2f}x")

    if args.save:
        save_dict = {}
        if args.ablation_A:
            save_dict["degradation_rate_A"] = results[0]["degradation_rate"]
            save_dict["degraded_A"] = results[0]["degraded"]
        if args.ablation_B:
            r_B = results[-1]
            save_dict["degradation_rate_B"] = r_B["degradation_rate"]
            save_dict["degraded_B"] = r_B["degraded"]
        np.savez(args.save, **save_dict)
        print(f"\nResults saved to {args.save}")


if __name__ == "__main__":
    main()
