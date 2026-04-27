import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


STYLE = {
    "fig_w": 5.5,
    "fig_h": 4.2,
    "dpi": 150,
    "font_family": "DejaVu Sans",
    "title_fs": 11,
    "label_fs": 10,
    "tick_fs": 9,
    "legend_fs": 9,
    "linewidth": 2.2,
    "markersize": 7,
}

C_ZH = "#2166AC"   # blue  — z_H
C_ZL = "#D6604D"   # red-orange — z_L

K_VALUES = [2, 4, 8, 12]

PANELS = [
    ("accuracy",    "Exact Accuracy (%)",    "Exact Accuracy"),
    ("degradation", "Degradation Rate (%)",  "Degradation  (correct → incorrect)"),
    ("recovery",    "Recovery Rate (%)",     "Recovery  (incorrect → correct)"),
]


def load_flags(path: str) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    return data["correct_flags"].astype(bool)


def compute_metrics(baseline: np.ndarray, ablation: np.ndarray) -> dict:
    n = min(len(baseline), len(ablation))
    if len(baseline) != len(ablation):
        print(f"  WARNING: size mismatch — truncating to {n}")
    b = baseline[:n]
    a = ablation[:n]

    n_correct   = b.sum()
    n_incorrect = (~b).sum()
    recovery_count   = (~b & a).sum()
    degradation_count = (b & ~a).sum()

    return {
        "recovery_rate"   : float(recovery_count   / n_incorrect) if n_incorrect > 0 else float("nan"),
        "degradation_rate": float(degradation_count / n_correct)  if n_correct   > 0 else float("nan"),
        "accuracy"        : float(a.mean()),
        "recovery_count"   : int(recovery_count),
        "degradation_count": int(degradation_count),
        "n_incorrect"      : int(n_incorrect),
        "n_correct"        : int(n_correct),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Degradation / recovery / accuracy vs reset step k")
    p.add_argument("--baseline", required=True)
    for ch in ("zH", "zL"):
        for k in K_VALUES:
            p.add_argument(f"--{ch}_k{k}", default=None,
                           help=f"z_raw.npz for {ch} reset at step k={k}")
    p.add_argument("--save_dir", default=None,
                   help="Output directory. Three PNGs are written inside. "
                        "If omitted, plt.show() is called.")
    return p


def gather(baseline_flags: np.ndarray, paths: dict) -> dict:
    ks, recovery, degradation, accuracy = [], [], [], []
    for k in K_VALUES:
        path = paths.get(k)
        if path is None:
            continue
        abl_flags = load_flags(path)
        m = compute_metrics(baseline_flags, abl_flags)
        ks.append(k)
        recovery.append(m["recovery_rate"]    * 100)
        degradation.append(m["degradation_rate"] * 100)
        accuracy.append(m["accuracy"]            * 100)
        print(f"  k={k:2d}  recovery={m['recovery_rate']:.2%}  "
              f"degradation={m['degradation_rate']:.2%}  acc={m['accuracy']:.3f}  "
              f"({m['recovery_count']}/{m['n_incorrect']} recovered)")
    return {"k": ks, "recovery": recovery, "degradation": degradation, "accuracy": accuracy}


def make_panel(
    key: str,
    ylabel: str,
    title: str,
    zH: dict,
    zL: dict,
    has_zH: bool,
    has_zL: bool,
    baseline_accuracy: float,
) -> plt.Figure:
    matplotlib.rcParams.update({"font.family": STYLE["font_family"]})
    lw = STYLE["linewidth"]
    ms = STYLE["markersize"]

    fig, ax = plt.subplots(figsize=(STYLE["fig_w"], STYLE["fig_h"]),
                           dpi=STYLE["dpi"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if key == "accuracy":
        ax.axhline(baseline_accuracy * 100, color="#555555",
                   linestyle=":", linewidth=1.4, alpha=0.7,
                   label=f"Baseline ({baseline_accuracy*100:.1f}%)")
    else:
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="-", alpha=0.4)

    if has_zH:
        ax.plot(zH["k"], zH[key], color=C_ZH, lw=lw,
                marker="o", ms=ms, label="reset z_H")
    if has_zL:
        ax.plot(zL["k"], zL[key], color=C_ZL, lw=lw,
                marker="o", ms=ms, label="reset z_L")

    ax.set_xlabel("Reset step  k", fontsize=STYLE["label_fs"])
    ax.set_ylabel(ylabel, fontsize=STYLE["label_fs"])
    ax.set_title(title, fontsize=STYLE["title_fs"])
    ax.set_xticks(K_VALUES)
    ax.set_xticklabels([str(k) for k in K_VALUES], fontsize=STYLE["tick_fs"])
    ax.tick_params(axis="y", labelsize=STYLE["tick_fs"])
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.legend(fontsize=STYLE["legend_fs"], framealpha=0.9)

    if key != "accuracy":
        ymin, _ = ax.get_ylim()
        ax.set_ylim(max(ymin, -1.0), None)

    fig.tight_layout()
    return fig


def plot_degradation_curve(
    baseline_flags: np.ndarray,
    zH_paths: dict,
    zL_paths: dict,
    baseline_accuracy: float,
    save_dir: str | None = None,
) -> list[plt.Figure]:
    print("\n=== z_H reset ===")
    zH = gather(baseline_flags, zH_paths)
    print("\n=== z_L reset ===")
    zL = gather(baseline_flags, zL_paths)

    has_zH = len(zH["k"]) > 0
    has_zL = len(zL["k"]) > 0
    if not has_zH and not has_zL:
        raise ValueError("No valid ablation checkpoints found.")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    figs = []
    for key, ylabel, title in PANELS:
        fig = make_panel(key, ylabel, title, zH, zL,
                         has_zH, has_zL, baseline_accuracy)
        if save_dir:
            out = os.path.join(save_dir, f"{key}.png")
            fig.savefig(out, bbox_inches="tight")
            print(f"  Saved: {out}")
        else:
            plt.show()
        figs.append(fig)
        plt.close(fig)

    return figs


def main():
    parser = build_parser()
    args = parser.parse_args()

    print(f"\nLoading baseline: {args.baseline}")
    baseline_flags    = load_flags(args.baseline)
    baseline_accuracy = baseline_flags.mean()
    print(f"  Baseline accuracy: {baseline_accuracy:.4f}  "
          f"(correct={baseline_flags.sum()}, incorrect={(~baseline_flags).sum()})")

    zH_paths = {k: getattr(args, f"zH_k{k}") for k in K_VALUES}
    zL_paths = {k: getattr(args, f"zL_k{k}") for k in K_VALUES}

    plot_degradation_curve(
        baseline_flags=baseline_flags,
        zH_paths=zH_paths,
        zL_paths=zL_paths,
        baseline_accuracy=baseline_accuracy,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
