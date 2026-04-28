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
    "ci_alpha": 0.20,
}

C_ZH = "#2166AC"   # blue  — z_H
C_ZL = "#D6604D"   # red-orange — z_L
C_DIFF = "#222222"
C_SIG = "#C72020"

K_VALUES = [2, 4, 6, 8, 10, 12, 14]

# (metric_key, ylabel, title)
PANELS = [
    ("accuracy",    "Exact Accuracy (%)",    "Exact Accuracy"),
    ("degradation", "Degradation Rate (%)",  "Degradation  (correct → incorrect)"),
    ("recovery",    "Recovery Rate (%)",     "Recovery  (incorrect → correct)"),
]

# (metric_key, ylabel, title) — paired diff panels (z_H − z_L)
DIFF_PANELS = [
    ("recovery",    r"$\Delta$ Recovery  ($z_H - z_L$, pp)",
                    "Channel Asymmetry: Recovery"),
    ("degradation", r"$\Delta$ Degradation  ($z_H - z_L$, pp)",
                    "Channel Asymmetry: Degradation"),
]


# ──────────────────────────────────────────────────────────────────────
# Data loading & bootstrap
# ──────────────────────────────────────────────────────────────────────

def load_flags(path: str) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    return data["correct_flags"].astype(bool)


def _stratified_indices(b: np.ndarray):
    cor_idx = np.where(b)[0]
    inc_idx = np.where(~b)[0]
    return cor_idx, inc_idx


def bootstrap_metrics(
    baseline: np.ndarray,
    ablation: np.ndarray,
    n_boot: int,
    alpha: float,
    seed: int,
    stratified: bool,
) -> dict:
    """
    Compute point estimates + bootstrap CI for recovery / degradation / accuracy.

    stratified=True : resample baseline-correct and baseline-incorrect strata
                       independently (CI conditional on fixed stratum sizes).
    stratified=False: resample 50048 puzzles uniformly.
    """
    n = min(len(baseline), len(ablation))
    if len(baseline) != len(ablation):
        print(f"  WARNING: size mismatch — truncating to {n}")
    b = baseline[:n]
    a = ablation[:n]

    n_correct   = int(b.sum())
    n_incorrect = int((~b).sum())

    # ── Point estimates ──
    recovery_count    = int((~b & a).sum())
    degradation_count = int((b & ~a).sum())
    point = {
        "recovery_rate"   : recovery_count    / n_incorrect if n_incorrect > 0 else float("nan"),
        "degradation_rate": degradation_count / n_correct   if n_correct   > 0 else float("nan"),
        "accuracy"        : float(a.mean()),
    }

    # ── Bootstrap loop ──
    rng = np.random.default_rng(seed)
    rec_boot = np.empty(n_boot)
    deg_boot = np.empty(n_boot)
    acc_boot = np.empty(n_boot)

    if stratified:
        cor_idx, inc_idx = _stratified_indices(b)
        for i in range(n_boot):
            boot_cor = rng.choice(cor_idx, size=n_correct,   replace=True) if n_correct   > 0 else np.empty(0, dtype=int)
            boot_inc = rng.choice(inc_idx, size=n_incorrect, replace=True) if n_incorrect > 0 else np.empty(0, dtype=int)
            rec_boot[i] = a[boot_inc].mean()    if n_incorrect > 0 else np.nan
            deg_boot[i] = (~a[boot_cor]).mean() if n_correct   > 0 else np.nan
            # Accuracy under stratified resampling: weight by fixed stratum sizes
            acc_boot[i] = (a[boot_cor].sum() + a[boot_inc].sum()) / n
    else:
        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)
            bb = b[idx]
            aa = a[idx]
            ni = int((~bb).sum())
            nc = int(bb.sum())
            rec_boot[i] = (~bb & aa).sum() / ni if ni > 0 else np.nan
            deg_boot[i] = (bb & ~aa).sum() / nc if nc > 0 else np.nan
            acc_boot[i] = aa.mean()

    out = {
        **point,
        "recovery_count"    : recovery_count,
        "degradation_count" : degradation_count,
        "n_incorrect"       : n_incorrect,
        "n_correct"         : n_correct,
    }
    for name, arr in [("recovery_rate", rec_boot),
                      ("degradation_rate", deg_boot),
                      ("accuracy", acc_boot)]:
        out[f"{name}_lower"] = float(np.nanquantile(arr, alpha / 2))
        out[f"{name}_upper"] = float(np.nanquantile(arr, 1 - alpha / 2))
    return out


def bootstrap_diff(
    baseline: np.ndarray,
    abl_zH: np.ndarray,
    abl_zL: np.ndarray,
    n_boot: int,
    alpha: float,
    seed: int,
    stratified: bool,
) -> dict:
    """
    Paired bootstrap for z_H − z_L diff in recovery and degradation.

    z_H and z_L ablations target the same 50048 puzzles, so the diff
    must use shared resampling indices (paired structure).
    """
    n = min(len(baseline), len(abl_zH), len(abl_zL))
    b  = baseline[:n]
    aH = abl_zH[:n]
    aL = abl_zL[:n]

    n_correct   = int(b.sum())
    n_incorrect = int((~b).sum())

    # ── Point diff (full data) ──
    rec_H_full = aH[~b].mean()    if n_incorrect > 0 else np.nan
    rec_L_full = aL[~b].mean()    if n_incorrect > 0 else np.nan
    deg_H_full = (~aH[b]).mean()  if n_correct   > 0 else np.nan
    deg_L_full = (~aL[b]).mean()  if n_correct   > 0 else np.nan

    # ── Bootstrap loop ──
    rng = np.random.default_rng(seed)
    rec_diff = np.empty(n_boot)
    deg_diff = np.empty(n_boot)

    if stratified:
        cor_idx, inc_idx = _stratified_indices(b)
        for i in range(n_boot):
            boot_cor = rng.choice(cor_idx, size=n_correct,   replace=True) if n_correct   > 0 else np.empty(0, dtype=int)
            boot_inc = rng.choice(inc_idx, size=n_incorrect, replace=True) if n_incorrect > 0 else np.empty(0, dtype=int)
            rec_diff[i] = (aH[boot_inc].mean()   - aL[boot_inc].mean())   if n_incorrect > 0 else np.nan
            deg_diff[i] = ((~aH[boot_cor]).mean() - (~aL[boot_cor]).mean()) if n_correct > 0 else np.nan
    else:
        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)
            bb = b[idx]
            ni = (~bb).sum()
            nc = bb.sum()
            if ni > 0:
                rec_diff[i] = aH[idx][~bb].mean() - aL[idx][~bb].mean()
            else:
                rec_diff[i] = np.nan
            if nc > 0:
                deg_diff[i] = (~aH[idx][bb]).mean() - (~aL[idx][bb]).mean()
            else:
                deg_diff[i] = np.nan

    return {
        "recovery_diff_point"   : float(rec_H_full - rec_L_full),
        "recovery_diff_lower"   : float(np.nanquantile(rec_diff, alpha / 2)),
        "recovery_diff_upper"   : float(np.nanquantile(rec_diff, 1 - alpha / 2)),
        "degradation_diff_point": float(deg_H_full - deg_L_full),
        "degradation_diff_lower": float(np.nanquantile(deg_diff, alpha / 2)),
        "degradation_diff_upper": float(np.nanquantile(deg_diff, 1 - alpha / 2)),
    }


# ──────────────────────────────────────────────────────────────────────
# Per-channel and per-diff aggregation across k
# ──────────────────────────────────────────────────────────────────────

def gather(
    baseline_flags: np.ndarray,
    paths: dict,
    *,
    n_boot: int,
    alpha: float,
    stratified: bool,
) -> dict:
    """
    Aggregate per-channel point + CI across reset steps k.
    Returns dict with 'k' and, for each metric, the arrays:
        {metric}, {metric}_lower, {metric}_upper  (all in percentage units).
    Also returns 'flags_by_k': raw (50048,) bool arrays — needed for diff bootstrap.
    """
    out = {
        "k"                   : [],
        "recovery"            : [], "recovery_lower"   : [], "recovery_upper"   : [],
        "degradation"         : [], "degradation_lower": [], "degradation_upper": [],
        "accuracy"            : [], "accuracy_lower"   : [], "accuracy_upper"   : [],
        "flags_by_k"          : {},
    }
    for k in K_VALUES:
        path = paths.get(k)
        if path is None:
            continue
        abl_flags = load_flags(path)
        m = bootstrap_metrics(baseline_flags, abl_flags,
                              n_boot=n_boot, alpha=alpha,
                              seed=42 + k, stratified=stratified)
        out["k"].append(k)
        for key, src in [("recovery"   , "recovery_rate"),
                         ("degradation", "degradation_rate"),
                         ("accuracy"   , "accuracy")]:
            out[key          ].append(m[src             ] * 100)
            out[f"{key}_lower"].append(m[f"{src}_lower"] * 100)
            out[f"{key}_upper"].append(m[f"{src}_upper"] * 100)
        out["flags_by_k"][k] = abl_flags

        print(f"  k={k:2d}  rec={m['recovery_rate']*100:5.2f} "
              f"[{m['recovery_rate_lower']*100:5.2f},{m['recovery_rate_upper']*100:5.2f}]  "
              f"deg={m['degradation_rate']*100:5.2f} "
              f"[{m['degradation_rate_lower']*100:5.2f},{m['degradation_rate_upper']*100:5.2f}]  "
              f"acc={m['accuracy']*100:5.2f} "
              f"[{m['accuracy_lower']*100:5.2f},{m['accuracy_upper']*100:5.2f}]  "
              f"({m['recovery_count']}/{m['n_incorrect']} recovered)")
    return out


def gather_diff(
    baseline_flags: np.ndarray,
    zH: dict,
    zL: dict,
    *,
    n_boot: int,
    alpha: float,
    stratified: bool,
) -> dict:
    """
    Per-k paired diff (z_H − z_L) for recovery and degradation, in percentage points.
    Only computed for k present in both zH and zL.
    """
    common_k = [k for k in zH["flags_by_k"] if k in zL["flags_by_k"]]
    out = {
        "k": [],
        "recovery"          : [], "recovery_lower"   : [], "recovery_upper"   : [],
        "degradation"       : [], "degradation_lower": [], "degradation_upper": [],
    }
    for k in common_k:
        d = bootstrap_diff(
            baseline_flags,
            zH["flags_by_k"][k],
            zL["flags_by_k"][k],
            n_boot=n_boot, alpha=alpha,
            seed=4242 + k, stratified=stratified,
        )
        out["k"].append(k)
        for key, prefix in [("recovery", "recovery_diff"), ("degradation", "degradation_diff")]:
            out[key            ].append(d[f"{prefix}_point"] * 100)
            out[f"{key}_lower"].append(d[f"{prefix}_lower"] * 100)
            out[f"{key}_upper"].append(d[f"{prefix}_upper"] * 100)

        sig_rec = (d["recovery_diff_lower"] > 0) or (d["recovery_diff_upper"] < 0)
        sig_deg = (d["degradation_diff_lower"] > 0) or (d["degradation_diff_upper"] < 0)
        print(f"  k={k:2d}  Δrec={d['recovery_diff_point']*100:+5.2f} "
              f"[{d['recovery_diff_lower']*100:+5.2f},{d['recovery_diff_upper']*100:+5.2f}]{'*' if sig_rec else ' '}  "
              f"Δdeg={d['degradation_diff_point']*100:+5.2f} "
              f"[{d['degradation_diff_lower']*100:+5.2f},{d['degradation_diff_upper']*100:+5.2f}]{'*' if sig_deg else ' '}")
    return out


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────

def _setup_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


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
    """Per-channel rate plot with shaded 95% CI band."""
    matplotlib.rcParams.update({"font.family": STYLE["font_family"]})
    lw = STYLE["linewidth"]
    ms = STYLE["markersize"]

    fig, ax = plt.subplots(figsize=(STYLE["fig_w"], STYLE["fig_h"]),
                           dpi=STYLE["dpi"])
    _setup_axes(ax)

    if key == "accuracy":
        ax.axhline(baseline_accuracy * 100, color="#555555",
                   linestyle=":", linewidth=1.4, alpha=0.7,
                   label=f"Baseline ({baseline_accuracy*100:.1f}%)")
    else:
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="-", alpha=0.4)

    for src, color, label, present in [(zH, C_ZH, "reset z_H", has_zH),
                                        (zL, C_ZL, "reset z_L", has_zL)]:
        if not present:
            continue
        ax.fill_between(src["k"], src[f"{key}_lower"], src[f"{key}_upper"],
                        color=color, alpha=STYLE["ci_alpha"], linewidth=0)
        ax.plot(src["k"], src[key], color=color, lw=lw,
                marker="o", ms=ms, label=label)

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


def make_diff_panel(
    key: str,
    ylabel: str,
    title: str,
    diff: dict,
) -> plt.Figure:
    """Paired diff (z_H − z_L) with shaded 95% CI band and significance markers."""
    matplotlib.rcParams.update({"font.family": STYLE["font_family"]})
    lw = STYLE["linewidth"]
    ms = STYLE["markersize"]

    fig, ax = plt.subplots(figsize=(STYLE["fig_w"], STYLE["fig_h"]),
                           dpi=STYLE["dpi"])
    _setup_axes(ax)

    ax.axhline(0, color="grey", linewidth=1.0, linestyle="--",
               alpha=0.7, label="Symmetry (null)")

    ks      = np.asarray(diff["k"])
    points  = np.asarray(diff[key])
    lowers  = np.asarray(diff[f"{key}_lower"])
    uppers  = np.asarray(diff[f"{key}_upper"])

    ax.fill_between(ks, lowers, uppers,
                    color=C_DIFF, alpha=STYLE["ci_alpha"], linewidth=0,
                    label="95% CI (paired bootstrap)")
    ax.plot(ks, points, color=C_DIFF, lw=lw,
            marker="o", ms=ms, label=r"$z_H - z_L$")

    # CI 不跨 0 → 顯著差異
    sig = (lowers > 0) | (uppers < 0)
    if sig.any():
        ax.scatter(ks[sig], points[sig], marker="*",
                   s=ms * ms * 5, color=C_SIG, zorder=5,
                   edgecolors="white", linewidths=0.6,
                   label="CI excludes 0")

    ax.set_xlabel("Reset step  k", fontsize=STYLE["label_fs"])
    ax.set_ylabel(ylabel, fontsize=STYLE["label_fs"])
    ax.set_title(title, fontsize=STYLE["title_fs"])
    ax.set_xticks(K_VALUES)
    ax.set_xticklabels([str(k) for k in K_VALUES], fontsize=STYLE["tick_fs"])
    ax.tick_params(axis="y", labelsize=STYLE["tick_fs"])
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.1f"))
    ax.legend(fontsize=STYLE["legend_fs"], framealpha=0.9)

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────

def plot_degradation_curve(
    baseline_flags: np.ndarray,
    zH_paths: dict,
    zL_paths: dict,
    baseline_accuracy: float,
    *,
    n_boot: int,
    alpha: float,
    stratified: bool,
    save_dir: str | None,
) -> list[plt.Figure]:
    print(f"\n=== z_H reset (n_boot={n_boot}, stratified={stratified}, "
          f"CI={(1 - alpha) * 100:.0f}%) ===")
    zH = gather(baseline_flags, zH_paths,
                n_boot=n_boot, alpha=alpha, stratified=stratified)
    print("\n=== z_L reset ===")
    zL = gather(baseline_flags, zL_paths,
                n_boot=n_boot, alpha=alpha, stratified=stratified)

    has_zH = len(zH["k"]) > 0
    has_zL = len(zL["k"]) > 0
    if not has_zH and not has_zL:
        raise ValueError("No valid ablation checkpoints found.")

    diff = None
    if has_zH and has_zL:
        print("\n=== Paired diff  (z_H − z_L) ===")
        diff = gather_diff(baseline_flags, zH, zL,
                           n_boot=n_boot, alpha=alpha, stratified=stratified)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    figs = []

    # ── Per-channel rate panels ──
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

    # ── Diff panels (only if both channels present) ──
    if diff is not None:
        for key, ylabel, title in DIFF_PANELS:
            fig = make_diff_panel(key, ylabel, title, diff)
            if save_dir:
                out = os.path.join(save_dir, f"diff_{key}.png")
                fig.savefig(out, bbox_inches="tight")
                print(f"  Saved: {out}")
            else:
                plt.show()
            figs.append(fig)
            plt.close(fig)

    return figs


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Degradation / recovery / accuracy vs reset step k, "
                    "with 95% bootstrap CI and z_H − z_L paired-diff panels."
    )
    p.add_argument("--baseline", required=True)
    for ch in ("zH", "zL"):
        for k in K_VALUES:
            p.add_argument(f"--{ch}_k{k}", default=None,
                           help=f"z_raw.npz for {ch} reset at step k={k}")
    p.add_argument("--save_dir", default=None,
                   help="Output directory. Five PNGs are written inside "
                        "(accuracy / degradation / recovery / diff_recovery / diff_degradation). "
                        "If omitted, plt.show() is called.")
    p.add_argument("--n_boot", type=int, default=1000,
                   help="Bootstrap iterations (default 1000).")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Significance level; 0.05 → 95%% CI (default 0.05).")
    p.add_argument("--unconditional", action="store_true",
                   help="Use unconditional bootstrap instead of stratified "
                        "(stratified is the default and recommended for fixed test sets).")
    return p


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
        n_boot=args.n_boot,
        alpha=args.alpha,
        stratified=not args.unconditional,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()