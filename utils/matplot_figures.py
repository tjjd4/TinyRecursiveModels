from typing import List, Tuple, Optional
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

from utils.math import linear_cka

# individual plot functions (each returns fig for wandb logging)

def plot_pca_split(proj, sample_ids, flags, pca, save_dir, n_show=60, z_label="z_H"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    correct_local = [i for i, f in enumerate(flags) if f]
    incorrect_local = [i for i, f in enumerate(flags) if not f]

    for ax, local_idxs, color, title in [
        (axes[0], correct_local[:n_show], "steelblue", "Correct"),
        (axes[1], incorrect_local[:n_show], "firebrick", "Incorrect"),
    ]:
        for li in local_idxs:
            pts = proj[sample_ids == li]
            T = pts.shape[0]
            alphas = np.linspace(0.2, 1.0, max(T, 2))
            for t in range(T - 1):
                ax.plot(pts[t:t+2, 0], pts[t:t+2, 1],
                        color=color, alpha=float(alphas[t]), lw=0.9)
            ax.scatter(pts[0, 0],  pts[0, 1],  color=color, s=18, alpha=0.5, marker="o", zorder=3)
            ax.scatter(pts[-1, 0], pts[-1, 1], color=color, s=40, alpha=0.9, marker="*", zorder=4)
        ax.set_title(f"{title}  (n={len(local_idxs)})", fontsize=11)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.grid(True, lw=0.3, alpha=0.5)

    plt.suptitle(f"{z_label}  PCA trajectories  ○=step1  ★=final", fontsize=11)
    plt.tight_layout()
    return fig


def plot_pca_combined(proj, sample_ids, flags, pca, proj_inits, save_dir, n_show=80, z_label="z_H"):
    fig, ax = plt.subplots(figsize=(8, 7))
    for li, is_correct in enumerate(flags[:n_show]):
        color = "steelblue" if is_correct else "firebrick"
        pts = proj[sample_ids == li]
        T = pts.shape[0]
        alphas = np.linspace(0.15, 0.7, max(T, 2))
        for t in range(T - 1):
            ax.plot(pts[t:t+2, 0], pts[t:t+2, 1],
                    color=color, alpha=float(alphas[t]), lw=0.7)
        ax.scatter(pts[-1, 0], pts[-1, 1], color=color, s=25, alpha=0.85, marker="*", zorder=4)

    legend_elements = [
        Line2D([0], [0], color="steelblue", lw=2, label="Correct"),
        Line2D([0], [0], color="firebrick", lw=2, label="Incorrect"),
        Line2D([0], [0], color="gray", lw=0, marker="*", markersize=8, label="Final step"),
    ]
    ax.legend(handles=legend_elements, fontsize=10)
    ax.set_title(f"{z_label}  PCA trajectories  (correct vs incorrect)", fontsize=12)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.grid(True, lw=0.3, alpha=0.5)
    plt.tight_layout()
    return fig


def plot_forward_residual(residuals, flags, save_dir, z_label="z_H"):
    max_T = max(r.shape[0] for r in residuals) if residuals else 0
    if max_T == 0:
        return None

    def _mean_std(idxs):
        padded = []
        for i in idxs:
            r = residuals[i]
            if r.shape[0] < max_T:
                r = np.pad(r, (0, max_T - r.shape[0]), constant_values=np.nan)
            padded.append(r)
        arr = np.array(padded)
        return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)

    correct_idxs = [i for i, f in enumerate(flags) if f]
    incorrect_idxs = [i for i, f in enumerate(flags) if not f]
    steps = np.arange(1, max_T + 1)

    fig, ax = plt.subplots(figsize=(9, 4))
    for idxs, label, color in [
        (correct_idxs, "Correct", "steelblue"),
        (incorrect_idxs, "Incorrect", "firebrick"),
    ]:
        if not idxs:
            continue
        mean, std = _mean_std(idxs)
        ax.plot(steps, mean, color=color, lw=2, label=label)
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel("Supervision Step Index #", fontsize=11)
    ax.set_ylabel(f"||{z_label}[t] - {z_label}[t-1]||", fontsize=10)
    ax.set_title(f"{z_label}  Forward Residual  (correct vs incorrect)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, lw=0.3, alpha=0.5)
    plt.tight_layout()
    return fig


def plot_pca_variance(pca, save_dir, z_label="z_H"):
    n = len(pca.explained_variance_ratio_)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(range(1, n + 1), pca.explained_variance_ratio_ * 100, color="steelblue")
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance (%)")
    axes[0].set_title("Scree Plot")

    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    axes[1].plot(range(1, n + 1), cumvar, "o-", color="steelblue")
    axes[1].axhline(90, color="gray", linestyle="--", lw=0.8, label="90%")
    axes[1].set_xlabel("# Principal Components")
    axes[1].set_ylabel("Cumulative Variance (%)")
    axes[1].set_title("Cumulative Variance")
    axes[1].legend()

    plt.suptitle(f"{z_label} PCA (mean-pooled over sequence positions)", fontsize=11)
    plt.tight_layout()
    return fig


def plot_displacement_hist(trajs, flags, save_dir, z_label="z_H"):
    correct_disp, incorrect_disp = [], []
    for traj, is_correct in zip(trajs, flags):
        if traj.shape[0] > 1:
            disp = float(np.linalg.norm(np.diff(traj, axis=0), axis=-1).sum())
        else:
            disp = 0.0
        (correct_disp if is_correct else incorrect_disp).append(disp)

    all_vals = correct_disp + incorrect_disp
    if not all_vals:
        return None
    bins = np.linspace(min(all_vals), max(all_vals), 40)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(correct_disp,   bins=bins, alpha=0.6, color="steelblue",
            label=f"Correct  (n={len(correct_disp)})",   density=True)
    ax.hist(incorrect_disp, bins=bins, alpha=0.6, color="firebrick",
            label=f"Incorrect  (n={len(incorrect_disp)})", density=True)
    ax.set_xlabel(f"Total {z_label} displacement  (sum of step-wise L2 norms)")
    ax.set_ylabel("Density")
    ax.set_title(f"{z_label} trajectory total displacement  (correct vs incorrect)")
    ax.legend()
    ax.grid(True, lw=0.3, alpha=0.5)
    plt.tight_layout()
    return fig


def plot_step1_vs_final(proj, sample_ids, flags, save_dir, z_label="z_H"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, is_c, color, title in [
        (axes[0], True,  "steelblue", "Correct"),
        (axes[1], False, "firebrick",  "Incorrect"),
    ]:
        s0, sF = [], []
        for li, f in enumerate(flags):
            if f != is_c:
                continue
            pts = proj[sample_ids == li]
            if pts.shape[0] == 0:
                continue
            s0.append(pts[0])
            sF.append(pts[-1])
        if not s0:
            ax.set_title(f"{title} (no data)")
            continue
        s0 = np.array(s0); sF = np.array(sF)
        ax.scatter(s0[:, 0], s0[:, 1], alpha=0.3, s=12, color="gray",  label="Step 0", zorder=2)
        ax.scatter(sF[:, 0], sF[:, 1], alpha=0.5, s=12, color=color,   label="Final",  zorder=3)
        ax.set_title(f"{title}  (n={len(s0)})", fontsize=11)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.legend(fontsize=9)
        ax.grid(True, lw=0.3, alpha=0.5)
    plt.suptitle(f"{z_label} PCA: Step-1 vs Final step", fontsize=11)
    plt.tight_layout()
    return fig


def plot_init_to_final_split(proj_inits, proj, sample_ids, flags, pca, save_dir, n_show=60, z_label="z_H"):
    """
    proj_inits  : (n_samples, 2)  — H_init projected position
    proj        : (N_total_steps, 2)
    sample_ids  : (N_total_steps,)  local index per step
    flags       : list[bool], length = n_samples
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    correct_local   = [i for i, f in enumerate(flags) if f]
    incorrect_local = [i for i, f in enumerate(flags) if not f]

    cmap_correct   = plt.get_cmap("viridis_r")
    cmap_incorrect = plt.get_cmap("plasma_r")

    for ax, local_idxs, cmap, title in [
        (axes[0], correct_local[:n_show],   cmap_correct,   "Correct"),
        (axes[1], incorrect_local[:n_show], cmap_incorrect, "Incorrect"),
    ]:
        for li in local_idxs:
            pts = proj[sample_ids == li]   # (T, 2)
            T = pts.shape[0]
            if T == 0:
                continue
            colors = [cmap(t / max(T - 1, 1)) for t in range(T)]

            # ── init marker (before step 0) ──────────────────────────
            ax.scatter(proj_inits[li, 0], proj_inits[li, 1],
                        color="black", s=60, marker="x",
                        linewidths=1.2, zorder=5, alpha=0.7)
            # draw a faint dashed line from H_init → step 0
            ax.plot([proj_inits[li, 0], pts[0, 0]],
                    [proj_inits[li, 1], pts[0, 1]],
                    color="gray", lw=0.6, linestyle="--",
                    alpha=0.4, zorder=2)

            # ── trajectory segments (colored by step) ──────────────────
            for t in range(T - 1):
                ax.plot(pts[t:t+2, 0], pts[t:t+2, 1],
                        color=colors[t], lw=0.9, alpha=0.75, zorder=3)

            # ── per-step dots ───────────────────────────────────────────
            for t in range(T):
                ax.scatter(pts[t, 0], pts[t, 1],
                           color=colors[t], s=10, alpha=0.65,
                           zorder=4, linewidths=0)

            # ── start / end markers ─────────────────────────────────────
            ax.scatter(pts[0, 0],  pts[0, 1],  color=colors[0],
                       s=22, alpha=0.9, marker="o", zorder=6,
                       edgecolors="white", linewidths=0.5)
            ax.scatter(pts[-1, 0], pts[-1, 1], color=colors[-1],
                       s=50, alpha=0.95, marker="*", zorder=6,
                       edgecolors="white", linewidths=0.5)

        ax.set_title(f"{title}  (n={len(local_idxs)})", fontsize=11)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.grid(True, lw=0.3, alpha=0.5)

        # ── colorbar showing step progression ──────────────────────────
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04,
                     label="step  (early → late)")

    plt.suptitle(
        f"{z_label}  PCA trajectories  ○=step1  ★=final   ✕=H_init",
        fontsize=11
    )
    plt.tight_layout()
    return fig


def plot_hinit_vs_final(proj_inits, proj, sample_ids, flags, save_dir, z_label="z_H"):
    """Init reset position (×) vs final step (★) for correct/incorrect."""
    init_name = "H_init" if z_label == "z_H" else "L_init"
    fig, ax = plt.subplots(figsize=(8, 7))

    # Init points colored by correct/incorrect
    for li, is_correct in enumerate(flags):
        color = "steelblue" if is_correct else "firebrick"
        ax.scatter(proj_inits[li, 0], proj_inits[li, 1],
                   color=color, s=25, alpha=0.3, marker="x", zorder=5)

    # Final step, colored by correct/incorrect
    for li, is_correct in enumerate(flags):
        color = "steelblue" if is_correct else "firebrick"
        pts = proj[sample_ids == li]
        if pts.shape[0] == 0:
            continue
        ax.scatter(pts[-1, 0], pts[-1, 1],
                   color=color, s=15, alpha=0.5, marker="*", zorder=3)

    legend_elements = [
        Line2D([0], [0], color="steelblue", lw=0, marker="x", markersize=9, alpha=0.4,
               label=f"{init_name} → correct"),
        Line2D([0], [0], color="firebrick", lw=0, marker="x", markersize=9, alpha=0.4,
               label=f"{init_name} → incorrect"),
        Line2D([0], [0], color="steelblue", lw=0, marker="*", markersize=9,
               label="Final (correct)"),
        Line2D([0], [0], color="firebrick", lw=0, marker="*", markersize=9,
               label="Final (incorrect)"),
    ]
    ax.legend(handles=legend_elements, fontsize=10)
    ax.set_title(f"{init_name} reset position vs Final step  (PCA space)", fontsize=12)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.grid(True, lw=0.3, alpha=0.5)
    plt.tight_layout()
    return fig


def plot_pos_residual_heatmap_role(
    pos_residuals: List[np.ndarray],   # per-puzzle (T-1, L_full)
    given_masks: List[np.ndarray],     # per-puzzle (81,) bool
    flags: List[bool],
    role: str,                         # "given" or "empty"
    puzzle_emb_len,
):
    """
    9x9 heatmap with conditional averaging: only count a cell's residual
    when it matches the target *role* in that puzzle.
    Left: correct puzzles, Right: incorrect puzzles.
    Grey cells = no data (that cell never appeared as the target role).
    """
    assert role in ("given", "empty")
    is_target = True if role == "given" else False   # mask value to select

    correct_idxs   = [i for i, f in enumerate(flags) if f]
    incorrect_idxs = [i for i, f in enumerate(flags) if not f]

    def _mean_grid(idxs):
        """Conditional average: only accumulate when cell role matches."""
        if not idxs:
            return None
        accum = np.zeros(81, dtype=np.float64)
        count = np.zeros(81, dtype=np.float64)
        for i in idxs:
            r = pos_residuals[i]                                  # (T-1, L_full)
            cell_r = r[:, puzzle_emb_len:puzzle_emb_len + 81]     # (T-1, 81)
            cell_mean = cell_r.mean(axis=0)                       # (81,)
            gm = given_masks[i]                                   # (81,) bool
            sel = gm if is_target else ~gm
            accum[sel] += cell_mean[sel]
            count[sel] += 1
        grid = np.where(count > 0, accum / count, np.nan)
        return grid.reshape(9, 9)

    grid_c = _mean_grid(correct_idxs)
    grid_i = _mean_grid(incorrect_idxs)

    grids = [g for g in [grid_c, grid_i] if g is not None]
    if not grids:
        return None
    vmin = float(np.nanmin([np.nanmin(g) for g in grids]))
    vmax = float(np.nanmax([np.nanmax(g) for g in grids]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, grid, title in [
        (axes[0], grid_c, f"Correct  (n={len(correct_idxs)})"),
        (axes[1], grid_i, f"Incorrect  (n={len(incorrect_idxs)})"),
    ]:
        if grid is None:
            ax.set_title(f"{title}\n(no data)")
            continue
        im = ax.imshow(grid, vmin=vmin, vmax=vmax, cmap="hot_r", aspect="equal")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Mark cells with no data (NaN) as grey hatched
        for r in range(9):
            for c in range(9):
                if np.isnan(grid[r, c]):
                    rect = plt.Rectangle(
                        (c - 0.5, r - 0.5), 1, 1,
                        linewidth=0, facecolor="lightgrey", zorder=0
                    )
                    ax.add_patch(rect)

        # 3x3 box lines
        for line in [2.5, 5.5]:
            ax.axhline(line, color="white", lw=1.5)
            ax.axvline(line, color="white", lw=1.5)

        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(9))
        ax.set_yticks(range(9))

    role_label = "Given" if role == "given" else "Empty"
    plt.suptitle(
        f"z_H Per-Cell Mean Residual — {role_label} Cells Only\n"
        "Grey = no data for that role,  Bright = high activity",
        fontsize=11
    )
    plt.tight_layout()
    return fig


def plot_pos_residual_heatmap_given(pos_residuals, given_masks, flags, puzzle_emb_len):
    return plot_pos_residual_heatmap_role(pos_residuals, given_masks, flags, role="given", puzzle_emb_len=puzzle_emb_len)


def plot_pos_residual_heatmap_empty(pos_residuals, given_masks, flags, puzzle_emb_len):
    return plot_pos_residual_heatmap_role(pos_residuals, given_masks, flags, role="empty", puzzle_emb_len=puzzle_emb_len)


def plot_pos_residual_by_step(
    pos_residuals: List[np.ndarray],
    given_masks: List[np.ndarray],
    flags: List[bool],
    puzzle_emb_len: int,
):
    """
    Line plot: average residual of given cells vs empty cells as step changes.
    Each draws two lines for correct and incorrect (four lines in total).
    """
    correct_idxs   = [i for i, f in enumerate(flags) if f]
    incorrect_idxs = [i for i, f in enumerate(flags) if not f]

    max_T = max(r.shape[0] for r in pos_residuals) if pos_residuals else 0
    if max_T == 0:
        return None

    def _given_empty_mean_by_step(idxs):
        """
        Return (max_T, 2): [:, 0] = given mean, [:, 1] = empty mean
        """
        if not idxs:
            return None
        given_steps  = [[] for _ in range(max_T)]
        empty_steps  = [[] for _ in range(max_T)]
        for i in idxs:
            r     = pos_residuals[i]                              # (T-1, L_full)
            cells = r[:, puzzle_emb_len:puzzle_emb_len + 81]     # (T-1, 81)
            gm    = given_masks[i]                                # (81,) bool
            T     = cells.shape[0]
            for t in range(max_T):
                if t < T:
                    given_steps[t].extend(cells[t, gm].tolist())
                    empty_steps[t].extend(cells[t, ~gm].tolist())
        given_mean = np.array([np.mean(v) if v else np.nan for v in given_steps])
        empty_mean = np.array([np.mean(v) if v else np.nan for v in empty_steps])
        return np.stack([given_mean, empty_mean], axis=1)   # (max_T, 2)

    res_c = _given_empty_mean_by_step(correct_idxs)
    res_i = _given_empty_mean_by_step(incorrect_idxs)

    steps = np.arange(1, max_T + 1)
    fig, ax = plt.subplots(figsize=(9, 4))

    styles = {
        ("correct",   "given"): ("steelblue", "-",  "Correct / Given"),
        ("correct",   "empty"): ("steelblue", "--", "Correct / Empty"),
        ("incorrect", "given"): ("firebrick",  "-",  "Incorrect / Given"),
        ("incorrect", "empty"): ("firebrick",  "--", "Incorrect / Empty"),
    }
    for (group, cell_type), (color, ls, label) in styles.items():
        res = res_c if group == "correct" else res_i
        if res is None:
            continue
        col = 0 if cell_type == "given" else 1
        ax.plot(steps, res[:, col], color=color, linestyle=ls, lw=2, label=label)

    ax.set_xlabel("Supervision Step Index #", fontsize=11)
    ax.set_ylabel("||z_H[t] - z_H[t-1]|| / sqrt(D)  (per cell)", fontsize=10)
    ax.set_title("z_H Per-Cell Residual: Given vs Empty Cells", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, lw=0.3, alpha=0.5)
    plt.tight_layout()
    return fig


def plot_rating_distribution(ratings, flags, save_dir):
    ratings_arr = np.array(ratings, dtype=float)
    flags_arr   = np.array(flags)
    
    r_max = int(np.percentile(ratings_arr, 99))  # trim extreme long tail
    r_min = int(ratings_arr.min())
    bins = np.arange(r_min, r_max + 2, 1)            # interval is 1

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), 
                              gridspec_kw={'height_ratios': [2, 1]})
    
    # ── count histogram ──────────────────────────────────────────
    ax = axes[0]
    ax.hist(ratings_arr[flags_arr],  bins=bins, alpha=0.7,
            color="steelblue", label=f"Correct  (n={flags_arr.sum()})")
    ax.hist(ratings_arr[~flags_arr], bins=bins, alpha=0.7,
            color="firebrick", label=f"Incorrect  (n={(~flags_arr).sum()})")
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Rating Distribution: Correct vs Incorrect Puzzles", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.set_xlim(-1, r_max + 1)
    
    # ── incorrect rate per rating ───────────────────────────────
    ax2 = axes[1]
    bin_edges = bins
    incorrect_rate = []
    bin_centers = []
    min_count = 10  # bin with too few samples will not be plotted
    
    for i in range(len(bin_edges) - 1):
        mask = (ratings_arr >= bin_edges[i]) & (ratings_arr < bin_edges[i+1])
        total = mask.sum()
        if total < min_count:
            incorrect_rate.append(np.nan)
        else:
            incorrect_rate.append((~flags_arr[mask]).sum() / total)
        bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)
    
    incorrect_rate = np.array(incorrect_rate)
    bin_centers    = np.array(bin_centers)
    
    valid = ~np.isnan(incorrect_rate)
    ax2.bar(bin_centers[valid], incorrect_rate[valid], 
            width=1.0, color="firebrick", alpha=0.7)
    ax2.axhline(y=(~flags_arr).sum() / len(flags_arr), 
                color="gray", linestyle="--", lw=1.2, label="Overall incorrect rate")
    ax2.set_xlabel("Puzzle Rating", fontsize=11)
    ax2.set_ylabel("Incorrect Rate", fontsize=11)
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=9)
    ax2.grid(True, lw=0.3, alpha=0.5)
    ax2.set_xlim(-1, r_max + 1)
    
    plt.tight_layout()
    return fig


def plot_residual_vs_rating(residuals, ratings, flags, save_dir, z_label="z_H"):
    if not ratings:
        return None
    
    ratings_arr = np.array(ratings)
    final_resids = np.array([r[-1] for r in residuals])
    flags_arr = np.array(flags)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    for is_correct, color, label in [
        (True,  "steelblue", "Correct"),
        (False, "firebrick", "Incorrect"),
    ]:
        mask = flags_arr == is_correct
        ax.scatter(
            ratings_arr[mask], final_resids[mask],
            color=color, alpha=0.3, s=8, label=label
        )
        # Add a LOWESS or rolling mean trend line
        if mask.sum() > 10:
            sorted_idx = np.argsort(ratings_arr[mask])
            x_sorted = ratings_arr[mask][sorted_idx]
            y_sorted = final_resids[mask][sorted_idx]
            # Use rolling mean to make trend line
            window = max(1, len(x_sorted) // 20)
            y_smooth = np.convolve(y_sorted, np.ones(window)/window, mode='valid')
            x_smooth = x_sorted[window//2: window//2 + len(y_smooth)]
            ax.plot(x_smooth, y_smooth, color=color, lw=2.5, alpha=0.9)
    
    ax.set_xlabel("Puzzle Rating", fontsize=11)
    ax.set_ylabel(f"Final Step Residual  ||{z_label}[T] - {z_label}[T-1]||", fontsize=10)
    ax.set_title(f"{z_label} Final Residual vs Puzzle Difficulty Rating", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, lw=0.3, alpha=0.5)
    plt.tight_layout()
    return fig


def plot_accuracy_vs_rating(flags, ratings, save_dir, n_bins=20):
    if not ratings:
        return None
    
    ratings_arr = np.array(ratings, dtype=float)
    flags_arr   = np.array(flags,   dtype=float)
    
    # Use percentile boundaries to ensure similar sample count per bin
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(ratings_arr, percentiles)
    bin_edges = np.unique(bin_edges)  # Remove duplicate edges
    
    bin_centers, accuracies, counts = [], [], []
    for i in range(len(bin_edges) - 1):
        mask = (ratings_arr >= bin_edges[i]) & (ratings_arr < bin_edges[i+1])
        if mask.sum() == 0:
            continue
        bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)
        accuracies.append(flags_arr[mask].mean())
        counts.append(mask.sum())
    
    bin_centers = np.array(bin_centers)
    accuracies  = np.array(accuracies)
    counts      = np.array(counts)
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    
    ax1.plot(bin_centers, accuracies, 'o-', color="steelblue", lw=2, ms=5, label="Accuracy")
    ax1.set_ylabel("Accuracy", color="steelblue", fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis='y', labelcolor='steelblue')
    
    ax2.bar(bin_centers, counts, width=(bin_edges[1:] - bin_edges[:-1]) * 0.8,
            alpha=0.3, color="gray", label="Sample count")
    ax2.set_ylabel("Sample Count", color="gray", fontsize=11)
    ax2.tick_params(axis='y', labelcolor='gray')
    
    ax1.set_xlabel("Puzzle Rating", fontsize=11)
    ax1.set_title("Model Accuracy vs Puzzle Difficulty Rating", fontsize=12)
    ax1.grid(True, lw=0.3, alpha=0.5)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_residual_by_rating_colormap(residuals, ratings, flags, save_dir, z_label="z_H", n_show=200):
    if not ratings:
        return None
    
    ratings_arr = np.array(ratings[:n_show], dtype=float)
    flags_arr   = np.array(flags[:n_show])
    
    # Normalize rating to [0, 1] for colormap
    r_log = np.log1p(ratings_arr)          # log(1 + rating), avoid log(0)
    r_min, r_max = r_log.min(), r_log.max()
    r_norm = (r_log - r_min) / (r_max - r_min + 1e-8)
    
    cmap = plt.cm.coolwarm  # low rating = blue, high rating = red
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    
    for ax, is_correct, title in [
        (axes[0], True,  "Correct"),
        (axes[1], False, "Incorrect"),
    ]:
        shown = 0
        for i, (resid, flag, rn) in enumerate(zip(residuals[:n_show], flags_arr, r_norm)):
            if flag != is_correct:
                continue
            steps = np.arange(1, resid.shape[0] + 1)
            ax.plot(steps, resid, color=cmap(rn), alpha=0.4, lw=0.8)
            shown += 1
        
        ax.set_xlabel("Supervision Step Index #", fontsize=10)
        ax.set_ylabel(f"||{z_label}[t] - {z_label}[t-1]||", fontsize=9)
        ax.set_title(f"{title}  (n={shown})", fontsize=11)
        ax.grid(True, lw=0.3, alpha=0.5)
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=r_min, vmax=r_max))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label="log(1 + Puzzle Rating)", fraction=0.02, pad=0.04)
    
    plt.suptitle(f"{z_label} Forward Residual colored by Puzzle Rating", fontsize=12)
    return fig


def plot_recursion_residual(
    rec_z_H: List[np.ndarray],          # list of (T, H_cycles,         L, D)
    rec_z_L: List[np.ndarray],          # list of (T, H_cycles*L_cycles, L, D)
    rec_correct_flags: List[bool],
    H_cycles: int,
    L_cycles: int,
    save_dir: str,
    H_init: Optional[np.ndarray] = None,
    L_init: Optional[np.ndarray] = None,
) -> Optional[plt.Figure]:
    if not rec_z_H:
        return None
 
    T = rec_z_H[0].shape[0]
 
    # ── residual helper ──────────────────────────────────────────────────────
    def _residual(arr: np.ndarray, init: Optional[np.ndarray]) -> np.ndarray:
        T_, S_, L_, D_ = arr.shape
        flat = arr.reshape(T_ * S_, L_, D_).mean(axis=1)   # (T*S, D)
        if init is not None:
            if hasattr(init, "detach"):
                init_np = init.detach().cpu().numpy()
            else:
                init_np = np.asarray(init)
            init_vec = init_np.reshape(-1, D_).mean(axis=0, keepdims=True)  # (1, D)
            full = np.concatenate([init_vec, flat], axis=0)                 # (T*S+1, D)
        else:
            full = flat
        return np.linalg.norm(np.diff(full, axis=0), axis=-1) / math.sqrt(D_)
 
    # ── split by correctness ─────────────────────────────────────────────────
    correct_idxs   = [i for i, f in enumerate(rec_correct_flags) if     f]
    incorrect_idxs = [i for i, f in enumerate(rec_correct_flags) if not f]
 
    resid_H_c = [_residual(rec_z_H[i], H_init) for i in correct_idxs]
    resid_H_i = [_residual(rec_z_H[i], H_init) for i in incorrect_idxs]
    resid_L_c = [_residual(rec_z_L[i], L_init) for i in correct_idxs]
    resid_L_i = [_residual(rec_z_L[i], L_init) for i in incorrect_idxs]
 
    if not resid_H_c and not resid_H_i:
        return None
 
    N_H = T * H_cycles
    N_L = T * H_cycles * L_cycles
 
    def _x_axis(N: int, has_init: bool) -> np.ndarray:
        return np.arange(1, N + 1, dtype=float) if has_init else np.arange(2, N + 1, dtype=float)
 
    x_H = _x_axis(N_H, H_init is not None)
    x_L = _x_axis(N_L, L_init is not None)
 

    h_green  = [t * H_cycles for t in range(1, T + 1)]
    l_green  = [t * (H_cycles * L_cycles) for t in range(1, T + 1)]
 
    h_orange = [
        t * H_cycles + h
        for t in range(T)
        for h in range(1, H_cycles)          # within-step updates (not the last)
    ]
    l_orange = [
        t * (H_cycles * L_cycles) + (h + 1) * L_cycles
        for t in range(T)
        for h in range(H_cycles - 1)         # z_H fires excl. step boundary
    ]
 
    # labeled tick positions: union of green + orange for each panel
    h_labeled = set(h_green) | set(h_orange)   # = {1,2,3,...,48} all positions
    l_labeled = set(l_green) | set(l_orange)   # = {6,12,18,...,288} every L_cycles
 
    # ── figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(20, 9), constrained_layout=True)
 
    COLOR_CORRECT   = "steelblue"
    COLOR_INCORRECT = "firebrick"
    COLOR_GREEN     = "seagreen"
    COLOR_ORANGE    = "goldenrod"
    ALPHA_IND       = 0.60
    LW_IND          = 1.4
 
    panels = [
        (axes[0], "z_H", x_H, resid_H_c, resid_H_i,
         h_green, h_orange, h_labeled, N_H, H_cycles),
        (axes[1], "z_L", x_L, resid_L_c, resid_L_i,
         l_green, l_orange, l_labeled, N_L, H_cycles * L_cycles),
    ]
 
    for ax, label, x, resid_c, resid_i, green_pos, orange_pos, labeled, N, ups in panels:
 
        # trajectories
        for idx, resid in enumerate(resid_c):
            ax.plot(x, resid, color=COLOR_CORRECT, alpha=ALPHA_IND, lw=LW_IND,
                    label="Correct" if idx == 0 else None)
        for idx, resid in enumerate(resid_i):
            ax.plot(x, resid, color=COLOR_INCORRECT, alpha=ALPHA_IND, lw=LW_IND,
                    label="Incorrect" if idx == 0 else None)
 
        # orange dashed: within-step z_H updates
        for idx, xo in enumerate(orange_pos):
            ax.axvline(xo, color=COLOR_ORANGE, lw=0.9, ls="--", alpha=0.65,
                       label="z_H update (within step)" if idx == 0 else None)
 
        # green solid: supervision step boundaries (drawn on top of orange)
        for idx, xg in enumerate(green_pos):
            ax.axvline(xg, color=COLOR_GREEN, lw=1.4, ls="-", alpha=0.80,
                       label="Supervision step boundary" if idx == 0 else None)
 
        # ── x-axis: tick at every integer, label only at line positions ──────
        ax.set_xlim(0.5, N + 0.5)
        all_pos = np.arange(1, N + 1)
        ax.set_xticks(all_pos)
        tick_labels = [str(p) if p in labeled else "" for p in all_pos]
        ax.set_xticklabels(tick_labels, fontsize=6, rotation=90)
        ax.set_xlabel(f"{label} Update Index", fontsize=10)
 
        # ── top axis: step labels centred on each step's x-range ─────────────
        # step t (0-indexed) spans x = [t*ups+1 .. (t+1)*ups]
        # centre = t*ups + (ups+1)/2
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        step_centres = [t * ups + (ups + 1) / 2 for t in range(T)]
        ax2.set_xticks(step_centres)
        ax2.set_xticklabels([f"S{t + 1}" for t in range(T)], fontsize=8)
        ax2.tick_params(axis="x", length=0)
 
        ax.set_ylabel(f"‖{label}[k] − {label}[k−1]‖ / √D", fontsize=10)
        n_c = len(resid_c)
        n_i = len(resid_i)
        init_note = " (from init)" if (H_init if label == "z_H" else L_init) is not None else ""
        ax.set_title(
            f"{label} Residual{init_note} — "
            f"{ups} updates/step,  correct n={n_c},  incorrect n={n_i}",
            fontsize=11,
        )
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, lw=0.3, alpha=0.3, axis="y")
 
    init_str = (
        "H_init & L_init included" if (H_init is not None and L_init is not None)
        else "H_init included" if H_init is not None
        else "L_init included" if L_init is not None
        else "no init"
    )
    plt.suptitle(
        f"TRM Recursion-Level Residual  "
        f"(H_cycles={H_cycles}, L_cycles={L_cycles}, T={T} steps, {init_str})\n"
        f"z_H x-axis: {N_H} updates  |  z_L x-axis: {N_L} updates  |  "
        f"Green solid = step boundary  |  Orange dashed = z_H update within step",
        fontsize=11,
    )
    return fig


def plot_pred_stability(step_pred_stable, correct_flags: List[bool], save_dir: str):
    """Histogram of 'stable match' step, split by correct/incorrect."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    correct_stable = [step_pred_stable[i] 
                      for i, f in enumerate(correct_flags) if f]
    incorrect_stable = [step_pred_stable[i] 
                        for i, f in enumerate(correct_flags) if not f]
    
    max_T = max(step_pred_stable) + 1 if step_pred_stable else 16
    bins = np.arange(0, max_T + 1) - 0.5
    
    ax.hist(correct_stable, bins=bins, alpha=0.6, color='green', 
            label=f'Correct (mean={np.mean(correct_stable):.1f})', density=True)
    ax.hist(incorrect_stable, bins=bins, alpha=0.6, color='red',
            label=f'Incorrect (mean={np.mean(incorrect_stable):.1f})', density=True)
    ax.set_xlabel('Stable Match Step (prediction stops changing)')
    ax.set_ylabel('Density')
    ax.set_title('Prediction Stability (Early Stopping Analysis)')
    ax.set_xticks(np.arange(1, max_T))
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig

def _padded_mean(indices, acc_list, max_T):
    if len(indices) == 0:
        return np.full(max_T, np.nan)
    arrs = []
    for i in indices:
        a = acc_list[i]
        padded = np.pad(a, (0, max_T - len(a)), constant_values=np.nan)
        arrs.append(padded)
    return np.nanmean(arrs, axis=0)


def _split_indices(correct_flags):
    correct_idx = [i for i, f in enumerate(correct_flags) if f]
    incorrect_idx = [i for i, f in enumerate(correct_flags) if not f]
    return correct_idx, incorrect_idx


def plot_logit_lens_accuracy(step_cell_acc, step_empty_acc, step_given_acc, correct_flags: List[bool], save_dir: str, z_label: str = "z_H"):
    """
    Per-step cell accuracy, split by correct/incorrect and given/empty.
    Works for both z_H and z_L by changing channel_name.
    """
    max_T = max(a.shape[0] for a in step_cell_acc)
    correct_idx, incorrect_idx = _split_indices(correct_flags)
    steps = np.arange(1, max_T + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: overall cell accuracy
    ax = axes[0]
    ax.plot(steps, _padded_mean(correct_idx, step_cell_acc, max_T),
            'g-o', ms=4, label='Correct puzzles')
    ax.plot(steps, _padded_mean(incorrect_idx, step_cell_acc, max_T),
            'r-o', ms=4, label='Incorrect puzzles')
    ax.set_xlabel('Supervision Step')
    ax.set_ylabel('Cell-level Accuracy')
    ax.set_xticks(steps)
    ax.set_title(f'Per-step Cell Accuracy — {z_label} Logit Lens')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Right: by cell type
    ax = axes[1]
    ax.plot(steps, _padded_mean(correct_idx, step_empty_acc, max_T),
            'g-o', ms=4, label='Correct (empty cells)')
    ax.plot(steps, _padded_mean(incorrect_idx, step_empty_acc, max_T),
            'r-o', ms=4, label='Incorrect (empty cells)')
    ax.plot(steps, _padded_mean(correct_idx, step_given_acc, max_T),
            'g--s', ms=4, alpha=0.5, label='Correct (given cells)')
    ax.plot(steps, _padded_mean(incorrect_idx, step_given_acc, max_T),
            'r--s', ms=4, alpha=0.5, label='Incorrect (given cells)')
    ax.set_xlabel('Supervision Step')
    ax.set_ylabel('Cell-level Accuracy')
    ax.set_xticks(steps)
    ax.set_title(f'Per-step Accuracy by Cell Type — {z_label} Logit Lens')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig

def plot_disagreement(
    empty_both_correct, empty_both_wrong_same, empty_both_wrong_diff,
    empty_only_z_H_correct, empty_only_z_L_correct,
    given_both_correct, given_both_wrong_same, given_both_wrong_diff,
    given_only_z_H_correct, given_only_z_L_correct,
    correct_flags: List[bool],
    save_dir: str,
):
    """
    Per-step z_H vs z_L prediction agreement, split by cell type.
    5 mutually exclusive categories that sum to 1.0.
    Top row: empty cells, Bottom row: given cells.
    """
    max_T = max(a.shape[0] for a in empty_both_correct)
    correct_idx, incorrect_idx = _split_indices(correct_flags)
    steps = np.arange(1, max_T + 1)
 
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
 
    datasets = [
        (axes[0], "Empty Cells",
         empty_both_correct, empty_both_wrong_same, empty_both_wrong_diff,
         empty_only_z_H_correct, empty_only_z_L_correct),
        (axes[1], "Given Cells",
         given_both_correct, given_both_wrong_same, given_both_wrong_diff,
         given_only_z_H_correct, given_only_z_L_correct),
    ]
 
    labels = ['Both correct', 'Both wrong (same)', 'Both wrong (diff)',
              'Only z_H correct', 'Only z_L correct']
    colors = ['#4CAF50', '#D32F2F', '#FF8A65', '#1976D2', '#7B1FA2']
 
    for ax_row, cell_type, bc, bws, bwd, ohc, olc in datasets:
        for ax, idx, puzzle_type in [
            (ax_row[0], correct_idx, "Correct Puzzles"),
            (ax_row[1], incorrect_idx, "Incorrect Puzzles"),
        ]:
            vals = [_padded_mean(idx, d, max_T) for d in [bc, bws, bwd, ohc, olc]]
 
            ax.stackplot(steps, *vals,
                         labels=labels, colors=colors, alpha=0.8)
            ax.set_title(f'{cell_type} — {puzzle_type}')
            ax.legend(loc='center right', fontsize=7)
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
 
    for ax in axes[:, 0]:
        ax.set_ylabel('Proportion of Cells')
    for ax in axes[1]:
        ax.set_xlabel('Supervision Step')
        ax.set_xticks(steps)
 
    fig.tight_layout()
    return fig

def plot_cosine_similarity(empty_cos_sim, given_cos_sim, correct_flags: List[bool], save_dir: str):
    def _padded_stats(indices, data_list, max_T):
        if not indices:
            return np.full(max_T, np.nan), np.full(max_T, np.nan)
        arrs = []
        for i in indices:
            a = data_list[i]
            padded = np.pad(a, (0, max_T - len(a)), constant_values=np.nan)
            arrs.append(padded)
        stacked = np.array(arrs)
        return np.nanmean(stacked, axis=0), np.nanstd(stacked, axis=0)
    """
    Per-step cosine similarity between z_H and z_L, 4 lines:
    given-correct, given-incorrect, empty-correct, empty-incorrect.
    """
    max_T = max(a.shape[0] for a in empty_cos_sim)
    correct_idx, incorrect_idx = _split_indices(correct_flags)
    steps = np.arange(1, max_T + 1)
 
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
 
    lines = [
        (correct_idx, given_cos_sim, 'green', '--s', 'Given — Correct'),
        (incorrect_idx, given_cos_sim, 'red', '--s', 'Given — Incorrect'),
        (correct_idx, empty_cos_sim, 'green', '-o', 'Empty — Correct'),
        (incorrect_idx, empty_cos_sim, 'red', '-o', 'Empty — Incorrect'),
    ]
 
    for idx, data, color, fmt, label in lines:
        mean, std = _padded_stats(idx, data, max_T)
        is_given = '--' in fmt
        alpha_fill = 0.08 if is_given else 0.15
        ax.plot(steps, mean, fmt, color=color, ms=4,
                alpha=0.5 if is_given else 1.0, label=label)
        ax.fill_between(steps, mean - std, mean + std,
                        color=color, alpha=alpha_fill)
 
    ax.set_xlabel('Supervision Step')
    ax.set_ylabel('Cosine Similarity (z_H, z_L)')
    ax.set_xticks(steps)
    ax.set_title('Per-step Cosine Similarity between z_H and z_L')
    ax.legend()
    ax.grid(True, alpha=0.3)
 
    fig.tight_layout()
    return fig


def plot_cka_matrices(all_z, correct_mask, save_dir: str, z_label: str = "z_H"):
    """
    all_z: (N, T, D) numpy array, mean-pooled over cell positions
    correct_mask: (N,) bool array
    """
    def compute_cka_matrix(Z):
        """
        Z: (N, T, D) — N puzzles, T steps, D hidden dim
        Returns: (T, T) CKA matrix
        """
        T = Z.shape[1]
        cka_mat = np.zeros((T, T))
        for i in range(T):
            for j in range(i, T):
                val = linear_cka(Z[:, i, :], Z[:, j, :])
                cka_mat[i, j] = val
                cka_mat[j, i] = val
        return cka_mat

    all_z = np.array(all_z)           # (N, T, D)
    correct_mask = np.array(correct_mask, dtype=bool)  # (N,)

    z_correct   = all_z[correct_mask]    # (N_c, T, D)
    z_incorrect = all_z[~correct_mask]   # (N_i, T, D)

    print(f"Computing CKA for {z_correct.shape[0]} correct puzzles...")
    cka_correct   = compute_cka_matrix(z_correct)
    print(f"Computing CKA for {z_incorrect.shape[0]} incorrect puzzles...")
    cka_incorrect = compute_cka_matrix(z_incorrect)
    diff_matrix   = cka_correct - cka_incorrect

    T = cka_correct.shape[0]
    tick_labels = [str(i+1) for i in range(T)]

    fig = plt.figure(figsize=(18, 5.5))
    gs  = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.35)

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    cax = fig.add_subplot(gs[3])

    im_kwargs = dict(vmin=0, vmax=1, cmap='viridis', aspect='auto')
    diff_max  = np.abs(diff_matrix).max()

    im0 = ax0.imshow(cka_correct,   **im_kwargs)
    im1 = ax1.imshow(cka_incorrect, **im_kwargs)
    im2 = ax2.imshow(diff_matrix,   vmin=-diff_max, vmax=diff_max,
                     cmap='RdBu_r', aspect='auto')

    for ax, title in zip([ax0, ax1, ax2], [
        f'{z_label} CKA — Correct (n={z_correct.shape[0]:,})',
        f'{z_label} CKA — Incorrect (n={z_incorrect.shape[0]:,})',
        'Difference (Correct − Incorrect)'
    ]):
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(T))
        ax.set_yticks(range(T))
        ax.set_xticklabels(tick_labels, fontsize=7)
        ax.set_yticklabels(tick_labels, fontsize=7)
        ax.set_xlabel('Step', fontsize=9)
        ax.set_ylabel('Step', fontsize=9)

    # shared colorbar for first two panels
    plt.colorbar(im0, cax=cax, label='CKA')

    # separate colorbar for diff panel (right side)
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('ΔCKA', fontsize=8)

    fig.suptitle(f'Linear CKA Inter-Step Similarity: {z_label} Trajectories', 
                 fontsize=13, y=1.02)
    return fig

def _padded_mean_std(indices, data_list, max_T):
    if len(indices) == 0:
        nan = np.full(max_T, np.nan)
        return nan, nan
    arrs = np.array([
        np.pad(data_list[i].astype(float), (0, max_T - len(data_list[i])),
               constant_values=np.nan)
        for i in indices
    ])
    return np.nanmean(arrs, axis=0), np.nanstd(arrs, axis=0)


def _get_bin_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    percentiles = np.linspace(0, 100, n_bins + 1)
    return np.unique(np.percentile(values, percentiles))


def _bin_group_indices(
    values: np.ndarray,
    edges: np.ndarray,
    group_idx: np.ndarray,
) -> List[Tuple[int, int, np.ndarray]]:
    result = []
    n_bins = len(edges) - 1
    for k in range(n_bins):
        lo, hi = edges[k], edges[k + 1]
        last_bin = (k == n_bins - 1)
        in_range = (values >= lo) & (values <= hi if last_bin else values < hi)
        bin_idx = group_idx[in_range[group_idx]]
        result.append((int(lo), int(hi), bin_idx))
    return result


def plot_violation_curve(step_preds_all: List[np.ndarray], given_masks: List[np.ndarray], ratings: List[float], correct_flags: List[bool], save_dir: str, n_bins: int = 4):
    def _count_violations(pred_81):
        board = pred_81.reshape(9, 9)
        violations = 0
        for i in range(9):
            row = board[i, :]
            col = board[:, i]
            box = board[(i//3)*3:(i//3)*3+3, (i%3)*3:(i%3)*3+3].flatten()
            for group in [row, col, box]:
                vals = group[group >= 2]
                violations += len(vals) - len(np.unique(vals))
        return violations

    violation_curves = [
        np.array([_count_violations(preds[t]) for t in range(len(preds))])
        for preds in step_preds_all
    ]
    max_T = max(len(c) for c in violation_curves)
    steps = np.arange(1, max_T + 1)
    ratings_arr = np.array(ratings, dtype=float)
    given_counts = np.array([g.sum() for g in given_masks])
    correct_flags_arr = np.array(correct_flags)
    correct_idx = np.where(correct_flags_arr)[0]
    incorrect_idx = np.where(~correct_flags_arr)[0]

    rating_edges = _get_bin_edges(ratings_arr, n_bins)
    given_edges = _get_bin_edges(given_counts, n_bins)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    subplot_cfgs = [
        (axes[0], ratings_arr, rating_edges, 'Puzzle Rating (tdoku backtracks)'),
        (axes[1], given_counts, given_edges, 'Given-cell Count'),
    ]

    for ax, values, edges, xlabel in subplot_cfgs:
        n_valid_bins = len(edges) - 1
        correct_colors = plt.cm.Blues(np.linspace(0.45, 0.95, n_valid_bins))
        incorrect_colors = plt.cm.Reds(np.linspace(0.45, 0.95, n_valid_bins))

        for k, (lo, hi, c_idx) in enumerate(_bin_group_indices(values, edges, correct_idx)):
            mean, std = _padded_mean_std(c_idx, violation_curves, max_T)
            ax.plot(steps, mean, '-o', color=correct_colors[k], ms=4, label=f'Correct {lo}–{hi} (n={len(c_idx)})')
            ax.fill_between(steps, mean - std, mean + std, color=correct_colors[k], alpha=0.12)

        for k, (lo, hi, ic_idx) in enumerate(_bin_group_indices(values, edges, incorrect_idx)):
            mean, std = _padded_mean_std(ic_idx, violation_curves, max_T)
            ax.plot(steps, mean, '--o', color=incorrect_colors[k], ms=4, label=f'Incorrect {lo}–{hi} (n={len(ic_idx)})')
            ax.fill_between(steps, mean - std, mean + std, color=incorrect_colors[k], alpha=0.12)

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Mean Violation Count')
        ax.set_xticks(steps)
        ax.set_title(f'Violation Curve — Stratified by {xlabel}')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Per-step Sudoku Constraint Violations')
    fig.tight_layout()
    return fig


def plot_difficulty_stratification(step_empty_acc: List[np.ndarray], given_masks: List[np.ndarray], ratings: List[float], correct_flags: List[bool], save_dir: str, n_bins: int = 4):
    ratings_arr = np.array(ratings, dtype=float)
    given_counts = np.array([g.sum() for g in given_masks])
    correct_flags_arr = np.array(correct_flags)
    correct_idx = np.where(correct_flags_arr)[0]
    incorrect_idx = np.where(~correct_flags_arr)[0]
    max_T = max(len(a) for a in step_empty_acc)
    steps = np.arange(1, max_T + 1)

    rating_edges = _get_bin_edges(ratings_arr, n_bins)
    given_edges = _get_bin_edges(given_counts, n_bins)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    subplot_cfgs = [
        (axes[0], ratings_arr,  rating_edges, 'Puzzle Rating (tdoku backtracks)'),
        (axes[1], given_counts, given_edges,  'Given-cell Count'),
    ]

    for ax, values, edges, xlabel in subplot_cfgs:
        n_valid_bins = len(edges) - 1
        correct_colors = plt.cm.Blues(np.linspace(0.45, 0.95, n_valid_bins))
        incorrect_colors = plt.cm.Reds(np.linspace(0.45, 0.95, n_valid_bins))

        for k, (lo, hi, c_idx) in enumerate(_bin_group_indices(values, edges, correct_idx)):
            mean, std = _padded_mean_std(c_idx, step_empty_acc, max_T)
            ax.plot(steps, mean, '-o', color=correct_colors[k], ms=4, label=f'Correct {lo}–{hi} (n={len(c_idx)})')
            ax.fill_between(steps, mean - std, mean + std, color=correct_colors[k], alpha=0.12)

        for k, (lo, hi, ic_idx) in enumerate(_bin_group_indices(values, edges, incorrect_idx)):
            mean, std = _padded_mean_std(ic_idx, step_empty_acc, max_T)
            ax.plot(steps, mean, '--o', color=incorrect_colors[k], ms=4, label=f'Incorrect {lo}–{hi} (n={len(ic_idx)})')
            ax.fill_between(steps, mean - std, mean + std, color=incorrect_colors[k], alpha=0.12)

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Mean Empty-cell Accuracy')
        ax.set_xticks(steps)
        ax.set_title(f'Accuracy — Stratified by {xlabel}')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Difficulty Stratification\nEmpty-cell Accuracy by Difficulty — Rating vs Given-cell Count')
    fig.tight_layout()
    return fig


def plot_logit_lens_entropy(step_z_H_empty_entropy, step_z_H_given_entropy, step_z_L_empty_entropy, step_z_L_given_entropy, correct_flags, save_dir, max_entropy=None):

    max_T = max(a.shape[0] for a in step_z_H_empty_entropy)
    correct_idx, incorrect_idx = _split_indices(correct_flags)
    steps = np.arange(1, max_T + 1)

    # ln(9) ≈ 2.197 for 9-class uniform distribution
    if max_entropy is None:
        max_entropy = np.log(9)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(steps, _padded_mean(correct_idx,   step_z_H_empty_entropy, max_T), 'g-o',  ms=4, label='Correct (empty)')
    ax.plot(steps, _padded_mean(incorrect_idx, step_z_H_empty_entropy, max_T), 'r-o',  ms=4, label='Incorrect (empty)')
    ax.plot(steps, _padded_mean(correct_idx,   step_z_H_given_entropy, max_T), 'g--s', ms=4, alpha=0.5, label='Correct (given)')
    ax.plot(steps, _padded_mean(incorrect_idx, step_z_H_given_entropy, max_T), 'r--s', ms=4, alpha=0.5, label='Incorrect (given)')
    ax.axhline(max_entropy, color='gray', ls=':', alpha=0.5, label=f'Max entropy (ln9={max_entropy:.2f})')
    ax.set_xlabel('Supervision Step')
    ax.set_ylabel('Mean Softmax Entropy (nats)')
    ax.set_title('z_H Logit Lens — Softmax Entropy')
    ax.set_xticks(steps)
    ax.set_ylim(0, max_entropy * 1.15)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(steps, _padded_mean(correct_idx,   step_z_L_empty_entropy, max_T), 'g-o',  ms=4, label='Correct (empty)')
    ax.plot(steps, _padded_mean(incorrect_idx, step_z_L_empty_entropy, max_T), 'r-o',  ms=4, label='Incorrect (empty)')
    ax.plot(steps, _padded_mean(correct_idx,   step_z_L_given_entropy, max_T), 'g--s', ms=4, alpha=0.5, label='Correct (given)')
    ax.plot(steps, _padded_mean(incorrect_idx, step_z_L_given_entropy, max_T), 'r--s', ms=4, alpha=0.5, label='Incorrect (given)')
    ax.axhline(max_entropy, color='gray', ls=':', alpha=0.5, label=f'Max entropy (ln9={max_entropy:.2f})')
    ax.set_xlabel('Supervision Step')
    ax.set_ylabel('Mean Softmax Entropy (nats)')
    ax.set_title('z_L Logit Lens — Softmax Entropy')
    ax.set_xticks(steps)
    ax.set_ylim(0, max_entropy * 1.15)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def _get_errors(idx_list, t, source, n_cells):
    errors = []
    for i in idx_list:
        T_i = source[i].shape[0]
        t_use = min(t, T_i - 1)
        err = n_cells[i] - int(source[i][t_use])
        errors.append(err)
    return np.array(errors)

def _get_error_rates(idx_list, t, source, n_cells):
    rates = []
    for i in idx_list:
        T_i = source[i].shape[0]
        t_use = min(t, T_i - 1)
        err = n_cells[i] - int(source[i][t_use])
        rates.append(err / n_cells[i] if n_cells[i] > 0 else 0.0)
    return np.array(rates)


def plot_severity(
    step_empty_correct_count, step_given_correct_count,
    n_empty, n_given, correct_flags: List[bool], save_dir: str
):
    def _plot_error_hist(ax, errors, color, title, xlabel, show_zero=False):
        max_bin = max(int(errors.max()) + 2, 3)
        ax.hist(errors, bins=range(0, max_bin),
                color=color, edgecolor='black', alpha=0.8)
        if show_zero:
            n_zero = int((errors == 0).sum())
            ax.set_title(f'{title}\nmean={np.mean(errors):.2f}, '
                        f'zero-error={n_zero}/{len(errors)}')
        else:
            ax.axvline(np.mean(errors), color='red', ls='--', lw=1.5,
                    label=f'mean={np.mean(errors):.1f}')
            ax.axvline(np.median(errors), color='blue', ls='--', lw=1.5,
                    label=f'median={np.median(errors):.0f}')
            ax.set_title(f'{title}\nmean={np.mean(errors):.1f}, '
                        f'median={np.median(errors):.0f}')
            ax.legend(fontsize=7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Number of puzzles')
        ax.grid(True, alpha=0.3)

    correct_idx, incorrect_idx = _split_indices(correct_flags)
    highlight_steps = [0, 7, 15]
    step_labels = ['Step 1', 'Step 8', 'Step 16']

    fig, axes = plt.subplots(4, 3, figsize=(18, 22))

    row_configs = [
        # (idx,        source,                   n_cells,  color,           cell_type,    group,     show_zero)
        (incorrect_idx, step_empty_correct_count, n_empty,  'salmon',        'Empty cell', 'Incorrect', False),
        (incorrect_idx, step_given_correct_count, n_given,  'lightskyblue',  'Given cell', 'Incorrect', True),
        (correct_idx,   step_empty_correct_count, n_empty,  'mediumseagreen','Empty cell', 'Correct',   False),
        (correct_idx,   step_given_correct_count, n_given,  'lightgreen',    'Given cell', 'Correct',   True),
    ]

    for row, (idx, source, n_cells, color, cell_type, group, show_zero) in enumerate(row_configs):
        for col, (t, slabel) in enumerate(zip(highlight_steps, step_labels)):
            ax = axes[row, col]
            errors = _get_errors(idx, t, source, n_cells)
            _plot_error_hist(
                ax, errors, color,
                title=f'{group} — {slabel}',
                xlabel=f'{cell_type} error count',
                show_zero=show_zero
            )

    # Row labels
    row_titles = [
        'Incorrect: Empty cell errors\n(Near-miss vs Catastrophic failure?)',
        'Incorrect: Given cell errors\n(Constraint anchoring intact?)',
        'Correct: Empty cell errors\n(Convergence speed across steps?)',
        'Correct: Given cell errors\n(Constraint anchoring in correct trajectories?)',
    ]
    for row, title in enumerate(row_titles):
        axes[row, 0].set_ylabel(f'{title}\n\nNumber of puzzles', fontsize=8)

    fig.suptitle('E2.6a Fig1: Error Severity & Constraint Anchoring\n'
                 'Correct vs Incorrect × Empty vs Given cells',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_recursion_effect(
    step_empty_correct_count, n_empty,
    correct_flags: List[bool], save_dir: str
):
    correct_idx, incorrect_idx = _split_indices(correct_flags)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, idx_list, label, color in [
        (axes[0], incorrect_idx, 'Incorrect', 'salmon'),
        (axes[1], correct_idx,   'Correct',   'mediumseagreen'),
    ]:
        e1  = _get_errors(idx_list, 0,  step_empty_correct_count, n_empty)
        e16 = _get_errors(idx_list, 15, step_empty_correct_count, n_empty)
        ax.scatter(e1, e16, alpha=0.25, s=6, color=color, rasterized=True)
        lim = max(e1.max(), e16.max()) + 1
        ax.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.5, label='y = x (no change)')
        pct_imp   = (e16 < e1).mean() * 100
        pct_worse = (e16 > e1).mean() * 100
        pct_same  = (e16 == e1).mean() * 100
        ax.set_xlabel('Empty cell error count — Step 1')
        ax.set_ylabel('Empty cell error count — Step 16')
        ax.set_title(f'{label}: Step 1 vs Step 16\n'
                     f'improved={pct_imp:.1f}%  same={pct_same:.1f}%  worse={pct_worse:.1f}%')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Delta histogram overlay
    ax = axes[2]
    e1_inc  = _get_errors(incorrect_idx, 0,  step_empty_correct_count, n_empty)
    e16_inc = _get_errors(incorrect_idx, 15, step_empty_correct_count, n_empty)
    e1_cor  = _get_errors(correct_idx,   0,  step_empty_correct_count, n_empty)
    e16_cor = _get_errors(correct_idx,   15, step_empty_correct_count, n_empty)
    delta_inc = e16_inc - e1_inc
    delta_cor = e16_cor - e1_cor
    bin_min = int(min(delta_inc.min(), delta_cor.min())) - 1
    bin_max = int(max(delta_inc.max(), delta_cor.max())) + 2
    bins = range(bin_min, bin_max)
    ax.hist(delta_inc, bins=bins, color='salmon', alpha=0.6,
            edgecolor='black', lw=0.3,
            label=f'Incorrect (mean={np.mean(delta_inc):.1f})')
    ax.hist(delta_cor, bins=bins, color='mediumseagreen', alpha=0.6,
            edgecolor='black', lw=0.3,
            label=f'Correct (mean={np.mean(delta_cor):.1f})')
    ax.axvline(0, color='black', ls='--', lw=1.2, alpha=0.7, label='no change')
    ax.set_xlabel('Δ error count (Step 16 − Step 1)')
    ax.set_ylabel('Number of puzzles')
    ax.set_title('Error change across recursion\n(negative = improved)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle('E2.6a Fig2: Marginal Effect of Recursion (Step 1 → Step 16)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_trajectory_heatmap(
    step_empty_correct_count, n_empty,
    correct_flags: List[bool], save_dir: str,
    n_steps: int = 16,
):
    def _get_trajectory(idx_list, source, n_cells, n_steps=16):
        trajs = []
        for i in idx_list:
            T_i = source[i].shape[0]
            row = []
            for t in range(n_steps):
                t_use = min(t, T_i - 1)
                err = n_cells[i] - int(source[i][t_use])
                row.append(err)
            trajs.append(row)
        return np.array(trajs)  # (N, 16)
 
    def spearman_sort(idx_list, step_empty_correct_count, n_empty, n_steps):
        trajs = _get_trajectory(idx_list, step_empty_correct_count, n_empty, n_steps)
        steps = np.arange(n_steps)
        rhos = np.array([
            stats.spearmanr(steps, trajs[i]).statistic
            for i in range(len(idx_list))
        ])
        valid_mask = ~np.isnan(rhos)
        n_constant = (~valid_mask).sum()
        trajs_valid = trajs[valid_mask]
        rhos_valid  = rhos[valid_mask]
        order = np.argsort(rhos_valid)
        return trajs_valid[order], rhos_valid[order], n_constant
 
    correct_idx, incorrect_idx = _split_indices(correct_flags)
 
    groups = {}
    for label, idx_list in [("Incorrect", incorrect_idx), ("Correct", correct_idx)]:
        trajs, rhos, n_const = spearman_sort(
            idx_list, step_empty_correct_count, n_empty, n_steps
        )
        n_total = len(idx_list)
        n_valid = len(rhos)
        n_imp   = (rhos < -0.3).sum()
        n_wors  = (rhos >  0.3).sum()
        n_osc   = n_valid - n_imp - n_wors
        groups[label] = dict(
            trajs=trajs, rhos=rhos,
            n_total=n_total, n_valid=n_valid, n_constant=n_const,
            n_improving=n_imp, n_oscillating=n_osc, n_worsening=n_wors,
        )
 
    global_vmax = np.nanpercentile(
        np.concatenate([groups["Incorrect"]["trajs"].ravel(),
                        groups["Correct"]["trajs"].ravel()]), 99
    )
 
    # ── layout ────────────────────────────────────────────────────────────────
    # Width ratio: heatmap_incorrect : heatmap_correct : histogram = 2 : 3 : 2
    # (correct has far more puzzles → taller, so give it more width share)
    fig, axes = plt.subplots(
        1, 3,
        figsize=(26, 11),
        gridspec_kw={"width_ratios": [2, 3, 2]},
    )
    fig.subplots_adjust(
        left=0.06, right=0.97,
        top=0.84,   bottom=0.08,
        wspace=0.38,
    )
 
    step_ticks = np.arange(n_steps)
 
    # ── heatmaps ─────────────────────────────────────────────────────────────
    for ax, label, cmap in [
        (axes[0], "Incorrect", "RdYlGn_r"),
        (axes[1], "Correct",   "RdYlGn_r"),
    ]:
        g = groups[label]
        trajs  = g["trajs"]
        rhos   = g["rhos"]
        n_valid  = g["n_valid"]
        n_const  = g["n_constant"]
        n_total  = g["n_total"]
        n_imp    = g["n_improving"]
        n_osc    = g["n_oscillating"]
        n_wors   = g["n_worsening"]
 
        im = ax.imshow(
            trajs, aspect="auto", cmap=cmap, interpolation="nearest",
            vmin=0, vmax=global_vmax,
            extent=[-0.5, n_steps - 0.5, n_valid, 0],
        )
        plt.colorbar(im, ax=ax, label="Empty cell error count",
                     fraction=0.035, pad=0.03, shrink=0.85)
 
        ax.set_xlabel("Supervision step", fontsize=10)
        ax.set_ylabel("Puzzle (sorted by Spearman ρ: improving → worsening)", fontsize=9)
        ax.set_xticks(step_ticks)
        ax.set_xticklabels([str(s + 1) for s in step_ticks], fontsize=7)
 
        # group boundary lines
        ax.axhline(n_imp,         color="white", lw=1.5, ls="--", alpha=0.85)
        ax.axhline(n_imp + n_osc, color="white", lw=1.5, ls="--", alpha=0.85)
 
        # group labels inside heatmap (right side)
        for yp, gl in [
            (n_imp / 2,                  "improving"),
            (n_imp + n_osc / 2,          "oscillating"),
            (n_imp + n_osc + n_wors / 2, "worsening"),
        ]:
            ax.text(n_steps - 0.6, yp, gl,
                    va="center", ha="right", fontsize=7.5, color="white",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.40, lw=0))
 
        # ── per-axis title: two-line compact format ──────────────────────────
        mean_rho   = np.nanmean(rhos)
        median_rho = np.nanmedian(rhos)
 
        # Line 1: header + constant-traj note (if any)
        line1 = f"{label} puzzles  (n={n_total}; {n_valid} with valid ρ)"
        if n_const > 0:
            line1 += f"  [constant traj excluded: {n_const}]"
 
        # Line 2: group counts (compact, single line)
        line2 = (
            f"improving (ρ<−0.3): {n_imp} ({n_imp/n_valid*100:.1f}%)   "
            f"oscillating (|ρ|≤0.3): {n_osc} ({n_osc/n_valid*100:.1f}%)   "
            f"worsening (ρ>0.3): {n_wors} ({n_wors/n_valid*100:.1f}%)"
        )
 
        # Line 3: summary stats
        line3 = f"mean ρ = {mean_rho:.3f}     median ρ = {median_rho:.3f}"
 
        ax.set_title(
            f"{line1}\n{line2}\n{line3}",
            fontsize=7.8,
            loc="left",
            pad=8,
            linespacing=1.55,
        )
 
    # ── histogram ─────────────────────────────────────────────────────────────
    ax = axes[2]
    bins = np.linspace(-1, 1, 41)
 
    rhos_inc = groups["Incorrect"]["rhos"]
    rhos_cor = groups["Correct"]["rhos"]
    mean_inc, median_inc = np.nanmean(rhos_inc), np.nanmedian(rhos_inc)
    mean_cor, median_cor = np.nanmean(rhos_cor), np.nanmedian(rhos_cor)
 
    ax.hist(rhos_inc, bins=bins, color="salmon",         alpha=0.70,
            edgecolor="black", lw=0.3,
            label=f"Incorrect  (mean={mean_inc:.2f}, median={median_inc:.2f})")
    ax.hist(rhos_cor, bins=bins, color="mediumseagreen", alpha=0.70,
            edgecolor="black", lw=0.3,
            label=f"Correct    (mean={mean_cor:.2f}, median={median_cor:.2f})")
 
    ax.axvline(-0.3, color="dimgray", ls="--", lw=1.2, alpha=0.8, label="|ρ|=0.3 threshold")
    ax.axvline( 0.3, color="dimgray", ls="--", lw=1.2, alpha=0.8)
    ax.axvline( 0.0, color="black",   ls="-",  lw=0.8, alpha=0.4)
    ax.axvline(median_inc, color="firebrick", ls=":", lw=1.8, alpha=0.9,
               label=f"Incorrect median ({median_inc:.2f})")
    ax.axvline(median_cor, color="darkgreen",  ls=":", lw=1.8, alpha=0.9,
               label=f"Correct median ({median_cor:.2f})")
 
    n_const_inc = groups["Incorrect"]["n_constant"]
    n_const_cor = groups["Correct"]["n_constant"]
    if n_const_inc > 0 or n_const_cor > 0:
        note = (f"Excluded (constant traj, ρ=NaN):\n"
                f"  Incorrect: {n_const_inc}  |  Correct: {n_const_cor}")
        ax.text(0.98, 0.97, note, transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow",
                          ec="goldenrod", lw=1))
 
    ax.set_xlabel("Spearman ρ  (step index vs empty-cell error count)", fontsize=10)
    ax.set_ylabel("Number of puzzles", fontsize=10)
    ax.set_title(
        "Trajectory monotonicity distribution\n"
        "(ρ→−1: monotone improving,  ρ→+1: worsening,  |ρ|≤0.3: oscillating)",
        fontsize=9,
        pad=8,
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
 
    # ── suptitle (above all axes, no overlap) ────────────────────────────────
    fig.suptitle(
        "E2.6a Fig3: Per-puzzle Trajectory Shape — Monotone vs Oscillating\n"
        "(sorted by Spearman ρ; dashed lines = group boundaries; "
        "unified colorbar scale; constant trajectories excluded from ρ)",
        fontsize=12, fontweight="bold",
        y=0.98,
    )
 
    return fig

def plot_cdf(
    step_empty_correct_count, n_empty,
    correct_flags: List[bool], save_dir: str
):
    correct_idx, incorrect_idx = _split_indices(correct_flags)
    highlight_steps = [0, 7, 15]
    step_labels = ['Step 1', 'Step 8', 'Step 16']

    fig, ax = plt.subplots(figsize=(12, 6))

    colors_inc = ['#e74c3c', '#c0392b', '#7b241c']
    colors_cor = ['#27ae60', '#1e8449', '#145a32']

    for col, (t, slabel) in enumerate(zip(highlight_steps, step_labels)):
        for idx_list, colors, ls, group in [
            (incorrect_idx, colors_inc, '-',  'Incorrect'),
            (correct_idx,   colors_cor, '--', 'Correct'),
        ]:
            rates = _get_error_rates(idx_list, t, step_empty_correct_count, n_empty)
            sorted_r = np.sort(rates)
            cdf = np.arange(1, len(sorted_r) + 1) / len(sorted_r)
            ax.plot(sorted_r, cdf, color=colors[col], lw=2.0, ls=ls,
                    label=f'{group} {slabel} (mean={rates.mean():.2f})')

    ax.axvline(0.5, color='gray', ls=':', lw=1, alpha=0.5, label='50% error rate')
    ax.set_xlabel('Empty cell error rate (error count / n_empty cells)')
    ax.set_ylabel('Cumulative fraction of puzzles')
    ax.set_title('E2.6a Fig4: CDF of per-puzzle empty cell error rate\n'
                 'Correct vs Incorrect across steps  '
                 '(solid=incorrect, dashed=correct; left shift = improvement)')
    ax.legend(fontsize=8, ncol=2, loc='lower right')
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def _padded_collect(indices, arrays, max_T):
    """Collect per-step values as list-of-arrays for violin plots."""
    # returns list of length max_T, each element = 1-D array of values at that step
    per_step = [[] for _ in range(max_T)]
    for i in indices:
        t = arrays[i].shape[0]
        for s in range(t):
            per_step[s].append(float(arrays[i][s]))
    return [np.array(v) if v else np.array([np.nan]) for v in per_step]

def _style_violin(vp, color: str) -> None:
    for body in vp['bodies']:
        body.set_facecolor(color)
        body.set_alpha(0.5)
    for key in ('cmeans', 'cmins', 'cmaxes', 'cbars'):
        if key in vp:
            vp[key].set_color(color)

def _gather_at_steps(indices: List[int], arr_list: List[np.ndarray],
                     steps: Tuple[int, ...]) -> dict:
    """Gather per-puzzle scalar values at specific 1-indexed steps."""
    result = {s: [] for s in steps}
    for i in indices:
        a = arr_list[i]
        for s in steps:
            if s - 1 < a.shape[0]:
                result[s].append(float(a[s - 1]))
    return result
 
 
def _build_violin_pair(ax, corr_data: dict, incorr_data: dict,
                       selected_steps: Tuple[int, ...]) -> None:
    """Draw paired violins (correct=green, incorrect=red) on a single axes."""
    pos_c, pos_i, data_c, data_i = [], [], [], []
    for k, s in enumerate(selected_steps):
        pos = k * 3
        pos_c.append(pos - 0.4)
        pos_i.append(pos + 0.4)
        data_c.append(corr_data[s] if corr_data[s] else [np.nan])
        data_i.append(incorr_data[s] if incorr_data[s] else [np.nan])
 
    vp_c = ax.violinplot(data_c, positions=pos_c,
                         showmeans=True, showmedians=False, widths=0.7)
    vp_i = ax.violinplot(data_i, positions=pos_i,
                         showmeans=True, showmedians=False, widths=0.7)
    _style_violin(vp_c, 'green')
    _style_violin(vp_i, 'red')
 
    ax.set_xticks([k * 3 for k in range(len(selected_steps))])
    ax.set_xticklabels([f'Step {s}' for s in selected_steps])
    ax.legend(handles=[Patch(facecolor='green', alpha=0.5, label='Correct'),
                       Patch(facecolor='red', alpha=0.5, label='Incorrect')],
              fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.3, axis='y')
 
 
def _rank_per_step_mean(indices: List[int],
                        step_output_correct_rank: List[np.ndarray],
                        given_masks: List[np.ndarray],
                        mask_fn, max_T: int,
                        agg_fn=None) -> np.ndarray:
    """Compute per-step aggregated value from rank data.
    
    agg_fn: callable(cell_ranks_1d) -> scalar.  Default: mean rank.
    """
    if agg_fn is None:
        agg_fn = lambda r: r.astype(np.float32).mean()
    vals = np.full((len(indices), max_T), np.nan)
    for row, i in enumerate(indices):
        rank_arr = step_output_correct_rank[i]      # (T, 81)
        cell_mask = mask_fn(given_masks[i])          # (81,)
        T = rank_arr.shape[0]
        for t in range(T):
            cell_ranks = rank_arr[t, cell_mask]
            if len(cell_ranks) > 0:
                vals[row, t] = agg_fn(cell_ranks)
    return np.nanmean(vals, axis=0)
 
 

def plot_top1_prob(
    step_output_empty_top1_prob: List[np.ndarray],
    step_output_given_top1_prob: List[np.ndarray],
    correct_flags: List[bool],
    save_dir: str,
    selected_steps: Tuple[int, ...] = (1, 4, 8, 12, 16),
) -> plt.Figure:
    """
    Proposal Fig1: violin plot of per-puzzle mean top-1 probability.
    Two panels (Empty / Given), y-axis [0.90, 1.00].
    """
    correct_idx, incorrect_idx = _split_indices(correct_flags)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
 
    for ax, arr_list, title in zip(
        axes,
        [step_output_empty_top1_prob, step_output_given_top1_prob],
        ['Empty Cells', 'Given Cells'],
    ):
        corr = _gather_at_steps(correct_idx, arr_list, selected_steps)
        incorr = _gather_at_steps(incorrect_idx, arr_list, selected_steps)
        _build_violin_pair(ax, corr, incorr, selected_steps)
        ax.set_title(f'Top-1 Probability — {title}')
        ax.set_ylabel('Top-1 Probability')
        ax.set_ylim(0.90, 1.00)
 
    fig.suptitle('E2.1b Fig1: Top-1 Probability', fontweight='bold')
    fig.tight_layout()
    return fig

 
def plot_margin(
    step_output_empty_margin: List[np.ndarray],
    step_output_given_margin: List[np.ndarray],
    correct_flags: List[bool],
    save_dir: str,
    selected_steps: Tuple[int, ...] = (1, 4, 8, 12, 16),
) -> plt.Figure:
    """Supplementary: margin (top-1 − top-2) violin, same layout as Fig1."""
    correct_idx, incorrect_idx = _split_indices(correct_flags)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
 
    for ax, arr_list, title in zip(
        axes,
        [step_output_empty_margin, step_output_given_margin],
        ['Empty Cells', 'Given Cells'],
    ):
        corr = _gather_at_steps(correct_idx, arr_list, selected_steps)
        incorr = _gather_at_steps(incorrect_idx, arr_list, selected_steps)
        _build_violin_pair(ax, corr, incorr, selected_steps)
        ax.set_title(f'Margin (Top-1 − Top-2) — {title}')
        ax.set_ylabel('Margin')
        ax.set_ylim(0.90, 1.00)
 
    fig.suptitle('E2.1b Fig1b: Margin', fontweight='bold')
    fig.tight_layout()
    return fig


def plot_correct_answer_rank_histogram(
    step_output_correct_rank: List[np.ndarray],
    given_masks: List[np.ndarray],
    correct_flags: List[bool],
    save_dir: str,
    selected_steps: Tuple[int, ...] = (1, 8, 16),
) -> plt.Figure:
    """
    Proposal Fig2: rank histogram (1-9) of correct answer.
    Layout: 2 rows (Empty / Given) × 2 cols (Incorrect / Correct).
    Each panel overlays histograms at selected steps.
    """
    correct_idx, incorrect_idx = _split_indices(correct_flags)
    ranks = np.arange(1, 10)
    colors = {1: '#1f77b4', 8: '#ff7f0e', 16: '#2ca02c'}
    n_sel = len(selected_steps)
 
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey='row')
 
    for row, (cell_label, mask_fn) in enumerate([
        ('Empty Cells', lambda gm: ~gm),
        ('Given Cells', lambda gm: gm),
    ]):
        for col, (indices, group_label) in enumerate([
            (incorrect_idx, 'Incorrect'),
            (correct_idx, 'Correct'),
        ]):
            ax = axes[row, col]
            for si, s in enumerate(selected_steps):
                all_ranks = []
                for i in indices:
                    rank_arr = step_output_correct_rank[i]    # (T, 81)
                    cell_mask = mask_fn(given_masks[i])        # (81,)
                    if s - 1 < rank_arr.shape[0]:
                        all_ranks.append(rank_arr[s - 1, cell_mask])
                if all_ranks:
                    all_ranks = np.concatenate(all_ranks)
                    counts, _ = np.histogram(all_ranks, bins=np.arange(0.5, 10.5, 1))
                    props = counts / max(counts.sum(), 1)
                    offset = (si - (n_sel - 1) / 2) * 0.25
                    ax.bar(ranks + offset, props, width=0.25,
                           color=colors[s], alpha=0.75, label=f'Step {s}')
 
            ax.set_title(f'{cell_label} — {group_label}')
            ax.set_xlabel('Rank (1=best, 9=worst)')
            ax.set_ylabel('Proportion')
            ax.set_xticks(ranks)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
 
    fig.suptitle('E2.1b Fig2: Correct Answer Rank Distribution', fontweight='bold')
    fig.tight_layout()
    return fig


def plot_correct_is_top2(
    step_output_correct_rank: List[np.ndarray],
    given_masks: List[np.ndarray],
    correct_flags: List[bool],
    save_dir: str,
) -> plt.Figure:
    """
    Proposal Fig3 (optional): fraction of cells where correct answer rank == 2.
    Near-miss: correct answer is 2nd-ranked but not 1st.
    y-axis auto-zoomed. Two panels: Empty / Given.
    """
    correct_idx, incorrect_idx = _split_indices(correct_flags)
    max_T = max(r.shape[0] for r in step_output_correct_rank)
    steps = np.arange(1, max_T + 1)
    is_top2_fn = lambda r: float((r == 2).mean())
 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
 
    for ax, (cell_label, mask_fn) in zip(axes, [
        ('Empty Cells', lambda gm: ~gm),
        ('Given Cells', lambda gm: gm),
    ]):
        for indices, color, label in [
            (correct_idx, 'g', 'Correct'),
            (incorrect_idx, 'r', 'Incorrect'),
        ]:
            mean_curve = _rank_per_step_mean(
                indices, step_output_correct_rank, given_masks,
                mask_fn, max_T, agg_fn=is_top2_fn)
            ax.plot(steps, mean_curve, f'{color}-o', ms=4, label=label)
 
        # auto y-range from plotted data
        all_y = np.concatenate([l.get_ydata() for l in ax.get_lines()])
        all_y = all_y[~np.isnan(all_y)]
        if len(all_y) > 0:
            ax.set_ylim(max(0, all_y.min() - 0.02), min(1, all_y.max() + 0.02))
 
        ax.set_title(f'Correct Answer is Rank 2 — {cell_label}')
        ax.set_xlabel('Supervision Step')
        ax.set_ylabel('Proportion')
        ax.set_xticks(steps)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
 
    fig.suptitle('E2.1b Fig3: Top-2 Correct Rate (Zoomed)', fontweight='bold')
    fig.tight_layout()
    return fig
 
 
# ── Supplementary : Mean Rank Line Plot ─────────────────────
 
def plot_correct_answer_rank_mean(
    step_output_correct_rank: List[np.ndarray],
    given_masks: List[np.ndarray],
    correct_flags: List[bool],
    save_dir: str,
) -> plt.Figure:
    """
    Supplementary: mean correct answer rank (1-9) per step.
    Two panels: Empty / Given. y-axis inverted (1=top).
    """
    correct_idx, incorrect_idx = _split_indices(correct_flags)
    max_T = max(r.shape[0] for r in step_output_correct_rank)
    steps = np.arange(1, max_T + 1)
 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
 
    for ax, (cell_label, mask_fn) in zip(axes, [
        ('Empty Cells', lambda gm: ~gm),
        ('Given Cells', lambda gm: gm),
    ]):
        for indices, color, label in [
            (correct_idx, 'g', 'Correct'),
            (incorrect_idx, 'r', 'Incorrect'),
        ]:
            mean_curve = _rank_per_step_mean(
                indices, step_output_correct_rank, given_masks,
                mask_fn, max_T)
            ax.plot(steps, mean_curve, f'{color}-o', ms=4, label=label)
 
        ax.set_title(f'Mean Rank of Correct Answer — {cell_label}')
        ax.set_xlabel('Supervision Step')
        ax.set_ylabel('Mean Rank (1=best, 9=worst)')
        ax.set_xticks(steps)
        ax.set_ylim(0.5, 9.5)
        ax.invert_yaxis()
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
 
    fig.suptitle('E2.1b Supplementary: Mean Correct Answer Rank', fontweight='bold')
    fig.tight_layout()
    return fig