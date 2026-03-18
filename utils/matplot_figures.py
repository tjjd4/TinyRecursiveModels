from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# individual plot functions (each returns fig for wandb logging)

def _plot_pca_split(proj, sample_ids, flags, pca, save_dir, n_show=60, z_label="z_H"):
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

    plt.suptitle(f"TRM  {z_label}  PCA trajectories  ○=step1  ★=final", fontsize=11)
    plt.tight_layout()
    return fig


def _plot_pca_combined(proj, sample_ids, flags, pca, save_dir, n_show=80, z_label="z_H"):
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
    ax.set_title(f"TRM  {z_label}  PCA trajectories  (correct vs incorrect)", fontsize=12)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.grid(True, lw=0.3, alpha=0.5)
    plt.tight_layout()
    return fig


def _plot_forward_residual(residuals, flags, save_dir, z_label="z_H"):
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
    ax.set_title(f"TRM  {z_label}  Forward Residual  (correct vs incorrect)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, lw=0.3, alpha=0.5)
    plt.tight_layout()
    return fig


def _plot_pca_variance(pca, save_dir, z_label="z_H"):
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

    plt.suptitle(f"PCA of TRM  {z_label}  (mean-pooled over sequence positions)", fontsize=11)
    plt.tight_layout()
    return fig


def _plot_displacement_hist(trajs, flags, save_dir, z_label="z_H"):
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


def _plot_step1_vs_final(proj, sample_ids, flags, save_dir, z_label="z_H"):
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
    plt.suptitle(f"{z_label} PCA: Step-0 vs Final step", fontsize=11)
    plt.tight_layout()
    return fig


def _plot_hinit_vs_final(proj_inits, proj, sample_ids, flags, save_dir, z_label="z_H"):
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


def _plot_pos_residual_heatmap_role(
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


def _plot_pos_residual_heatmap_given(pos_residuals, given_masks, flags, puzzle_emb_len):
    return _plot_pos_residual_heatmap_role(pos_residuals, given_masks, flags, role="given", puzzle_emb_len=puzzle_emb_len)


def _plot_pos_residual_heatmap_empty(pos_residuals, given_masks, flags, puzzle_emb_len):
    return _plot_pos_residual_heatmap_role(pos_residuals, given_masks, flags, role="empty", puzzle_emb_len=puzzle_emb_len)


def _plot_pos_residual_by_step(
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


def _plot_rating_distribution(ratings, flags, save_dir):
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


def _plot_residual_vs_rating(residuals, ratings, flags, save_dir, z_label="z_H"):
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


def _plot_accuracy_vs_rating(flags, ratings, save_dir, n_bins=20):
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


def _plot_residual_by_rating_colormap(residuals, ratings, flags, save_dir, z_label="z_H", n_show=200):
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
