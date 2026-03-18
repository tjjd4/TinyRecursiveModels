from typing import Optional, List, Any
from dataclasses import dataclass
import os
import math
import yaml
import shutil
import copy

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig

from puzzle_dataset_with_rating import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path, load_checkpoint_from_path
from utils.matplot_figures import _plot_pca_split, _plot_pca_combined, _plot_forward_residual, _plot_pca_variance, _plot_displacement_hist, _plot_step1_vs_final, _plot_hinit_vs_final, _plot_pos_residual_heatmap_given, _plot_pos_residual_heatmap_empty, _plot_pos_residual_by_step, _plot_rating_distribution, _plot_residual_vs_rating, _plot_accuracy_vs_rating, _plot_residual_by_rating_colormap

from models.losses.loss_fn import IGNORE_LABEL_ID


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class EvalConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    
    eval_loss: LossConfig
    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: str
    checkpoint_path: Optional[str] = None
    
    # Data
    data_paths: List[str]
    data_paths_test: List[str] = []
    # Evaluators
    evaluators: List[EvaluatorConfig] = []
    # Hyperparams
    global_batch_size: int

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    eval_save_outputs: List[str] = []

    # Z analysis (new)
    z_analysis: bool = True
    z_analysis_max_batches: int = 100
    z_analysis_max_samples_pca: int = 256
    z_analysis_pca_components: int = 10


@dataclass
class TrainState:
    model: nn.Module
    carry: Any
    step: int
    total_steps: int

def create_dataloader(config: EvalConfig, split: str, rank: int, world_size: int, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=config.data_paths_test if len(config.data_paths_test) > 0 and split == "test" else config.data_paths,
        rank=rank,
        num_replicas=world_size,
        **kwargs
    ), split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True
    )

    print(f"Dataset: {dataset.config.dataset_paths}")
    print(f"Split: {split}")
    print(f"Vocab size: {dataset.metadata.vocab_size}")
    print(f"Sequence length: {dataset.metadata.seq_len}")
    print(f"Total puzzles: {dataset.metadata.total_puzzles}")
    print()

    return dataloader, dataset.metadata


def load_model_from_checkpoint(config: EvalConfig, metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size // world_size,
        vocab_size=metadata.vocab_size,
        seq_len=metadata.seq_len,
        num_puzzle_identifiers=metadata.num_puzzle_identifiers,
        causal=False  # Non-autoregressive
    )

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.eval_loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        print(model)
        model = loss_head_cls(model, **config.eval_loss.__pydantic_extra__)  # type: ignore
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model)  # type: ignore

        # Load checkpoint
        if not os.path.exists(config.load_checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {config.load_checkpoint}")
        if rank == 0:
            load_checkpoint(model, config)

        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    return model

def init_train_state(config: EvalConfig, metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    # Estimated total training steps
    total_steps = math.ceil(metadata.total_puzzles * metadata.mean_puzzle_examples / config.global_batch_size)

    # Model
    model = load_model_from_checkpoint(config, metadata, rank, world_size)

    return TrainState(
        step=0,
        total_steps=total_steps,

        model=model,
        carry=None
    )

def load_checkpoint(model: nn.Module, config: EvalConfig):
    load_checkpoint_from_path(model, config.load_checkpoint)

def create_evaluators(config: EvalConfig, metadata: PuzzleDatasetMetadata) -> List[Any]:
    data_paths = config.data_paths_test if len(config.data_paths_test) > 0 else config.data_paths
    # Initialize evaluators
    evaluators = []
    for cfg in config.evaluators:
        for data_path in data_paths:
            cls = load_model_class(cfg.name, "evaluators.")(
                data_path=data_path, eval_metadata=metadata, **cfg.__pydantic_extra__
            )  # type: ignore
            evaluators.append(cls)

    return evaluators


# ─────────────────────────────────────────────────────────────────────────────
# Z ANALYSIS  (new, self-contained section)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ZAnalysisCollector:
    """Accumulates z_H snapshots and correctness flags across batches."""
    # Per-puzzle: list of (T_steps, L, D) float32 arrays (mean-pooled over L → (T,D))
    trajectories: List[np.ndarray] = None   # filled after collection
    z_L_trajectories: List[np.ndarray] = None
    correct_flags: List[bool] = None
    residuals: List[np.ndarray] = None      # per-puzzle (T-1,) arrays
    z_L_residuals: List[np.ndarray] = None

    def __post_init__(self):
        self.trajectories  = []
        self.z_L_trajectories = []
        self.correct_flags = []
        self.ratings       = []   # per-sample difficulty rating (optional)
        self.residuals     = []
        self.z_L_residuals = []
        self.pos_residuals = []
        self.given_masks   = []
        self.n_skipped     = 0

        # Internal accumulation buffers keyed by puzzle index in a batch
        self._z_H_per_step_batch: List[List[np.ndarray]] = []  # reset each batch
        self._z_L_per_step_batch: List[List[np.ndarray]] = []  # reset each batch
        self._z_H_pos_per_step_batch = []
        self._batch_open: bool = False

    def begin_batch(self, batch, config: EvalConfig):
        """Call once before the while-loop for each batch."""
        batch_size = config.global_batch_size
        self._z_H_per_step_batch = [[] for _ in range(batch_size)]
        self._z_L_per_step_batch = [[] for _ in range(batch_size)]
        self._z_H_pos_per_step_batch = [[] for _ in range(batch_size)]
        self._batch_open = True
        inputs_np = batch["inputs"].cpu().numpy()   # (B, seq_len)
        self._given_masks_batch = []
        for b in range(batch_size):
            cell_inputs = inputs_np[b, :81]   # (81,) cell labels
            given = cell_inputs != 1
            self._given_masks_batch.append(given)

    def record_step(self, carry):
        """Call inside the while-loop after every model forward, passing the new carry."""
        if not self._batch_open:
            return
        # carry.inner_carry.z_H : (B, L, D)
        z_H_cpu = carry.inner_carry.z_H.float().cpu().numpy()
        z_L_cpu = carry.inner_carry.z_L.float().cpu().numpy()
        for b in range(z_H_cpu.shape[0]):
            # Mean-pool over sequence positions → (D,)  to save memory
            self._z_H_per_step_batch[b].append(z_H_cpu[b].mean(axis=0))
            self._z_L_per_step_batch[b].append(z_L_cpu[b].mean(axis=0))
            self._z_H_pos_per_step_batch[b].append(z_H_cpu[b])

    def end_batch(self, carry, labels, final_preds, ratings=None):
        """
        Call after the while-loop (all_finish=True).
        carry  : final carry (contains z_H of the last step already recorded)
        labels : (B, L_seq) ground-truth token ids
        ratings : optional (B,) array of difficulty ratings for this batch
        """
        if not self._batch_open:
            return

        # Determine per-puzzle correctness from final prediction
        preds = final_preds
        mask = (labels != -100)
        correct = ((preds == labels) & mask).sum(-1) == mask.sum(-1)  # (B,) bool
        correct_np = correct.cpu().numpy()

        B = len(self._z_H_per_step_batch)
        for b in range(B):
            z_H_steps = self._z_H_per_step_batch[b]   # list of (D,) arrays
            z_L_steps = self._z_L_per_step_batch[b]

            has_trajectory = len(z_H_steps) > 0 and len(z_L_steps) > 0
            has_rating     = (ratings is not None) and (b < len(ratings))
            has_pos        = len(self._z_H_pos_per_step_batch[b]) > 0
            has_given_mask = b < len(self._given_masks_batch)

            if not (has_trajectory and has_rating and has_pos and has_given_mask):
                self.n_skipped += 1
                continue

            z_H_traj = np.stack(z_H_steps, axis=0)         # (T, D)
            z_L_traj = np.stack(z_L_steps, axis=0)
            
            self.ratings.append(int(ratings[b]))
            self.trajectories.append(z_H_traj)
            self.correct_flags.append(bool(correct_np[b]))
            self.z_L_trajectories.append(z_L_traj)

            if z_H_traj.shape[0] > 1:
                diffs = np.linalg.norm(np.diff(z_H_traj, axis=0), axis=-1)   # (T-1,)
            else:
                diffs = np.array([0.0])
            self.residuals.append(diffs)

            if z_L_traj.shape[0] > 1:
                z_L_diffs = np.linalg.norm(np.diff(z_L_traj, axis=0), axis=-1)   # (T-1,)
            else:
                z_L_diffs = np.array([0.0])
            self.z_L_residuals.append(z_L_diffs)

            pos_steps = self._z_H_pos_per_step_batch[b]   # list of (L, D)
            if len(pos_steps) > 1:
                pos_traj = np.stack(pos_steps, axis=0)     # (T, L, D)
                # L2 norm over D，不做 mean-pool
                pos_diffs = np.linalg.norm(
                    np.diff(pos_traj, axis=0), axis=-1
                ) / math.sqrt(pos_traj.shape[-1])          # (T-1, L)
            else:
                pos_diffs = np.zeros((1, pos_steps[0].shape[0]))
            self.pos_residuals.append(pos_diffs)
            self.given_masks.append(self._given_masks_batch[b])

        self._batch_open = False

    @property
    def n_samples(self):
        return len(self.trajectories)


def run_z_analysis(
    collector: ZAnalysisCollector,
    config: EvalConfig,
    save_dir: str,
    rank: int,
    train_state: TrainState,
    wandb_step: int = 0,
):
    """Fit PCA, save all plots, and log everything to wandb. Only runs on rank 0."""
    if rank != 0:
        return

    os.makedirs(save_dir, exist_ok=True)
    n = collector.n_samples
    n_skipped = collector.n_skipped
    n_correct = sum(collector.correct_flags)
    puzzle_emb_len = config.arch.puzzle_emb_len if hasattr(config.arch, "puzzle_emb_len") else 0
    print(f"\n[z_analysis] {n} puzzles  |  accuracy = {n_correct}/{n} = {n_correct/n:.2%}")
    print(f"ratings count:          {len(collector.ratings)}")
    print(f"trajectories count:     {len(collector.trajectories)}")
    print(f"z_L_trajectories count: {len(collector.z_L_trajectories)}")
    print(f"correct_flags count:    {len(collector.correct_flags)}")
    print(f"skipped count:          {n_skipped}")

    # ── save raw data ──────────────────────────────────────────────────────
    np.savez_compressed(
        os.path.join(save_dir, "z_raw.npz"),
        correct_flags=np.array(collector.correct_flags),
        trajectories=np.array(collector.trajectories, dtype=object),
    )
    print(f"[z_analysis] saved z_raw.npz")

    # ── PCA ────────────────────────────────────────────────────────────────
    max_s = config.z_analysis_max_samples_pca
    trajs = collector.trajectories
    z_L_trajs = collector.z_L_trajectories
    if min(len(trajs), len(z_L_trajs)) > max_s:
        rng = np.random.default_rng(0)
        idxs = rng.choice(min(len(trajs), len(z_L_trajs)), max_s, replace=False).tolist()
    else:
        idxs = list(range(min(len(trajs), len(z_L_trajs))))

    sub_trajs = [trajs[i] for i in idxs]
    sub_z_L_trajs = [z_L_trajs[i] for i in idxs]
    sub_flags = [collector.correct_flags[i] for i in idxs]

    all_z_H = np.concatenate(sub_trajs, axis=0)    # (sum_T, D)
    all_z_L = np.concatenate(sub_z_L_trajs, axis=0)
    sample_ids = np.concatenate(
        [np.full(t.shape[0], si, dtype=int) for si, t in enumerate(sub_trajs)]
    )

    cov_z_H = np.cov(all_z_H, rowvar=False)
    pr_z_H = float((np.trace(cov_z_H)**2) / np.trace(cov_z_H.dot(cov_z_H)))
    cov_z_L = np.cov(all_z_L, rowvar=False)
    pr_z_L = float((np.trace(cov_z_L)**2) / np.trace(cov_z_L.dot(cov_z_L)))

    n_comp = min(config.z_analysis_pca_components, all_z_H.shape[1], all_z_H.shape[0])
    pca = PCA(n_components=n_comp, random_state=0)
    proj = pca.fit_transform(all_z_H)   # (sum_T, n_comp)

    h_init = train_state.model.model.inner.H_init
    h_init_vec = h_init.float().cpu().numpy().reshape(1, -1)
    proj_hinit_single = pca.transform(h_init_vec)[:, :2]   # (1, 2)
    proj_inits = np.repeat(proj_hinit_single, len(sub_trajs), axis=0)  # (N, 2)

    n_comp_L = min(config.z_analysis_pca_components, all_z_L.shape[1], all_z_L.shape[0])
    pca_L = PCA(n_components=n_comp_L, random_state=0)
    proj_z_L = pca_L.fit_transform(all_z_L)   # (sum_T, n_comp_L)

    l_init = train_state.model.model.inner.L_init
    l_init_vec = l_init.float().cpu().numpy().reshape(1, -1)
    proj_linit_single = pca_L.transform(l_init_vec)[:, :2]   # (1, 2)
    proj_inits_L = np.repeat(proj_linit_single, len(sub_trajs), axis=0)  # (N, 2)

    # ── scalar metrics ─────────────────────────────────────────────────────
    wandb_log: dict = {}

    wandb_log["z_analysis/n_samples"]       = n
    wandb_log["z_analysis/n_skipped"]       = n_skipped
    wandb_log["z_analysis/accuracy"]        = n_correct / n if n > 0 else 0.0
    wandb_log["z_analysis/hinit_pc1"] = float(proj_hinit_single[0, 0])
    wandb_log["z_analysis/hinit_pc2"] = float(proj_hinit_single[0, 1])
    wandb_log["z_analysis/pca_pr_z_H"] = pr_z_H
    wandb_log["z_analysis/pca_pr_z_L"] = pr_z_L
    wandb_log["z_analysis/pca_pc1_var_pct"] = float(pca.explained_variance_ratio_[0] * 100)
    wandb_log["z_analysis/pca_pc2_var_pct"] = float(pca.explained_variance_ratio_[1] * 100)
    wandb_log["z_analysis/pca_top2_cumvar_pct"] = float(
        pca.explained_variance_ratio_[:2].sum() * 100
    )

    # Per-group residual and displacement stats
    correct_idxs = [i for i, f in enumerate(collector.correct_flags) if f]
    incorrect_idxs = [i for i, f in enumerate(collector.correct_flags) if not f]

    for group_name, group_idxs in [("correct", correct_idxs), ("incorrect", incorrect_idxs)]:
        if not group_idxs:
            continue
        # Mean final-step residual (last entry of each residual array)
        final_resids = [collector.residuals[i][-1] for i in group_idxs]
        mean_resids = [collector.residuals[i].mean() for i in group_idxs]
        # Total displacement
        disps = []
        for i in group_idxs:
            t = collector.trajectories[i]
            disps.append(float(np.linalg.norm(np.diff(t, axis=0), axis=-1).sum()) if t.shape[0] > 1 else 0.0)

        wandb_log[f"z_analysis/{group_name}/final_residual_mean"] = float(np.mean(final_resids))
        wandb_log[f"z_analysis/{group_name}/final_residual_std"]  = float(np.std(final_resids))
        wandb_log[f"z_analysis/{group_name}/mean_residual_mean"]  = float(np.mean(mean_resids))
        wandb_log[f"z_analysis/{group_name}/displacement_mean"]   = float(np.mean(disps))
        wandb_log[f"z_analysis/{group_name}/displacement_std"]    = float(np.std(disps))

    # Per-step residual table (wandb.Table for line chart)
    residual_table = _make_residual_table(collector.residuals, collector.correct_flags)
    if residual_table is not None:
        wandb_log["z_analysis/residual_by_step"] = residual_table

    # PCA explained variance table
    ev_table = wandb.Table(
        columns=["pc", "explained_var_pct", "cumulative_var_pct"],
        data=[
            [i + 1,
             float(pca.explained_variance_ratio_[i] * 100),
             float(pca.explained_variance_ratio_[:i+1].sum() * 100)]
            for i in range(len(pca.explained_variance_ratio_))
        ]
    )
    wandb_log["z_analysis/pca_explained_variance"] = ev_table

    # ── plots → wandb.Image ────────────────────────────────────────────────
    wandb_log["z_analysis/z_H_pca_split"] = _save_wandb(_plot_pca_split(proj, sample_ids, sub_flags, pca, save_dir, z_label="z_H"), save_dir, "z_H_trajectory_pca_split.png")
    wandb_log["z_analysis/z_H_pca_combined"] = _save_wandb(_plot_pca_combined(proj, sample_ids, sub_flags, pca, save_dir, z_label="z_H"), save_dir, "z_H_trajectory_pca_combined.png")
    wandb_log["z_analysis/z_H_forward_residual"] = _save_wandb(_plot_forward_residual(collector.residuals, collector.correct_flags, save_dir, z_label="z_H"), save_dir, "z_H_forward_residual.png")
    wandb_log["z_analysis/z_H_pca_variance"] = _save_wandb(_plot_pca_variance(pca, save_dir, z_label="z_H"), save_dir, "z_H_pca_variance.png")
    wandb_log["z_analysis/z_H_displacement_hist"] = _save_wandb(_plot_displacement_hist(collector.trajectories, collector.correct_flags, save_dir, z_label="z_H"), save_dir, "z_H_displacement_histogram.png")
    wandb_log["z_analysis/z_H_pca_step1_final"] = _save_wandb(_plot_step1_vs_final(proj, sample_ids, sub_flags, save_dir, z_label="z_H"), save_dir, "z_H_pca_step1_vs_final.png")
    wandb_log["z_analysis/z_H_pca_hinit_final"] = _save_wandb(_plot_hinit_vs_final(proj_inits, proj, sample_ids, sub_flags, save_dir, z_label="z_H"), save_dir, "z_H_pca_hinit_vs_final.png")
    wandb_log["z_analysis/z_H_pos_residual_heatmap_given"] = _save_wandb(_plot_pos_residual_heatmap_given(collector.pos_residuals, collector.given_masks, collector.correct_flags, puzzle_emb_len=puzzle_emb_len), save_dir, "z_H_pos_residual_heatmap_given.png")
    wandb_log["z_analysis/z_H_pos_residual_heatmap_empty"] = _save_wandb(_plot_pos_residual_heatmap_empty(collector.pos_residuals, collector.given_masks, collector.correct_flags, puzzle_emb_len=puzzle_emb_len), save_dir, "z_H_pos_residual_heatmap_empty.png")
    wandb_log["z_analysis/z_H_pos_residual_by_step"] = _save_wandb(_plot_pos_residual_by_step(collector.pos_residuals, collector.given_masks, collector.correct_flags, puzzle_emb_len=puzzle_emb_len), save_dir, "z_H_pos_residual_by_step.png")
    
    wandb_log["z_analysis/z_L_pca_split"] = _save_wandb(_plot_pca_split(proj_z_L, sample_ids, sub_flags, pca_L, save_dir, z_label="z_L"), save_dir, "z_L_trajectory_pca_split.png")
    wandb_log["z_analysis/z_L_pca_combined"] = _save_wandb(_plot_pca_combined(proj_z_L, sample_ids, sub_flags, pca_L, save_dir, z_label="z_L"), save_dir, "z_L_trajectory_pca_combined.png")
    wandb_log["z_analysis/z_L_forward_residual"] = _save_wandb(_plot_forward_residual(collector.z_L_residuals, collector.correct_flags, save_dir, z_label="z_L"), save_dir, "z_L_forward_residual.png")
    wandb_log["z_analysis/z_L_pca_variance"] = _save_wandb(_plot_pca_variance(pca_L, save_dir, z_label="z_L"), save_dir, "z_L_pca_variance.png")
    wandb_log["z_analysis/z_L_displacement_hist"] = _save_wandb(_plot_displacement_hist(collector.z_L_trajectories, collector.correct_flags, save_dir, z_label="z_L"), save_dir, "z_L_displacement_histogram.png")
    wandb_log["z_analysis/z_L_pca_step1_final"] = _save_wandb(_plot_step1_vs_final(proj_z_L, sample_ids, sub_flags, save_dir, z_label="z_L"), save_dir, "z_L_pca_step1_vs_final.png")
    wandb_log["z_analysis/z_L_pca_hinit_final"] = _save_wandb(_plot_hinit_vs_final(proj_inits_L, proj_z_L, sample_ids, sub_flags, save_dir, z_label="z_L"), save_dir, "z_L_pca_hinit_vs_final.png")
    if collector.ratings:
        wandb_log["z_analysis/rating_distribution"] = _save_wandb(_plot_rating_distribution(collector.ratings, collector.correct_flags, save_dir), save_dir, "rating_distribution.png")
        wandb_log["z_analysis/residual_vs_rating"] = _save_wandb(_plot_residual_vs_rating(collector.residuals, collector.ratings, collector.correct_flags, save_dir), save_dir, "residual_vs_rating.png")
        wandb_log["z_analysis/accuracy_vs_rating"] = _save_wandb(_plot_accuracy_vs_rating(collector.correct_flags, collector.ratings, save_dir), save_dir, "accuracy_vs_rating.png")
        wandb_log["z_analysis/residual_colormap_rating"] = _save_wandb(_plot_residual_by_rating_colormap(collector.residuals, collector.ratings, collector.correct_flags, save_dir), save_dir, "residual_colormap_rating.png")
    # Drop None values (plots that returned None due to insufficient data)
    wandb_log = {k: v for k, v in wandb_log.items() if v is not None}

    # ── log to wandb ───────────────────────────────────────────────────────
    if wandb.run is not None:
        wandb.log(wandb_log, step=wandb_step)
        print(f"[z_analysis] logged {len(wandb_log)} entries to wandb (step={wandb_step})")
    else:
        print("[z_analysis] wandb.run is None, skipping wandb.log")

    print(f"[z_analysis] all plots saved to {save_dir}/")


def _make_residual_table(residuals: List[np.ndarray], flags: List[bool]) -> Optional[wandb.Table]:
    """Build a wandb.Table of mean residual per step, split by correct/incorrect."""
    if not residuals:
        return None
    max_T = max(r.shape[0] for r in residuals)
    if max_T == 0:
        return None

    correct_idxs = [i for i, f in enumerate(flags) if f]
    incorrect_idxs = [i for i, f in enumerate(flags) if not f]

    def _padded_mean(idxs):
        if not idxs:
            return np.full(max_T, float("nan"))
        arr = np.array([
            np.pad(residuals[i], (0, max_T - residuals[i].shape[0]), constant_values=np.nan)
            for i in idxs
        ])
        return np.nanmean(arr, axis=0)

    mean_correct = _padded_mean(correct_idxs)
    mean_incorrect = _padded_mean(incorrect_idxs)

    table = wandb.Table(columns=["step", "correct_mean_residual", "incorrect_mean_residual"])
    for t in range(max_T):
        table.add_data(
            t + 1,
            float(mean_correct[t])   if not np.isnan(mean_correct[t])   else None,
            float(mean_incorrect[t]) if not np.isnan(mean_incorrect[t]) else None,
        )
    return table


def _save_wandb(fig, save_dir: str, filename: str) -> Optional[wandb.Image]:
    """Save a matplotlib figure to disk and return a wandb.Image. Returns None if fig is None."""
    if fig is None:
        return None
    path = os.path.join(save_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[z_analysis] saved {filename}")
    return wandb.Image(path)

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE  (original function, minimally extended)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    config: EvalConfig,
    train_state: TrainState,
    eval_loader: DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
    rank: int,
    world_size: int,
    cpu_group: Optional[dist.ProcessGroup],
):
    reduced_metrics = None
    progress_bar = None
    if rank == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps, desc="Evaluating", unit="batch")

    # ── z analysis setup (new) ────────────────────────────────────────────
    z_collector = None
    if config.z_analysis and rank == 0:
        z_collector = ZAnalysisCollector()
        print("[z_analysis] Enabled")
    z_batches_collected = 0

    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        if z_collector is not None:
            return_keys.add("preds")

        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)

        # Run evaluation
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}

        save_preds = {}

        metric_keys = []
        metric_values = None

        carry = None
        processed_batches: int = 0
        
        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            if rank == 0 and progress_bar is not None:
                progress_bar.update(processed_batches - progress_bar.n)  # type: ignore
            
            # To device
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)  # type: ignore

            # ── z collector: begin batch (new) ────────────────────────────
            collect_this_batch = (
                z_collector is not None
                and z_batches_collected < config.z_analysis_max_batches
            )
            if collect_this_batch:
                z_collector.begin_batch(batch, config)

            # Forward
            inference_steps = 0
            while True:
                carry, loss, metrics, preds, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                inference_steps += 1

                # ── z collector: record step (new) ────────────────────────
                if collect_this_batch:
                    z_collector.record_step(carry)

                if all_finish:
                    break

            # ── z collector: end batch (new) ──────────────────────────────
            if collect_this_batch:
                batch_ratings = batch.get("ratings")
                if batch_ratings is not None:
                    batch_ratings = batch_ratings.cpu().numpy()
                with torch.inference_mode():
                    z_collector.end_batch(carry, carry.current_data["labels"], preds["preds"], ratings=batch_ratings)
                z_batches_collected += 1

            if rank == 0 and progress_bar is not None:
                progress_bar.set_description(f"Batch {processed_batches}: {set_name} | Inference steps: {inference_steps}")

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        save_preds.setdefault(k, [])
                        save_preds[k].append(v.cpu())  # Move to CPU for saving GPU memory

            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)

            del carry, loss, preds, batch, all_finish

            # Aggregate metrics
            set_id = set_ids[set_name]

            if metric_values is None:
                metric_keys = list(
                    sorted(metrics.keys())
                )  # Sort keys to guarantee all processes use the same order.
                metric_values = torch.zeros(
                    (len(set_ids), len(metrics.values())), dtype=torch.float32, device="cuda"
                )

            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])

            del metrics

        # concatenate save preds
        save_preds = {k: torch.cat(v, dim=0) for k, v in save_preds.items()}

        # Save preds
        if config.checkpoint_path is not None and len(save_preds):
            # Each rank save predictions independently
            os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
            torch.save(
                save_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}")
            )

        del save_preds

        # Reduce to rank 0
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {
                        metric_name: reduced_metrics[set_id, metric_id]
                        for metric_id, metric_name in enumerate(metric_keys)
                    }
                    for set_id, set_name in enumerate(set_ids)
                }

                # Postprocess
                for set_name, m in reduced_metrics.items():
                    count = m.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in m.items()}

        # Run evaluators
        if rank == 0:
            print(f"\nRunning {len(evaluators)} evaluator(s)...")
            
        for i, evaluator in enumerate(evaluators):
            if rank == 0:
                print(f"Running evaluator {i+1}/{len(evaluators)}: {evaluator.__class__.__name__}")
                
            # Path for saving
            evaluator_save_path = None
            if config.checkpoint_path is not None:
                evaluator_save_path = os.path.join(
                    config.checkpoint_path,
                    f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}",
                )
                os.makedirs(evaluator_save_path, exist_ok=True)

            # Run and log
            metrics = evaluator.result(evaluator_save_path, rank=rank, world_size=world_size, group=cpu_group)
            if rank == 0 and metrics is not None:
                if reduced_metrics is None:
                    reduced_metrics = {}

                reduced_metrics.update(metrics)
                print(f"  Completed {evaluator.__class__.__name__}")
                
        if rank == 0:
            print("All evaluators completed!")

    # ── z analysis: run PCA and save plots (new) ──────────────────────────
    if z_collector is not None and z_collector.n_samples > 0:
        z_save_dir = os.path.join(
            config.checkpoint_path or "checkpoints/z_analysis",
            f"z_analysis_step_{train_state.step}"
        )
        run_z_analysis(z_collector, config, z_save_dir, rank, train_state, wandb_step=train_state.step)

    return reduced_metrics

def save_code_and_config(config: EvalConfig):
    if config.checkpoint_path is None or wandb.run is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)

            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    # Log code
    wandb.run.log_code(config.checkpoint_path)


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> EvalConfig:
    objects = [None]
    if rank == 0:
        config = EvalConfig(**hydra_config)  # type: ignore

        # Naming
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_paths[0]).capitalize()}-eval-torch"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore


@hydra.main(config_path="config", config_name="cfg_eval", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1
    CPU_PROCESS_GROUP = None

    # Initialize distributed if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        
        # CPU GLOO process group
        CPU_PROCESS_GROUP = dist.new_group(backend="gloo")
        assert (
            dist.get_rank(CPU_PROCESS_GROUP) == RANK and dist.get_world_size(CPU_PROCESS_GROUP) == WORLD_SIZE
        )

    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed + RANK)

    # Dataset
    eval_loader, eval_metadata = create_dataloader(
        config, 
        split="test", 
        test_set_mode=True, 
        epochs_per_iter=1, 
        global_batch_size=config.global_batch_size, 
        rank=RANK, 
        world_size=WORLD_SIZE
    )
    try:
        evaluators = create_evaluators(config, eval_metadata)
    except:
        print("No evaluator found")
        evaluators = []

    # Train state
    train_state = init_train_state(config, eval_metadata, rank=RANK, world_size=WORLD_SIZE)

    # Progress bar and logger
    if RANK == 0:
        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump(), settings=wandb.Settings(_disable_stats=True))  # type: ignore
        wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)

    train_state.model.eval()

    # Run evaluation
    metrics = evaluate(
        config, 
        train_state, 
        eval_loader, 
        eval_metadata, 
        evaluators,
        rank=RANK, 
        world_size=WORLD_SIZE, 
        cpu_group=CPU_PROCESS_GROUP
    )

    if RANK == 0 and metrics is not None:
        wandb.log(metrics, step=train_state.step)

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()
