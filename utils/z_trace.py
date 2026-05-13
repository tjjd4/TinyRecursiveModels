import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch._dynamo
import ml_dtypes

from utils.math import logit_entropy


@dataclass
class ZTrace:
    H_cycles: int
    L_cycles: int
    puzzle_emb_len: int
    halt_max_steps: int
    rec_max_correct: int
    rec_max_incorrect: int
    cell_max_correct: int
    cell_max_incorrect: int

    trajectories: List[np.ndarray] = field(default_factory=list)  # (T, D)
    z_L_trajectories: List[np.ndarray] = field(default_factory=list)  # (T, D)
    correct_flags: List[bool] = field(default_factory=list)
    ratings: List[int] = field(default_factory=list)
    given_masks: List[np.ndarray] = field(default_factory=list)  # (81,)
    residuals: List[np.ndarray] = field(default_factory=list)  # (T-1,)
    z_L_residuals: List[np.ndarray] = field(default_factory=list)  # (T-1,)
    pos_residuals: List[np.ndarray] = field(default_factory=list)  # (T-1, L)

    # recursion level
    rec_z_H: List[np.ndarray] = field(default_factory=list)
    rec_z_L: List[np.ndarray] = field(default_factory=list)
    rec_correct_flags: List[bool] = field(default_factory=list)
    rec_ratings: List[int] = field(default_factory=list)

    # step-wise metrics
    step_cell_acc: List[np.ndarray] = field(default_factory=list)       # (T,) overall cell accuracy
    step_empty_acc: List[np.ndarray] = field(default_factory=list)      # (T,) empty cell accuracy  
    step_given_acc: List[np.ndarray] = field(default_factory=list)      # (T,) given cell accuracy
    step_pred_stable: List[int] = field(default_factory=list)           # scalar: earliest stable step
    step_preds_all: List[np.ndarray] = field(default_factory=list)      # (T, 81) argmax predictions

    _rec_n_correct: int = 0
    _rec_n_incorrect: int = 0

    n_stored: int = 0
    n_skipped: int = 0
    is_all_trace_collected: bool = False

    def __init__(self, H_cycles: int, L_cycles: int, puzzle_emb_len: int, halt_max_steps: int, rec_max_correct: int, rec_max_incorrect: int, cell_max_correct: int, cell_max_incorrect: int):
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.puzzle_emb_len = puzzle_emb_len
        self.halt_max_steps = halt_max_steps
        self.rec_max_correct = rec_max_correct
        self.rec_max_incorrect = rec_max_incorrect
        self.cell_max_correct = cell_max_correct
        self.cell_max_incorrect = cell_max_incorrect

        # record
        self._rec_z_H = []
        self._rec_z_L = []
        self._step_z_H = []
        self._step_z_L = []
        self._step_preds = []
        self._step_z_L_preds = []
        self._step_z_H_entropy = []
        self._step_z_L_entropy = []

        self._step_output_top2_probs = []
        self._step_output_argsort_idx = []

        # step level trajectories
        self.trajectories = []
        self.z_L_trajectories = []
        # self.z_H_puzzle_emb_trajs = []
        # self.z_L_puzzle_emb_trajs = []
        self.z_H_halt_trajs = []
        self.z_L_halt_trajs = []
        self.z_H_ctx_trajs = []
        self.z_L_ctx_trajs = []
        self.correct_flags = []
        self.ratings = []
        self.given_masks = []
        self.residuals = []
        self.z_L_residuals = []
        self.pos_residuals = []

        # step cell z_H and z_L
        self.z_H_cell_trajs = []
        self.z_L_cell_trajs = []
        self.cell_correct_flags = []
        self.cell_collected_idx = []

        # recursion level z_H and z_L
        self.rec_z_H = []
        self.rec_z_L = []
        self.rec_correct_flags = []
        self.rec_ratings = []

        # count of valid, given, empty cells
        self.n_valid = []
        self.n_given = []
        self.n_empty = []

        # count of correct predictions
        self.step_cell_correct_count = []
        self.step_given_correct_count = []
        self.step_empty_correct_count = []

        # how many steps the empty cell is in error state
        self.empty_error_step_count = []
        # number of unique predictions across steps per empty cell (1 = locked, >1 = changing)
        self.empty_pred_unique_count = []
        # step-to-step prediction value changes per empty cell (0..T-1) — frequency, not diversity
        self.empty_pred_change_count = []
        # number of wrong→correct transitions per empty cell
        self.empty_flips_to_correct = []
        # number of correct→wrong transitions per empty cell
        self.empty_flips_to_wrong = []
        # whether final-step prediction is correct per empty cell (bool)
        self.empty_final_correct = []
        # per-step fraction of empty cells whose prediction changed at boundary t→t+1, shape (T-1,)
        self.step_empty_change_rate = []

        # first time predicts all valid cells correctly
        self.step_first_correct = []

        # z_H logit lens
        self.step_cell_acc = []
        self.step_empty_acc = []
        self.step_given_acc = []
        self.step_pred_stable = []
        self.step_preds_all = []
        # z_L logit lens accuracy
        self.step_z_L_cell_acc = []
        self.step_z_L_empty_acc = []
        self.step_z_L_given_acc = []
        self.step_z_L_preds_all = []

        # z_H vs z_L comparison
        # empty cells
        self.step_empty_agree_correct = []
        self.step_empty_both_wrong_same = []
        self.step_empty_both_wrong_diff = []
        self.step_empty_only_z_H_correct = []
        self.step_empty_only_z_L_correct = []
        # given cells
        self.step_given_agree_correct = []
        self.step_given_both_wrong_same = []
        self.step_given_both_wrong_diff = []
        self.step_given_only_z_H_correct = []
        self.step_given_only_z_L_correct = []
        
        # cosine similarity
        self.step_empty_cos_sim = []
        self.step_given_cos_sim = []

        self.step_output_empty_top1_prob = []
        self.step_output_empty_margin = []
        self.step_output_given_top1_prob = []
        self.step_output_given_margin = []
        self.step_output_correct_rank = []  # List[np.ndarray], each (T, 81) int8
        # entropy
        self.step_z_H_empty_entropy = []
        self.step_z_H_given_entropy = []
        self.step_z_L_empty_entropy = []
        self.step_z_L_given_entropy = []

        self._rec_n_correct = 0
        self._rec_n_incorrect = 0
        self._cell_n_correct = 0
        self._cell_n_incorrect = 0

        self.n_stored = 0
        self.n_skipped = 0
        self.is_all_trace_collected = False

    @property
    def snapshots_per_step(self) -> int:
        """
        Number of record() calls per supervision step.
        Each H cycle: L_cycles z_L updates + 1 z_H update = L_cycles + 1
        Total: H_cycles * (L_cycles + 1)
        """
        return self.H_cycles * (self.L_cycles + 1)


    def record_z_L(self, z_L: torch.Tensor) -> None:
        """Called after z_L update. Only records z_L snapshot."""
        if self.is_all_trace_collected:
            return
        self._rec_z_L.append(z_L.detach())

    def record_z_H(self, z_H: torch.Tensor) -> None:
        """Called after z_H update. Only records z_H snapshot."""
        if self.is_all_trace_collected:
            return
        self._rec_z_H.append(z_H.detach())

    def record_step(self, z_H: torch.Tensor, z_L: torch.Tensor, output: torch.Tensor, z_L_logit_lens: torch.Tensor) -> None:
        self._step_z_H.append(z_H.detach())
        self._step_z_L.append(z_L.detach())
        self._step_preds.append(torch.argmax(output, dim=-1))
        self._step_z_L_preds.append(torch.argmax(z_L_logit_lens, dim=-1))

        probs = torch.softmax(output, dim=-1)                  # logits → probs，(B, seq_len, C)
        top2_vals, _ = torch.topk(probs, k=2, dim=-1)  # (B, seq_len, 2)
        self._step_output_top2_probs.append(top2_vals.detach())
        argsort_desc = probs.argsort(dim=-1, descending=True)   # (B, seq_len, C)
        self._step_output_argsort_idx.append(argsort_desc.detach())

        # logit lens entropy
        self._step_z_H_entropy.append(logit_entropy(output))
        self._step_z_L_entropy.append(logit_entropy(z_L_logit_lens))


    def clear_batch_buffers(self) -> None:
        """Call before each batch."""
        self._rec_z_H.clear()
        self._rec_z_L.clear()
        self._step_z_H.clear()
        self._step_z_L.clear()
        self._step_preds.clear()
        self._step_z_L_preds.clear()
        self._step_z_H_entropy.clear()
        self._step_z_L_entropy.clear()
        self._step_output_top2_probs.clear()
        self._step_output_argsort_idx.clear()
        # Update flag: should we collect recursion trace for the next batch?
        # Collect if we haven't reached the max for either correct or incorrect
        self.is_all_trace_collected = (
            self._rec_n_correct >= self.rec_max_correct and
            self._rec_n_incorrect >= self.rec_max_incorrect
        )


    @property
    def n_samples(self) -> int:
        return len(self.trajectories)


    def process_batch(
        self,
        batch_inputs_np: np.ndarray,           # (B, seq_len) int
        preds_np: np.ndarray,                  # (B, seq_len) int
        labels_np: np.ndarray,                 # (B, seq_len) int
        ratings_np: Optional[np.ndarray],      # (B,) int or None
        halt_steps_np: np.ndarray,             # (B,) int from carry.steps
    ) -> None:

        if not self._step_z_H:
            return

        # record_step() is called once per supervision step
        n_steps = len(self._step_z_H)

        # 驗證
        if n_steps != self.halt_max_steps:
            print(f"[ZTrace] WARNING: n_steps={n_steps} != halt_max_steps={self.halt_max_steps}")

        step_preds_np = torch.stack(self._step_preds).cpu().numpy()  # (n_steps, B, seq_len)
        step_z_L_preds_np = torch.stack(self._step_z_L_preds).cpu().numpy()  # (n_steps, B, seq_len)

        step_z_H_t = torch.stack(self._step_z_H).cpu()
        step_z_H_np = step_z_H_t.view(torch.int16).numpy().view(ml_dtypes.bfloat16)  # (n_steps, B, seq_len, D)
        step_z_L_t = torch.stack(self._step_z_L).cpu()
        step_z_L_np = step_z_L_t.view(torch.int16).numpy().view(ml_dtypes.bfloat16)  # (n_steps, B, seq_len, D)
        step_z_H_ent_t = torch.stack(self._step_z_H_entropy).cpu()
        step_z_H_ent_np = step_z_H_ent_t.view(torch.int16).numpy().view(ml_dtypes.bfloat16)  # (n_steps, B, seq_len)
        step_z_L_ent_t = torch.stack(self._step_z_L_entropy).cpu()
        step_z_L_ent_np = step_z_L_ent_t.view(torch.int16).numpy().view(ml_dtypes.bfloat16)  # (n_steps, B, seq_len)

        # top 2 probs and idx
        step_top2_probs_np = torch.stack(self._step_output_top2_probs).cpu().float().numpy()
        step_argsort_np = torch.stack(self._step_output_argsort_idx).cpu().numpy()

        # rec_z_H/L are called snapshots_per_step times per supervision step
        if self._rec_z_H:
            rec_z_H_np = torch.stack(self._rec_z_H).cpu().float().numpy()  # (n_steps * S, B, L, D)
            rec_z_L_np = torch.stack(self._rec_z_L).cpu().float().numpy()  # (n_steps * S, B, L, D)
        else:
            rec_z_H_np = None
            rec_z_L_np = None

        B = batch_inputs_np.shape[0]
        mask = (labels_np != -100)
        correct_np = ((preds_np == labels_np) & mask).sum(-1) == mask.sum(-1)

        for b in range(B):
            # Skip padding samples (all labels are -100)
            if not mask[b].any():
                self.n_skipped += 1
                continue

            is_correct = bool(correct_np[b])
            if ratings_np is None:
                rating = -1
            else:
                rating = int(ratings_np[b])

            T_actual = max(1, min(int(halt_steps_np[b]), n_steps))

            # Extract trajectory for this sample: (T_actual, L, D)
            z_H_last = step_z_H_np[:T_actual, b]
            z_L_last = step_z_L_np[:T_actual, b]

            D = z_H_last.shape[-1]
            L = z_H_last.shape[-2]

            cell_start = self.puzzle_emb_len  # TRM: 16, HRM: 1

            z_H_cell = z_H_last[:, cell_start:, :]   # (T_actual, 81, D)
            z_L_cell = z_L_last[:, cell_start:, :]   # (T_actual, 81, D)

            z_H_puzzle_emb = z_H_last[:, 0:cell_start, :]   # (T_actual, puzzle_emb_len, D)
            z_L_puzzle_emb = z_L_last[:, 0:cell_start, :]   # (T_actual, puzzle_emb_len, D)

            # mean-pool over sequence positions → (T_actual, D)
            z_H_cell_traj = z_H_cell.mean(axis=1)
            z_L_cell_traj = z_L_cell.mean(axis=1)
            # z_H_puzzle_emb_traj = z_H_puzzle_emb.mean(axis=1)
            # z_L_puzzle_emb_traj = z_L_puzzle_emb.mean(axis=1)
            z_H_halt_traj = z_H_puzzle_emb[:, 0, :].copy()         # (T_actual, D)
            z_L_halt_traj = z_L_puzzle_emb[:, 0, :].copy()         # (T_actual, D)
            z_H_ctx_traj = z_H_puzzle_emb[:, 1:, :].mean(axis=1)  # (T_actual, D)
            z_L_ctx_traj = z_L_puzzle_emb[:, 1:, :].mean(axis=1)  # (T_actual, D)

            # given mask from first 81 cell tokens
            given = batch_inputs_np[b, :] != 1  # (81,)

            # step-wise residuals
            if T_actual > 1:
                diffs = np.linalg.norm(np.diff(z_H_cell_traj, axis=0), axis=-1)       # (T-1,)
                z_L_diffs = np.linalg.norm(np.diff(z_L_cell_traj, axis=0), axis=-1)       # (T-1,)
                pos_diffs = (np.linalg.norm(np.diff(z_H_cell, axis=0), axis=-1) / math.sqrt(D)) # (T-1, L)
            else:
                diffs = np.array([0.0])
                z_L_diffs = np.array([0.0])
                pos_diffs = np.zeros((1, L))

            sample_step_preds = step_preds_np[:T_actual, b, :]   # (T_actual, 81)
            cell_labels = labels_np[b, :]                         # (81,)
            cell_mask = (cell_labels != -100)                       # (81,)
            empty = ~given & cell_mask

            # per-step correctness: (T_actual, 81) bool
            per_step_correct = (sample_step_preds == cell_labels[None, :]) & cell_mask[None, :]

            n_valid = cell_mask.sum()
            n_given = (given & cell_mask).sum()
            n_empty = empty.sum()
            given_valid = given & cell_mask

            cell_correct_count = per_step_correct[:, cell_mask].sum(axis=1).astype(np.float32)
            given_correct_count = per_step_correct[:, given_valid].sum(axis=1).astype(np.float32)
            empty_correct_count = per_step_correct[:, empty].sum(axis=1).astype(np.float32)

            cell_acc = cell_correct_count / max(n_valid, 1)
            given_acc = given_correct_count / max(n_given, 1)
            empty_acc = empty_correct_count / max(n_empty, 1)

            empty_step_preds = sample_step_preds[:, empty]            # (T_actual, n_empty)
            empty_per_step_correct = per_step_correct[:, empty]        # (T_actual, n_empty) bool
            empty_error_step_count = (~empty_per_step_correct).sum(axis=0).astype(np.int8)
            empty_pred_unique_count = np.array(
                [np.unique(col).size for col in empty_step_preds.T],
                dtype=np.int8,
            )
            empty_pred_change_count = (np.diff(empty_step_preds, axis=0) != 0).sum(axis=0).astype(np.int8)
            empty_flips_to_correct = (~empty_per_step_correct[:-1] &  empty_per_step_correct[1:]).sum(axis=0).astype(np.int8)
            empty_flips_to_wrong   = ( empty_per_step_correct[:-1] & ~empty_per_step_correct[1:]).sum(axis=0).astype(np.int8)
            empty_final_correct    = empty_per_step_correct[-1].copy()  # (n_empty,) bool
            if T_actual > 1 and empty_step_preds.shape[1] > 0:
                step_empty_change_rate = (np.diff(empty_step_preds, axis=0) != 0).mean(axis=1).astype(np.float32)
            else:
                step_empty_change_rate = np.zeros(max(T_actual - 1, 0), dtype=np.float32)

            # prediction stability: start from the last step and go backwards, the earliest step that makes the prediction no longer change
            stable_step = T_actual
            final_pred = sample_step_preds[-1]
            for k in range(T_actual - 2, -1, -1):
                if np.array_equal(sample_step_preds[k], final_pred):
                    stable_step = k+1
                else:
                    break

            first_correct_step = T_actual  # default
            for k in range(T_actual):
                if cell_correct_count[k] >= n_valid:
                    first_correct_step = k + 1
                    break
            
            # z_L logit lens
            sample_z_L_preds = step_z_L_preds_np[:T_actual, b, :]  # (T_actual, 81)

            z_L_per_step_correct = (sample_z_L_preds == cell_labels[None, :]) & cell_mask[None, :]

            z_L_cell_acc = z_L_per_step_correct[:, cell_mask].sum(axis=1).astype(np.float32) / max(n_valid, 1)
            z_L_given_acc = z_L_per_step_correct[:, given_valid].sum(axis=1).astype(np.float32) / max(n_given, 1)
            z_L_empty_acc = z_L_per_step_correct[:, empty].sum(axis=1).astype(np.float32) / max(n_empty, 1)

            # entropy
            ent_H = step_z_H_ent_np[:T_actual, b, :]  # (T_actual, )
            ent_L = step_z_L_ent_np[:T_actual, b, :]  # (T_actual, )
            z_H_empty_entropy = ent_H[:, empty].mean(axis=1).astype(np.float32)         # (T_actual,)
            z_H_given_entropy = ent_H[:, given_valid].mean(axis=1).astype(np.float32)
            z_L_empty_entropy = ent_L[:, empty].mean(axis=1).astype(np.float32)
            z_L_given_entropy = ent_L[:, given_valid].mean(axis=1).astype(np.float32)

            # z_H logit lens vs z_L logit lens
            both_correct = per_step_correct & z_L_per_step_correct          # (T, 81)
            both_wrong_same = (~per_step_correct & ~z_L_per_step_correct & (sample_step_preds == sample_z_L_preds))    # (T, 81)
            both_wrong_diff = (~per_step_correct & ~z_L_per_step_correct & (sample_step_preds != sample_z_L_preds))
            only_z_H_correct = per_step_correct & ~z_L_per_step_correct
            only_z_L_correct = ~per_step_correct & z_L_per_step_correct

            # per-step proportion of empty cells
            n_e = max(n_empty, 1)
            empty_agree_correct_rate = both_correct[:, empty].sum(axis=1).astype(np.float32) / n_e
            empty_both_wrong_same_rate = both_wrong_same[:, empty].sum(axis=1).astype(np.float32) / n_e
            empty_both_wrong_diff_rate = both_wrong_diff[:, empty].sum(axis=1).astype(np.float32) / n_e
            empty_only_z_H_correct_rate = only_z_H_correct[:, empty].sum(axis=1).astype(np.float32) / n_e
            empty_only_z_L_correct_rate = only_z_L_correct[:, empty].sum(axis=1).astype(np.float32) / n_e

            # per-step proportion of given cells
            n_g = max(n_given, 1)
            given_agree_correct_rate = both_correct[:, given_valid].sum(axis=1).astype(np.float32) / n_g
            given_both_wrong_same_rate = both_wrong_same[:, given_valid].sum(axis=1).astype(np.float32) / n_g
            given_both_wrong_diff_rate = both_wrong_diff[:, given_valid].sum(axis=1).astype(np.float32) / n_g
            given_only_z_H_correct_rate = only_z_H_correct[:, given_valid].sum(axis=1).astype(np.float32) / n_g
            given_only_z_L_correct_rate = only_z_L_correct[:, given_valid].sum(axis=1).astype(np.float32) / n_g

            # top-2 probs, margin, correct_rank
            sample_top2_probs = step_top2_probs_np[:T_actual, b, :, :]  # (T_actual, 81, 2)
            sample_argsort_idx = step_argsort_np[:T_actual, b, :, :]  # (T_actual, 81, 9)

            top1_prob = sample_top2_probs[..., 0]  # (T_actual, 81)
            top2_prob = sample_top2_probs[..., 1]  # (T_actual, 81)
            margin = top1_prob - top2_prob  # (T_actual, 81)

            C = sample_argsort_idx.shape[-1]                                  # 9
            rank_of_class = np.empty_like(sample_argsort_idx)                 # (T, 81, C)
            idx_t, idx_c = np.mgrid[:T_actual, :81]                      # broadcast grids
            for r in range(C):
                rank_of_class[idx_t, idx_c, sample_argsort_idx[..., r]] = r
            correct_rank = rank_of_class[
                idx_t, idx_c, np.broadcast_to(cell_labels[None, :], (T_actual, 81))
            ] + 1

            # empty cells aggregation
            output_empty_top1_prob = top1_prob[:, empty].mean(axis=1).astype(np.float32)  # (T,)
            output_empty_margin = margin[:, empty].mean(axis=1).astype(np.float32)  # (T,)

            # given cells aggregation
            output_given_top1_prob = top1_prob[:, given_valid].mean(axis=1).astype(np.float32)  # (T,)
            output_given_margin = margin[:, given_valid].mean(axis=1).astype(np.float32)  # (T,)

            # per-step cosine similarity between z_H and z_L
            empty_cos_sim = np.array([
                np.mean([
                    np.dot(z_H_cell[t, c], z_L_cell[t, c])
                    / (np.linalg.norm(z_H_cell[t, c]) * np.linalg.norm(z_L_cell[t, c]) + 1e-8)
                    for c in range(81) if empty[c]
                ]) if empty.any() else np.nan
                for t in range(T_actual)
            ])

            given_cos_sim = np.array([
                np.mean([
                    np.dot(z_H_cell[t, c], z_L_cell[t, c])
                    / (np.linalg.norm(z_H_cell[t, c]) * np.linalg.norm(z_L_cell[t, c]) + 1e-8)
                    for c in range(81) if given_valid[c]
                ]) if given_valid.any() else np.nan
                for t in range(T_actual)
            ])


            # cell level
            want_cell_c = is_correct and self._cell_n_correct < self.cell_max_correct
            want_cell_i = (not is_correct) and self._cell_n_incorrect < self.cell_max_incorrect
            if want_cell_c or want_cell_i:
                # z_H_cell / z_L_cell already exist as locals: (T_actual, 81, D) bfloat16
                self.z_H_cell_trajs.append(z_H_cell.astype(np.float16))
                self.z_L_cell_trajs.append(z_L_cell.astype(np.float16))
                self.cell_correct_flags.append(is_correct)
                self.cell_collected_idx.append(self.n_stored)
                if is_correct:
                    self._cell_n_correct += 1
                else:
                    self._cell_n_incorrect += 1

            # recursion level
            want_correct = (is_correct and self._rec_n_correct < self.rec_max_correct)
            want_incorrect = (not is_correct and self._rec_n_incorrect < self.rec_max_incorrect)
            self.is_all_trace_collected = not (want_correct or want_incorrect)

            if not self.is_all_trace_collected and rec_z_H_np is not None:
                # rec_z_H_np shape: (n_steps * H_cycles, B, L, D)
                # rec_z_L_np shape: (n_steps * L_cycles * H_cycles, B, L, D)
                rec_H = np.stack([rec_z_H_np[t * self.H_cycles : t * self.H_cycles + self.H_cycles, b, cell_start:, :] for t in range(T_actual)])  # (T_actual, H_cycles, L, D)
                rec_L = np.stack([rec_z_L_np[t * self.L_cycles * self.H_cycles : t * self.L_cycles * self.H_cycles + self.L_cycles * self.H_cycles, b, cell_start:, :] for t in range(T_actual)])  # (T_actual, L_cycles * H_cycles, L, D)

                self.rec_z_H.append(rec_H)
                self.rec_z_L.append(rec_L)
                self.rec_correct_flags.append(is_correct)
                self.rec_ratings.append(rating)

                if is_correct:
                    self._rec_n_correct += 1
                else:
                    self._rec_n_incorrect += 1

            self.trajectories.append(z_H_cell_traj)
            self.z_L_trajectories.append(z_L_cell_traj)
            # self.z_H_puzzle_emb_trajs.append(z_H_puzzle_emb_traj)
            # self.z_L_puzzle_emb_trajs.append(z_L_puzzle_emb_traj)
            self.z_H_halt_trajs.append(z_H_halt_traj)
            self.z_L_halt_trajs.append(z_L_halt_traj)
            self.z_H_ctx_trajs.append(z_H_ctx_traj)
            self.z_L_ctx_trajs.append(z_L_ctx_traj)
            self.correct_flags.append(is_correct)
            self.ratings.append(rating)
            self.given_masks.append(given)
            self.residuals.append(diffs)
            self.z_L_residuals.append(z_L_diffs)
            self.pos_residuals.append(pos_diffs)
            self.n_valid.append(n_valid)
            self.n_given.append(n_given)
            self.n_empty.append(n_empty)
            self.step_cell_correct_count.append(cell_correct_count)
            self.step_given_correct_count.append(given_correct_count)
            self.step_empty_correct_count.append(empty_correct_count)
            self.step_cell_acc.append(cell_acc)
            self.step_empty_acc.append(empty_acc)
            self.step_given_acc.append(given_acc)
            self.empty_error_step_count.append(empty_error_step_count)
            self.empty_pred_unique_count.append(empty_pred_unique_count)
            self.empty_pred_change_count.append(empty_pred_change_count)
            self.empty_flips_to_correct.append(empty_flips_to_correct)
            self.empty_flips_to_wrong.append(empty_flips_to_wrong)
            self.empty_final_correct.append(empty_final_correct)
            self.step_empty_change_rate.append(step_empty_change_rate)
            self.step_pred_stable.append(stable_step)
            self.step_first_correct.append(first_correct_step)
            self.step_preds_all.append(sample_step_preds.astype(np.int16))
            self.step_z_L_cell_acc.append(z_L_cell_acc)
            self.step_z_L_empty_acc.append(z_L_empty_acc)
            self.step_z_L_given_acc.append(z_L_given_acc)
            self.step_z_L_preds_all.append(sample_z_L_preds.astype(np.int16))
            self.step_empty_agree_correct.append(empty_agree_correct_rate)
            self.step_empty_both_wrong_same.append(empty_both_wrong_same_rate)
            self.step_empty_both_wrong_diff.append(empty_both_wrong_diff_rate)
            self.step_empty_only_z_H_correct.append(empty_only_z_H_correct_rate)
            self.step_empty_only_z_L_correct.append(empty_only_z_L_correct_rate)
            self.step_given_agree_correct.append(given_agree_correct_rate)
            self.step_given_both_wrong_same.append(given_both_wrong_same_rate)
            self.step_given_both_wrong_diff.append(given_both_wrong_diff_rate)
            self.step_given_only_z_H_correct.append(given_only_z_H_correct_rate)
            self.step_given_only_z_L_correct.append(given_only_z_L_correct_rate)
            self.step_empty_cos_sim.append(empty_cos_sim)
            self.step_given_cos_sim.append(given_cos_sim)
            self.step_z_H_empty_entropy.append(z_H_empty_entropy)
            self.step_z_H_given_entropy.append(z_H_given_entropy)
            self.step_z_L_empty_entropy.append(z_L_empty_entropy)
            self.step_z_L_given_entropy.append(z_L_given_entropy)
            self.step_output_empty_top1_prob.append(output_empty_top1_prob)
            self.step_output_empty_margin.append(output_empty_margin)
            self.step_output_given_top1_prob.append(output_given_top1_prob)
            self.step_output_given_margin.append(output_given_margin)
            self.step_output_correct_rank.append(correct_rank.astype(np.int8))   # (T_actual, 81) int8
            self.n_stored += 1


        del step_z_H_np, step_z_L_np
        del step_preds_np, step_z_L_preds_np
        del step_z_H_ent_np, step_z_L_ent_np
        del step_top2_probs_np, step_argsort_np
        if rec_z_H_np is not None:
            del rec_z_H_np, rec_z_L_np

        self.clear_batch_buffers()


    def verify_alignment(self) -> bool:
        lens = {
            "trajectories": len(self.trajectories),
            "z_L_trajectories": len(self.z_L_trajectories),
            "correct_flags": len(self.correct_flags),
            "ratings": len(self.ratings),
            "given_masks": len(self.given_masks),
            "residuals": len(self.residuals),
            "z_L_residuals": len(self.z_L_residuals),
            "pos_residuals": len(self.pos_residuals),
            "step_cell_acc": len(self.step_cell_acc),
            "step_pred_stable": len(self.step_pred_stable),
            "step_first_correct": len(self.step_first_correct),
            "step_z_L_cell_acc": len(self.step_z_L_cell_acc),
            "step_empty_agree_correct": len(self.step_empty_agree_correct),
            "step_empty_both_wrong_same": len(self.step_empty_both_wrong_same),
            "step_empty_both_wrong_diff": len(self.step_empty_both_wrong_diff),
            "step_empty_only_z_H_correct": len(self.step_empty_only_z_H_correct),
            "step_empty_only_z_L_correct": len(self.step_empty_only_z_L_correct),
            "step_given_agree_correct": len(self.step_given_agree_correct),
            "step_given_both_wrong_same": len(self.step_given_both_wrong_same),
            "step_given_both_wrong_diff": len(self.step_given_both_wrong_diff),
            "step_given_only_z_H_correct": len(self.step_given_only_z_H_correct),
            "step_given_only_z_L_correct": len(self.step_given_only_z_L_correct),
            "step_empty_cos_sim": len(self.step_empty_cos_sim),
            "step_given_cos_sim": len(self.step_given_cos_sim),
            "step_z_H_empty_entropy": len(self.step_z_H_empty_entropy),
            "step_z_H_given_entropy": len(self.step_z_H_given_entropy),
            "step_z_L_empty_entropy": len(self.step_z_L_empty_entropy),
            "step_z_L_given_entropy": len(self.step_z_L_given_entropy),
            "step_output_empty_top1_prob": len(self.step_output_empty_top1_prob),
            "step_output_empty_margin": len(self.step_output_empty_margin),
            "step_output_given_top1_prob": len(self.step_output_given_top1_prob),
            "step_output_given_margin": len(self.step_output_given_margin),
            "step_output_correct_rank": len(self.step_output_correct_rank),
        }
        ok = len(set(lens.values())) == 1
        print(f"\n[ZTrace] supervision step alignment: {'OK' if ok else 'FAILED'}")
        for k, v in lens.items():
            print(f"{k}: {v}")
        print(f"n_stored={self.n_stored} | n_skipped={self.n_skipped}")
        print(f"total seen={self.n_stored + self.n_skipped}")
        print(f"rec_correct={self._rec_n_correct} | rec_incorrect={self._rec_n_incorrect}")
        print(f"config: H_cycles={self.H_cycles} | L_cycles={self.L_cycles} | halt_max_steps={self.halt_max_steps} | snapshots_per_step={self.snapshots_per_step}")

        if self.ratings:
            r = np.array(self.ratings, dtype=float)
            f = np.array(self.correct_flags)
            r_min = int(r.min())
            n_min = int((r == r_min).sum())
            acc = float(f[r == r_min].mean())
            print(f"rating_min={r_min} | accuracy@min={acc:.3f} (n={n_min})")
