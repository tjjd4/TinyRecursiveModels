import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch._dynamo


@dataclass
class ZTrace:
    H_cycles: int
    L_cycles: int
    puzzle_emb_len: int
    halt_max_steps: int
    rec_max_correct: int
    rec_max_incorrect: int

    _step_snapshots_H: List[torch.Tensor] = field(default_factory=list)
    _step_snapshots_L: List[torch.Tensor] = field(default_factory=list)

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

    def __init__(self, H_cycles: int, L_cycles: int, puzzle_emb_len: int, halt_max_steps: int, rec_max_correct: int, rec_max_incorrect: int):
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.puzzle_emb_len = puzzle_emb_len
        self.halt_max_steps = halt_max_steps
        self.rec_max_correct = rec_max_correct
        self.rec_max_incorrect = rec_max_incorrect

        # record
        self._rec_z_H = []
        self._rec_z_L = []
        self._step_z_H = []
        self._step_z_L = []
        self._step_preds = []
        self._step_z_L_preds = []

        # step level trajectories
        self.trajectories = []
        self.z_L_trajectories = []
        self.correct_flags = []
        self.ratings = []
        self.given_masks = []
        self.residuals = []
        self.z_L_residuals = []
        self.pos_residuals = []

        # recursion level z_H and z_L
        self.rec_z_H = []
        self.rec_z_L = []
        self.rec_correct_flags = []
        self.rec_ratings = []

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

        self._rec_n_correct = 0
        self._rec_n_incorrect = 0
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


    def clear_batch_buffers(self) -> None:
        """Call before each batch."""
        self._rec_z_H.clear()
        self._rec_z_L.clear()
        self._step_z_H.clear()
        self._step_z_L.clear()
        self._step_preds.clear()
        self._step_z_L_preds.clear()
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

        step_z_H_np = torch.stack(self._step_z_H).cpu().float().numpy()  # (n_steps, B, L, D)
        step_z_L_np = torch.stack(self._step_z_L).cpu().float().numpy()  # (n_steps, B, L, D)
        step_preds_np = torch.stack(self._step_preds).cpu().numpy()  # (n_steps, B, seq_len, n_classes)
        step_z_L_preds_np = torch.stack(self._step_z_L_preds).cpu().numpy()  # (n_steps, B, seq_len, n_classes)

        # rec_z_H/L are called snapshots_per_step times per supervision step
        if self._rec_z_H:
            rec_z_H_np = torch.stack(self._rec_z_H).cpu().float().numpy()  # (n_steps * S, B, L, D)
            rec_z_L_np = torch.stack(self._rec_z_L).cpu().float().numpy()  # (n_steps * S, B, L, D)
        else:
            rec_z_H_np = None
            rec_z_L_np = None

        self._step_z_H.clear()
        self._step_z_L.clear()
        self._step_preds.clear()
        self._rec_z_H.clear()
        self._rec_z_L.clear()

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

            cell_start = self.puzzle_emb_len  # e.g. 2
            cell_end = self.puzzle_emb_len + 81

            z_H_cell = z_H_last[:, cell_start:cell_end, :]   # (T_actual, 81, D)
            z_L_cell = z_L_last[:, cell_start:cell_end, :]   # (T_actual, 81, D)

            # mean-pool over sequence positions → (T_actual, D)
            z_H_traj = z_H_cell.mean(axis=1)
            z_L_traj = z_L_cell.mean(axis=1)

            # given mask from first 81 cell tokens
            given = batch_inputs_np[b, :] != 1  # (81,)

            # step-wise residuals
            if T_actual > 1:
                diffs = np.linalg.norm(np.diff(z_H_traj, axis=0), axis=-1)       # (T-1,)
                z_L_diffs = np.linalg.norm(np.diff(z_L_traj, axis=0), axis=-1)       # (T-1,)
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

            cell_acc = per_step_correct[:, cell_mask].sum(axis=1).astype(np.float32) / max(n_valid, 1)
            given_acc = per_step_correct[:, given & cell_mask].sum(axis=1).astype(np.float32) / max(n_given, 1)
            empty_acc = per_step_correct[:, empty].sum(axis=1).astype(np.float32) / max(n_empty, 1)

            # prediction stability: start from the last step and go backwards, the earliest step that makes the prediction no longer change
            stable_step = T_actual
            final_pred = sample_step_preds[-1]
            for k in range(T_actual - 2, -1, -1):
                if np.array_equal(sample_step_preds[k], final_pred):
                    stable_step = k+1
                else:
                    break
            
            # z_L logit lens
            sample_z_L_preds = step_z_L_preds_np[:T_actual, b, :]  # (T_actual, 81)

            z_L_per_step_correct = (sample_z_L_preds == cell_labels[None, :]) & cell_mask[None, :]

            z_L_cell_acc = z_L_per_step_correct[:, cell_mask].sum(axis=1).astype(np.float32) / max(n_valid, 1)
            z_L_given_acc = z_L_per_step_correct[:, given & cell_mask].sum(axis=1).astype(np.float32) / max(n_given, 1)
            z_L_empty_acc = z_L_per_step_correct[:, empty].sum(axis=1).astype(np.float32) / max(n_empty, 1)

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
            given_valid = given & cell_mask
            n_g = max(n_given, 1)
            given_agree_correct_rate = both_correct[:, given_valid].sum(axis=1).astype(np.float32) / n_g
            given_both_wrong_same_rate = both_wrong_same[:, given_valid].sum(axis=1).astype(np.float32) / n_g
            given_both_wrong_diff_rate = both_wrong_diff[:, given_valid].sum(axis=1).astype(np.float32) / n_g
            given_only_z_H_correct_rate = only_z_H_correct[:, given_valid].sum(axis=1).astype(np.float32) / n_g
            given_only_z_L_correct_rate = only_z_L_correct[:, given_valid].sum(axis=1).astype(np.float32) / n_g

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
                    for c in range(81) if given[c]
                ]) if given.any() else np.nan
                for t in range(T_actual)
            ])

            self.trajectories.append(z_H_traj)
            self.z_L_trajectories.append(z_L_traj)
            self.correct_flags.append(is_correct)
            self.ratings.append(rating)
            self.given_masks.append(given)
            self.residuals.append(diffs)
            self.z_L_residuals.append(z_L_diffs)
            self.pos_residuals.append(pos_diffs)
            self.step_cell_acc.append(cell_acc)
            self.step_empty_acc.append(empty_acc)
            self.step_given_acc.append(given_acc)
            self.step_pred_stable.append(stable_step)
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
            self.n_stored += 1

            # recursion level
            want_correct = (is_correct and self._rec_n_correct < self.rec_max_correct)
            want_incorrect = (not is_correct and self._rec_n_incorrect < self.rec_max_incorrect)
            self.is_all_trace_collected = not (want_correct or want_incorrect)

            if not self.is_all_trace_collected and rec_z_H_np is not None:
                # rec_z_H_np shape: (n_steps * H_cycles, B, L, D)
                # rec_z_L_np shape: (n_steps * L_cycles * H_cycles, B, L, D)
                rec_H = np.stack([rec_z_H_np[t * self.H_cycles : t * self.H_cycles + self.H_cycles, b, cell_start:cell_end, :] for t in range(T_actual)])  # (T_actual, H_cycles, L, D)
                rec_L = np.stack([rec_z_L_np[t * self.L_cycles * self.H_cycles : t * self.L_cycles * self.H_cycles + self.L_cycles * self.H_cycles, b, cell_start:cell_end, :] for t in range(T_actual)])  # (T_actual, L_cycles * H_cycles, L, D)

                self.rec_z_H.append(rec_H)
                self.rec_z_L.append(rec_L)
                self.rec_correct_flags.append(is_correct)
                self.rec_ratings.append(rating)

                if is_correct:
                    self._rec_n_correct += 1
                else:
                    self._rec_n_incorrect += 1

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
        return ok