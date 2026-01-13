from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
import math
from models.losses.loss_fn import IGNORE_LABEL_ID

class GRPOLossHead(nn.Module):
    """
      - outputs_dict["logits"]: (B, L, V) non-autoregressive grid logits
      - outputs_dict["q_halt_logits"], outputs_dict["q_continue_logits"]: (B,) 供 halting action (0=continue,1=halt)
    """

    def __init__(
        self,
        model: nn.Module,
        num_generations: int,
        len_penalty: float,
        correct_reward: float,
        entropy_bonus: float,
    ):
        super().__init__()
        self.model = model

        self.num_generations = int(num_generations)
        self.len_penalty = float(len_penalty)
        self.correct_reward = float(correct_reward)
        self.entropy_bonus = float(entropy_bonus)

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    # expand batch before forwarding
    @staticmethod
    def expand_batch(batch: Dict[str, torch.Tensor], num_generations: int) -> Dict[str, torch.Tensor]:
        # (B, ...) -> (B*G, ...)
        return {k: v.repeat_interleave(num_generations, dim=0) for k, v in batch.items()}

    @staticmethod
    def _seq_exact_correct(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        pred:   (N, L) int
        labels: (N, L) int with IGNORE_LABEL_ID
        return: (N,) bool exact match on non-ignored positions
        """
        mask = labels != IGNORE_LABEL_ID
        valid_counts = mask.sum(dim=-1)
        correct_tokens = (pred == labels) & mask
        is_exact = torch.where(valid_counts > 0, correct_tokens.sum(dim=-1) == valid_counts, torch.zeros_like(valid_counts, dtype=torch.bool))
        return is_exact

    def forward(
        self,
        return_keys: Sequence[str],
        *,
        carry: Any,
        batch: Dict[str, torch.Tensor],
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        device = batch["inputs"].device
        N = batch["inputs"].shape[0]  # expanded batch size
        G = self.num_generations

        if N % G != 0:
            raise ValueError(f"Batch size ({N}) must be divisible by num_generations ({G})")

        B = N // G  # true batch size (before expansion)

        new_carry, outputs = self.model(carry=carry, batch=batch, **model_kwargs)
        labels = new_carry.current_data["labels"]

        halt_logits = torch.stack([outputs["q_continue_logits"], outputs["q_halt_logits"]], dim=-1)  # (N, 2)
        halt_logits = torch.nan_to_num(halt_logits, nan=0.0)

        halt_dist = torch.distributions.Categorical(logits=halt_logits)
        halt_action = halt_dist.sample()  # (N,) 0=Cont, 1=Halt

        # Mask for sequences that were active before this step
        step_mask = (~new_carry.halted).float()
        
        halt_step_logprob = halt_dist.log_prob(halt_action)  # (N,)

        # entropy (optional)
        halt_step_entropy = torch.zeros_like(halt_step_logprob)
        if self.entropy_bonus != 0.0:
            halt_step_entropy = halt_dist.entropy()  # (N,)

        # Accumulate log_prob and entropy for active sequences
        # step_mask ensures only accumulate for sequences that haven't halted yet (including the current step)
        new_carry.total_logprob = new_carry.total_logprob + halt_step_logprob * step_mask
        new_carry.total_entropy = new_carry.total_entropy + halt_step_entropy * step_mask

        with torch.no_grad():
            # Preds
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            # Calculate just_halted and step_mask BEFORE updating new_carry.halted
            just_halted = (~new_carry.halted) & (halt_action == 1)
            
            new_carry.halted = new_carry.halted | (halt_action == 1)

            # add to final actions if halted at this step
            if just_halted.any():
                new_carry.final_actions[just_halted] = outputs["preds"][just_halted]
                new_carry.final_steps[just_halted] = new_carry.current_step.int()

            # Correctness
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)

            is_correct = mask & (new_carry.final_actions == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts

            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                
                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.current_step.int(), 0).sum(),
            }

            # if not finished yet
            if not new_carry.halted.all():
                return new_carry, None, metrics, None, new_carry.halted.all()
                
            r_correct = seq_is_correct * self.correct_reward

            # length penalty (normalize to [0,1])
            r_len = torch.where(seq_is_correct, -self.len_penalty * (new_carry.final_steps / max(1, self.model.config.halt_max_steps)), torch.zeros_like(seq_is_correct))

            rewards = r_correct + r_len

            # group baseline (GRPO)
            rewards_grouped = rewards.view(B, G)
            baseline = rewards_grouped.mean(dim=1, keepdim=True)
            std = rewards_grouped.std(dim=1, keepdim=True) + 1e-8
            adv = (rewards_grouped - baseline) / std
            adv = adv.view(-1)

        # policy gradient loss
        pg_loss: torch.Tensor = -(adv * new_carry.total_logprob).sum()
        
        ent_loss: torch.Tensor = torch.tensor(0.0, device=device)
        # optional entropy bonus (maximize entropy => subtract negative)
        if self.entropy_bonus != 0.0:
            ent_loss: torch.Tensor = -self.entropy_bonus * (new_carry.total_entropy / new_carry.current_step.float().clamp_min(1)).sum()
        
        grpo_loss: torch.Tensor = (pg_loss + ent_loss) / float(G)

        metrics.update({
            "grpo_loss": grpo_loss.detach(),
            "pg_loss": pg_loss.detach(),
            "ent_loss": ent_loss.detach(),
            "reward": rewards.sum(),
        })

        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, grpo_loss, metrics, detached_outputs, new_carry.halted.all()