from typing import Any, Tuple, Dict, Sequence, Optional

import pydantic
import torch
import torch.nn.functional as F
from torch import nn
import math
from models.losses.loss_fn import IGNORE_LABEL_ID
from utils.functions import load_model_class

class RewardConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str

class GRPOLossConfig(pydantic.BaseModel):
    num_generations: int
    entropy_bonus: float
    reward: RewardConfig

class GRPOLossHead(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        config_dict: dict,
    ):
        super().__init__()
        self.model = model
        self.config = GRPOLossConfig(**config_dict)

        reward_cfg = dict(
            **self.config.reward.__pydantic_extra__,
        )

        reward_cls = load_model_class(self.config.reward.name)
        self.reward_fn = reward_cls(reward_cfg)

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
        G = self.config.num_generations

        if N % G != 0:
            raise ValueError(f"Batch size ({N}) must be divisible by num_generations ({G})")

        B = N // G  # true batch size (before expansion)

        new_carry, outputs = self.model(carry=carry, batch=batch, **model_kwargs)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
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
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

            # if not finished yet
            if not new_carry.halted.all():
                return new_carry, None, metrics, None, new_carry.halted.all()

            rewards = self.reward_fn.compute(
                seq_is_correct=seq_is_correct,
                final_steps=new_carry.final_steps,
                max_steps=self.model.config.halt_max_steps
            )

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
        if self.config.entropy_bonus != 0.0:
            ent_loss: torch.Tensor = -self.config.entropy_bonus * (new_carry.total_entropy / new_carry.final_steps.clamp_min(1)).mean()
        
        grpo_loss: torch.Tensor = (pg_loss + ent_loss) / float(G)

        metrics.update({
            "grpo_loss": grpo_loss.detach(),
            "pg_loss": pg_loss.detach(),
            "ent_loss": ent_loss.detach(),
            "reward": rewards.sum(),
        })

        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, grpo_loss, metrics, detached_outputs, new_carry.halted.all()