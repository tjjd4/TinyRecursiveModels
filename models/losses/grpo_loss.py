from typing import Any, Tuple, Dict, Sequence, Optional
import copy

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
    kl_beta: float
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

        # ref_model will be initialized after checkpoint loading via init_ref_model()
        self.ref_model = None

    def init_ref_model(self):
        if hasattr(self.config, "kl_beta") and self.config.kl_beta > 0.0:
            self.ref_model = copy.deepcopy(self.model)
            for p in self.ref_model.parameters():
                p.requires_grad = False
            self.ref_model.eval()
            print(" -> ref_model created (frozen copy of loaded model).")

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

        new_carry, outputs = self.model(carry=carry, batch=batch, **model_kwargs)
        labels = new_carry.current_data["labels"]
        mask = (labels != IGNORE_LABEL_ID)

        # Compute logprob and entropy then accumulate into carry (only training)
        if self.training:
            sampled_halt_action = outputs["sampled_halt_action"]
            sampled_token_action = outputs["sampled_token_action"]

            # Rebuild distributions from logits
            if not self.model.config.no_ACT_continue:
                halt_logits = torch.stack([outputs["q_continue_logits"], outputs["q_halt_logits"]], dim=-1)
                halt_logits = torch.nan_to_num(halt_logits, nan=0.0)
                halt_dist = torch.distributions.Categorical(logits=halt_logits)
            else:
                halt_dist = torch.distributions.Bernoulli(logits=outputs["q_halt_logits"])

            token_dist = torch.distributions.Categorical(logits=outputs["logits"])

            # Halt log_prob / entropy
            step_halt_logprob = halt_dist.log_prob(sampled_halt_action.float())  # (N,)
            step_halt_entropy = halt_dist.entropy()  # (N,)

            # Token log_prob / entropy (only valid positions)
            step_token_logprob = (token_dist.log_prob(sampled_token_action.long()) * mask).sum(dim=-1)  # (N,)
            step_token_entropy = (token_dist.entropy() * mask).sum(dim=-1)  # (N,)

            # Masks for accumulation
            active_mask_float = (~carry.halted).float()
            just_halted_mask_float = (new_carry.halted & ~carry.halted).float()

            # Accumulate logprob and entropy into carry
            new_carry.total_logprob = new_carry.total_logprob + step_halt_logprob * active_mask_float + step_token_logprob * just_halted_mask_float
            new_carry.total_entropy = new_carry.total_entropy + step_halt_entropy * active_mask_float + step_token_entropy * just_halted_mask_float

            # KL divergence against ref_model
            if self.ref_model is not None:
                with torch.no_grad():
                    num_tokens = mask.sum(-1).clamp_min(1)

                    _, ref_logits, (ref_q_halt, ref_q_cont) = self.ref_model.inner(carry.inner_carry, batch, **model_kwargs)

                    if not self.model.config.no_ACT_continue:
                        ref_halt_logits = torch.stack([ref_q_cont, ref_q_halt], dim=-1)
                        ref_halt_logits = torch.nan_to_num(ref_halt_logits, nan=0.0)
                        ref_halt_dist = torch.distributions.Categorical(logits=ref_halt_logits)
                    else:
                        ref_halt_logits = ref_q_halt
                        ref_halt_dist = torch.distributions.Bernoulli(logits=ref_halt_logits)

                    ref_token_dist = torch.distributions.Categorical(logits=ref_logits)

                    ref_halt_logprob = ref_halt_dist.log_prob(sampled_halt_action.float())  # (N,)
                    ref_token_logprob = (ref_token_dist.log_prob(sampled_token_action.long()) * mask).sum(dim=-1)  # (N,)

                step_halt_kl = step_halt_logprob - ref_halt_logprob
                # divide by num_tokens to normalize
                step_token_kl = (step_token_logprob - ref_token_logprob) / num_tokens

                step_total_kl = step_halt_kl * active_mask_float + step_token_kl * just_halted_mask_float

                # Accumulate KL into carry
                new_carry.total_kl = new_carry.total_kl + step_total_kl

        with torch.no_grad():
            # Correctness
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
            if not new_carry.halted.all() or not self.training:
                detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}
                return new_carry, None, metrics, detached_outputs, new_carry.halted.all()

            device = batch["inputs"].device
            N = batch["inputs"].shape[0]  # expanded batch size
            G = self.config.num_generations

            if N % G != 0:
                raise ValueError(f"Batch size ({N}) must be divisible by num_generations ({G})")

            B = N // G  # true batch size (before expansion)

            rewards = self.reward_fn.compute(
                seq_is_correct=seq_is_correct,
                final_steps=new_carry.steps,
                max_steps=self.model.config.halt_max_steps
            )

            # group baseline (GRPO)
            rewards_grouped = rewards.view(B, G)
            baseline = rewards_grouped.mean(dim=1, keepdim=True)
            std = rewards_grouped.std(dim=1, keepdim=True) + 1e-8
            adv = (rewards_grouped - baseline) / std
            adv = adv.view(-1)

            std_g = rewards_grouped.std(dim=1, unbiased=False)
            any_pos_g = (rewards_grouped > 0).any(dim=1).float() # (B,)
            zero_std_g = (std_g < 1e-8).float()                  # (B,)

            # 變成 sample-level，讓外部 /count 後仍是 ratio
            any_pos_s = any_pos_g.repeat_interleave(G)           # (N,)
            zero_std_s = zero_std_g.repeat_interleave(G)         # (N,)
            std_s = std_g.repeat_interleave(G)                   # (N,)

            # 只統計 valid_metrics 的那些 sample（跟你現有 count 對齊）
            vm = valid_metrics  # shape (N,)
            vm_f = vm.float()

        # policy gradient loss
        pg_loss: torch.Tensor = -(adv * new_carry.total_logprob).sum()
        
        ent_loss: torch.Tensor = torch.tensor(0.0, device=device)
        # optional entropy bonus (maximize entropy => subtract negative)
        if self.config.entropy_bonus != 0.0:
            ent_loss: torch.Tensor = -self.config.entropy_bonus * (new_carry.total_entropy / new_carry.steps.clamp_min(1)).sum()

        # KL divergence loss: sum across all recursive steps
        kl_loss: torch.Tensor = self.config.kl_beta * new_carry.total_kl.sum()

        grpo_loss: torch.Tensor = (pg_loss + ent_loss + kl_loss) / float(G)

        metrics.update({
            "grpo_loss": grpo_loss.detach(),
            "pg_loss": pg_loss.detach(),
            "ent_loss": ent_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "reward": rewards.sum(),
            "groups_with_any_correct": (any_pos_s * vm_f).sum(),
            "zero_std_groups": (zero_std_s * vm_f).sum(),
            "mean_group_std": (std_s * vm_f).sum(),
            "adv_mean_abs": (adv.abs() * vm_f).sum(),
        })

        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, grpo_loss, metrics, detached_outputs, new_carry.halted.all()