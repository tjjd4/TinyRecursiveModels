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
        max_steps: int,
        len_penalty: float,
        correct_reward: float,
        entropy_bonus: float,
    ):
        super().__init__()
        self.model = model

        self.num_generations = int(num_generations)
        self.max_steps = int(max_steps)

        self.len_penalty = float(len_penalty)
        self.correct_reward = float(correct_reward)
        self.entropy_bonus = float(entropy_bonus)

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    @staticmethod
    def _expand_batch(batch: Dict[str, torch.Tensor], num_generations: int) -> Dict[str, torch.Tensor]:
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
        batch: Dict[str, torch.Tensor],
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        device = batch["inputs"].device
        B = batch["inputs"].shape[0]
        G = self.num_generations
        N = B * G

        # expand batch for group sampling
        expanded_batch = self._expand_batch(batch, G)

        with torch.device("cuda"):
            carry = self.model.initial_carry(expanded_batch)

        # rollout state
        active = torch.ones(N, dtype=torch.bool, device=device)

        final_actions = torch.zeros_like(expanded_batch["inputs"], dtype=torch.long)
        final_steps = torch.zeros(N, dtype=torch.float32, device=device)

        total_logprob = torch.zeros(N, dtype=torch.float32, device=device)
        total_entropy = torch.zeros(N, dtype=torch.float32, device=device)

        # rollout loop
        for step in range(self.max_steps):
            if not active.any():
                break
                
            carry, outputs = self.model(carry=carry, batch=expanded_batch, **model_kwargs)

            logits = outputs["logits"]  # (N, L, V)
            q_halt_logits = outputs["q_halt_logits"]  # (N,)
            q_continue_logits = outputs["q_continue_logits"]  # (N,)

            grid_action = torch.argmax(logits, dim=-1)  # (N, L)

            halt_logits = torch.stack([q_continue_logits, q_halt_logits], dim=-1)  # (N, 2)
            halt_logits = torch.nan_to_num(halt_logits, nan=0.0)

            halt_dist = torch.distributions.Categorical(logits=halt_logits)
            halt_action = halt_dist.sample()  # (N,) 0=Cont, 1=Halt
            
            halt_step_logprob = halt_dist.log_prob(halt_action)  # (N,)

            # entropy (optional)
            if self.entropy_bonus != 0.0:
                halt_step_entropy = halt_dist.entropy()  # (N,)
            else:
                halt_step_entropy = torch.zeros_like(halt_step_logprob)

            step_mask = active.float()

            total_logprob = total_logprob + halt_step_logprob * step_mask
            total_entropy = total_entropy + halt_step_entropy * step_mask

            just_halted = active & (halt_action == 1)

            if just_halted.any():
                final_actions[just_halted] = grid_action[just_halted]
                final_steps[just_halted] = float(step + 1)

            active = active & (halt_action == 0)
        
        if active.any():
            if grid_action is None:
                carry, outputs = self.model(carry=carry, batch=expanded_batch, **model_kwargs)
                logits = outputs["logits"]
                grid_action = torch.argmax(logits, dim=-1)

        final_actions[active] = grid_action[active]
        final_steps[active] = float(self.max_steps)
            
        labels = expanded_batch["labels"]
        pred = final_actions

        exact = self._seq_exact_correct(pred, labels)
        r_correct = exact.float() * self.correct_reward

        # length penalty (normalize to [0,1])
        r_len = -self.len_penalty * (final_steps / max(1, self.max_steps))

        rewards = r_correct + r_len

        # group baseline (GRPO)
        rewards_grouped = rewards.view(B, G)
        baseline = rewards_grouped.mean(dim=1, keepdim=True)
        std = rewards_grouped.std(dim=1, keepdim=True) + 1e-8
        adv = (rewards_grouped - baseline) / std
        adv = adv.view(-1)

        # policy gradient loss
        pg_loss: torch.Tensor = -(adv.detach() * total_logprob).sum()
        
        ent_loss: torch.Tensor = torch.tensor(0.0, device=device)
        # optional entropy bonus (maximize entropy => subtract negative)
        if self.entropy_bonus != 0.0:
            ent_loss = -self.entropy_bonus * (total_entropy / max(1, self.max_steps)).sum()
        
        loss = (pg_loss + ent_loss) / float(G)

        # metrics (no grad)
        with torch.no_grad():
            metrics = {
                "count": torch.tensor(N, device=device, dtype=torch.float32),
                "loss": loss.detach(),
                "pg_loss": pg_loss.detach(),
                "exact_accuracy": exact.float().mean(),
                "avg_steps": final_steps.mean(),
                "avg_reward": rewards.mean(),
            }

        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}
        # rollout -> no carry
        carry = None
        all_finish = torch.tensor(True, device=device)

        return carry, loss, metrics, detached_outputs, all_finish