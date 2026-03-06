from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .trm import (
    TinyRecursiveReasoningModel_ACTV1InnerCarry,
    TinyRecursiveReasoningModel_ACTV1Config,
    TinyRecursiveReasoningModel_ACTV1_Inner,
)

@dataclass
class TinyRecursiveReasoningModel_GRPOCarry:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry
    steps: torch.Tensor
    halted: torch.Tensor

    total_halt_logprob: torch.Tensor    # (N,)
    total_token_logprob: torch.Tensor    # (N,)
    total_halt_entropy: torch.Tensor    # (N,)
    total_token_entropy: torch.Tensor    # (N,)
    final_actions: torch.Tensor    # (N, L)
    final_halt_actions: torch.Tensor    # (N, L)

    total_halt_kl: torch.Tensor
    total_token_kl: torch.Tensor

    current_data: Dict[str, torch.Tensor]

class TinyRecursiveReasoningModel_GRPOConfig(TinyRecursiveReasoningModel_ACTV1Config):
    temperature: float = 1.0    # 1.0 = no scaling, <1 more greedy, >1 more random
    top_p: float = 1.0          # 1.0 = no nucleus sampling, <1 enable top-p filtering


class TinyRecursiveReasoningModel_GRPO(nn.Module):
    """GRPO wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModel_GRPOConfig(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb


    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> TinyRecursiveReasoningModel_GRPOCarry:
        device = batch["inputs"].device
        batch_size = batch["inputs"].shape[0]

        return TinyRecursiveReasoningModel_GRPOCarry(
            inner_carry=self.inner.empty_carry(batch_size),
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32, device=device),
            halted=torch.ones(batch_size, dtype=torch.bool, device=device),

            total_halt_logprob=torch.zeros(batch_size, device=device),
            total_token_logprob=torch.zeros(batch_size, device=device),
            total_halt_entropy=torch.zeros(batch_size, device=device),
            total_token_entropy=torch.zeros(batch_size, device=device),
            final_actions=torch.zeros_like(batch["inputs"], dtype=torch.long),
            final_halt_actions=torch.zeros(batch_size, dtype=torch.long, device=device),

            total_halt_kl=torch.zeros(batch_size, device=device),
            total_token_kl=torch.zeros(batch_size, device=device),

            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_GRPOCarry):
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(reset_flag, carry.inner_carry)
        return TinyRecursiveReasoningModel_GRPOCarry(
            inner_carry=new_inner_carry,
            steps=torch.zeros_like(carry.steps),
            halted=torch.zeros_like(carry.halted),
            total_halt_logprob=torch.zeros_like(carry.total_halt_logprob),
            total_token_logprob=torch.zeros_like(carry.total_token_logprob),
            total_halt_entropy=torch.zeros_like(carry.total_halt_entropy),
            total_token_entropy=torch.zeros_like(carry.total_token_entropy),
            final_actions=torch.zeros_like(carry.final_actions),
            final_halt_actions=torch.zeros_like(carry.final_halt_actions),
            total_halt_kl=torch.zeros_like(carry.total_halt_kl),
            total_token_kl=torch.zeros_like(carry.total_token_kl),
            current_data={k: torch.empty_like(v) for k, v in carry.current_data.items()},
        )

    def forward(
        self,
        carry: TinyRecursiveReasoningModel_GRPOCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[TinyRecursiveReasoningModel_GRPOCarry, Dict[str, torch.Tensor]]:

        if carry.halted.all():
            with torch.device("cuda"):
                carry = self.reset_carry(carry.halted, carry)

        new_steps = torch.where(carry.halted.all(), 0, carry.steps)

        new_final_actions = carry.final_actions.clone()
        new_final_halt_actions = carry.final_halt_actions.clone()
        """
        for streaming training
        """
        # new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}
        
        """
        batch training (no streaming)
        """
        new_current_data = {k: v.clone() for k, v in batch.items()}

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(carry.inner_carry, new_current_data)


        outputs = {
            "logits": logits,  # (B, L, V)
            "q_halt_logits": q_halt_logits,        # (B,)
            "q_continue_logits": q_continue_logits # (B,)
        }

        """
        if no_ACT_continue == false, q_continue_logits weight = 0 => NO usage of q_continue_logits
        """
        if not self.config.no_ACT_continue:
            halt_logits = torch.stack([q_continue_logits, q_halt_logits], dim=-1)  # (N, 2)
            halt_logits = torch.nan_to_num(halt_logits, nan=0.0)

            halt_dist = torch.distributions.Categorical(logits=halt_logits)
        else:
            halt_logits = q_halt_logits
            halt_dist = torch.distributions.Bernoulli(logits=halt_logits)

        with torch.no_grad():
            # Step
            new_steps = torch.where(carry.halted, new_steps, new_steps + 1)

            # Training: sample for exploration; Eval: argmax for deterministic prediction
            if self.training:
                halt_action = halt_dist.sample()  # (N,) 0=Cont, 1=Halt
                scaled_logits = logits / self.config.temperature
                sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
                probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(probs, dim=-1)

                sorted_mask = (cumulative_probs - probs) >= self.config.top_p
                sorted_mask[..., 0] = False  # Ensure at least the top-1 token is never masked out
                sorted_logits[sorted_mask] = float('-inf')

                scaled_logits = sorted_logits.scatter(-1, sorted_indices, sorted_logits)

                token_action = torch.distributions.Categorical(logits=scaled_logits).sample()
            else:
                if not self.config.no_ACT_continue:
                    halt_action = halt_logits.argmax(dim=-1)  # (N,) deterministic
                else:
                    halt_action = (halt_logits > 0).long()
                token_action = logits.argmax(dim=-1)  # (N,) deterministic
            
            is_last_step = new_steps >= self.config.halt_max_steps

            halt_action = torch.where(is_last_step, torch.ones_like(halt_action), halt_action)

            new_halted = carry.halted | (halt_action == 1)
            # Mask for sequences that were just halted in this step
            just_halted = new_halted & (~carry.halted)

            just_halted_mask = just_halted.unsqueeze(-1)
            
            new_final_actions = torch.where(just_halted_mask, token_action, carry.final_actions)
            new_final_halt_actions = torch.where(just_halted, (q_halt_logits >= 0).long(), carry.final_halt_actions)

            outputs.update({
                "sampled_halt_action": halt_action,
                "sampled_token_action": token_action,
            })

        new_carry = TinyRecursiveReasoningModel_GRPOCarry(
            inner_carry=new_inner_carry,
            steps=new_steps,
            halted=new_halted,
            current_data=new_current_data,
            total_halt_logprob=carry.total_halt_logprob,
            total_token_logprob=carry.total_token_logprob,
            total_halt_entropy=carry.total_halt_entropy,
            total_token_entropy=carry.total_token_entropy,
            final_actions=new_final_actions,
            final_halt_actions=new_final_halt_actions,
            total_halt_kl=carry.total_halt_kl,
            total_token_kl=carry.total_token_kl,
        )

        return new_carry, outputs
