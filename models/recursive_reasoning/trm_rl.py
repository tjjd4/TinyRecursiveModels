from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn

from .trm import (
    TinyRecursiveReasoningModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1Carry,
    TinyRecursiveReasoningModel_ACTV1InnerCarry
)

@dataclass
class TinyRecursiveReasoningModel_GRPOCarry:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry
    current_step: torch.Tensor
    halted: torch.Tensor

    total_logprob: torch.Tensor    # (N,)
    total_entropy: torch.Tensor    # (N,)
    final_actions: torch.Tensor    # (N, L)
    final_steps: torch.Tensor      # (N,)

    current_data: Dict[str, torch.Tensor]


class TinyRecursiveReasoningModel_RL(TinyRecursiveReasoningModel_ACTV1):
    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> TinyRecursiveReasoningModel_GRPOCarry:
        device = batch["inputs"].device
        batch_size = batch["inputs"].shape[0]

        return TinyRecursiveReasoningModel_GRPOCarry(
            inner_carry=self.inner.empty_carry(batch_size),
            
            current_step=torch.tensor(0, device=device, dtype=torch.int),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),

            total_logprob=torch.zeros(batch_size, device=device),
            total_entropy=torch.zeros(batch_size, device=device),
            final_actions=torch.zeros_like(batch["inputs"], dtype=torch.long),
            final_steps=torch.zeros(batch_size, device=device),

            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(
        self,
        carry: TinyRecursiveReasoningModel_GRPOCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[TinyRecursiveReasoningModel_GRPOCarry, Dict[str, torch.Tensor]]:

        if carry.halted.all():
            with torch.device("cuda"):
                carry = self.initial_carry(batch)

        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_current_step = torch.where(carry.halted.all(), 0, carry.current_step)
        """
        for streaming training
        """
        # new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}
        
        """
        batch training (no streaming)
        """
        new_current_data = {k: v.clone() for k, v in batch.items()}

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,  # (B, L, V)
            "q_halt_logits": q_halt_logits,        # (B,)
            "q_continue_logits": q_continue_logits # (B,)
        }

        with torch.no_grad():
            # Step
            new_current_step = new_current_step + 1
            is_last_step = new_current_step >= self.config.halt_max_steps

            # ONLY check if is last step, give rl loss head to sample halt action
            halted = is_last_step

            # if self.training and self.config.halt_max_steps > 1:
            #     halted = halted | (q_halt_logits > q_continue_logits)

            #     # Exploration
            #     min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
            #     halted = halted & (new_steps >= min_halt_steps)

        new_carry = TinyRecursiveReasoningModel_GRPOCarry(
            inner_carry=new_inner_carry,
            current_step=new_current_step,
            halted=halted,
            current_data=new_current_data,
            total_logprob=carry.total_logprob,
            total_entropy=carry.total_entropy,
            final_actions=carry.final_actions,
            final_steps=carry.final_steps,
        )

        return new_carry, outputs
