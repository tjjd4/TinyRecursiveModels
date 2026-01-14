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

        new_final_actions = carry.final_actions.clone()
        new_final_steps = carry.final_steps.clone()
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

        # Step
        new_current_step = new_current_step + 1
        
        if not self.config.no_ACT_continue:
            halt_logits = torch.stack([q_continue_logits, q_halt_logits], dim=-1)  # (N, 2)
            halt_logits = torch.nan_to_num(halt_logits, nan=0.0)
            
            halt_dist = torch.distributions.Categorical(logits=halt_logits)
        else:
            halt_dist = torch.distributions.Bernoulli(logits=q_halt_logits)

        # Training: sample for exploration; Eval: argmax for deterministic prediction
        if self.training:
            halt_action = halt_dist.sample()  # (N,) 0=Cont, 1=Halt
        else:
            halt_action = halt_logits.argmax(dim=-1)  # (N,) deterministic

        if new_current_step >= self.config.halt_max_steps:
            halt_action = torch.ones_like(halt_action)
    
        # Mask for sequences that were active before this step
        step_mask = (~carry.halted).float()
        
        halt_step_logprob = halt_dist.log_prob(halt_action)  # (N,)
        # entropy (optional)
        halt_step_entropy = halt_dist.entropy()  # (N,)

        # Accumulate log_prob and entropy for active sequences
        # step_mask ensures only accumulate for sequences that haven't halted yet (including the current step)
        new_total_logprob = carry.total_logprob + halt_step_logprob * step_mask
        new_total_entropy = carry.total_entropy + halt_step_entropy * step_mask

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            # Calculate just_halted and step_mask BEFORE updating carry.halted
            just_halted = (~carry.halted) & (halt_action == 1)
            new_halted = carry.halted | (halt_action == 1)

            # add to final actions if halted at this step
            if just_halted.any():
                new_final_actions[just_halted] = preds[just_halted]
                new_final_steps[just_halted] = new_current_step.int()

        new_carry = TinyRecursiveReasoningModel_GRPOCarry(
            inner_carry=new_inner_carry,
            current_step=new_current_step,
            halted=new_halted,
            current_data=new_current_data,
            total_logprob=new_total_logprob,
            total_entropy=new_total_entropy,
            final_actions=new_final_actions,
            final_steps=new_final_steps,
        )

        return new_carry, outputs
