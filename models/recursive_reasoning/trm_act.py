from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
from torch import nn

from .trm import (
    TinyRecursiveReasoningModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1Carry,
    TinyRecursiveReasoningModel_ACTV1InnerCarry
)

@dataclass
class TinyRecursiveReasoningModel_ACTV2Carry:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]

    final_actions: Optional[torch.Tensor]

class TinyRecursiveReasoningModel_ACTV2(TinyRecursiveReasoningModel_ACTV1):
    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> TinyRecursiveReasoningModel_ACTV2Carry:
        device = batch["inputs"].device
        batch_size = batch["inputs"].shape[0]

        return TinyRecursiveReasoningModel_ACTV2Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32, device=device),
            halted=torch.ones(batch_size, dtype=torch.bool, device=device),

            current_data={k: torch.empty_like(v) for k, v in batch.items()},

            final_actions=torch.zeros_like(batch["inputs"], dtype=torch.long),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV2Carry):
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(reset_flag, carry.inner_carry)
        return TinyRecursiveReasoningModel_ACTV2Carry(
            inner_carry=new_inner_carry,
            steps=torch.zeros_like(carry.steps),
            halted=torch.zeros_like(carry.halted),
            current_data={k: torch.empty_like(v) for k, v in carry.current_data.items()},

            final_actions=torch.zeros_like(carry.final_actions),
        )

    def forward(
        self,
        carry: TinyRecursiveReasoningModel_ACTV2Carry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[TinyRecursiveReasoningModel_ACTV2Carry, Dict[str, torch.Tensor]]:

        # Update data, carry (removing halted sequences)
        if carry.halted.all():
            with torch.device("cuda"):
                carry = self.reset_carry(carry.halted, carry)

        new_steps = torch.where(carry.halted.all(), 0, carry.steps)

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
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        with torch.no_grad():
            # Step
            new_steps = torch.where(carry.halted, new_steps, new_steps + 1)
            is_last_step = new_steps >= self.config.halt_max_steps

            halted = is_last_step

            if self.config.no_ACT_continue:
                halted = halted | (q_halt_logits > 0)
            else:
                halted = halted | (q_halt_logits > q_continue_logits)

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    # Compute target Q
                    # NOTE: No replay buffer and target networks for computing target Q-value.
                    # As batch_size is large, there're many parallel envs.
                    # Similar concept as PQN https://arxiv.org/abs/2407.04811
                    _, _, (next_q_halt_logits, next_q_continue_logits), _, _ = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

            new_halted = carry.halted | halted
            just_halted = new_halted & (~carry.halted)
            just_halted_mask = just_halted.unsqueeze(-1)
            
            new_final_actions = torch.where(just_halted_mask, torch.argmax(logits, dim=-1), carry.final_actions)

        return TinyRecursiveReasoningModel_ACTV2Carry(
            inner_carry=new_inner_carry,
            steps=new_steps,
            halted=new_halted,
            current_data=new_current_data,
            final_actions=new_final_actions
        ), outputs