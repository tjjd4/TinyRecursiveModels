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
    steps: torch.Tensor
    halted: torch.Tensor

    total_logprob: torch.Tensor    # (N,)
    total_entropy: torch.Tensor    # (N,)
    final_actions: torch.Tensor    # (N, L)
    final_halt_actions: torch.Tensor    # (N, L)

    current_data: Dict[str, torch.Tensor]


class TinyRecursiveReasoningModel_RL(TinyRecursiveReasoningModel_ACTV1):
    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> TinyRecursiveReasoningModel_GRPOCarry:
        device = batch["inputs"].device
        batch_size = batch["inputs"].shape[0]

        return TinyRecursiveReasoningModel_GRPOCarry(
            inner_carry=self.inner.empty_carry(batch_size),
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32, device=device),
            halted=torch.ones(batch_size, dtype=torch.bool, device=device),

            total_logprob=torch.zeros(batch_size, device=device),
            total_entropy=torch.zeros(batch_size, device=device),
            final_actions=torch.zeros_like(batch["inputs"], dtype=torch.long),
            final_halt_actions=torch.zeros(batch_size, dtype=torch.long, device=device),

            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_GRPOCarry):
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(reset_flag, carry.inner_carry)
        return TinyRecursiveReasoningModel_GRPOCarry(
            inner_carry=new_inner_carry,
            steps=torch.zeros_like(carry.steps),
            halted=torch.zeros_like(carry.halted),
            total_logprob=torch.zeros_like(carry.total_logprob),
            total_entropy=torch.zeros_like(carry.total_entropy),
            final_actions=torch.zeros_like(carry.final_actions),
            final_halt_actions=torch.zeros_like(carry.final_halt_actions),
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

        # Step
        new_steps = torch.where(carry.halted, new_steps, new_steps + 1)

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

        token_dist = torch.distributions.Categorical(logits=logits)

        # Training: sample for exploration; Eval: argmax for deterministic prediction
        if self.training:
            halt_action = halt_dist.sample()  # (N,) 0=Cont, 1=Halt
            token_action = token_dist.sample()  # (N,) 
        else:
            if not self.config.no_ACT_continue:
                halt_action = halt_logits.argmax(dim=-1)  # (N,) deterministic
            else:
                halt_action = (halt_logits > 0).long()
            token_action = token_dist.argmax(dim=-1)  # (N,) deterministic
        
        is_last_step = new_steps >= self.config.halt_max_steps

        halt_action = torch.where(is_last_step, torch.ones_like(halt_action), halt_action)

        # Mask for sequences that were active before this step
        active_mask_float = (~carry.halted).float()
        new_halted = carry.halted | (halt_action == 1)

        # Mask for sequences that were just halted in this step
        just_halted = new_halted & (~carry.halted)
        just_halted_mask_float = just_halted.float()

        # halt log prob
        halt_step_logprob = halt_dist.log_prob(halt_action.float())  # (N,)
        halt_step_entropy = halt_dist.entropy()  # (N,)

        # token log prob
        token_step_logprob = token_dist.log_prob(token_action.float()).sum(dim=-1)
        token_step_entropy = token_dist.entropy().sum(dim=-1)

        # Accumulate log_prob and entropy for active sequences
        # active_mask ensures only accumulate for sequences that haven't halted yet (including the current step)
        new_total_logprob = carry.total_logprob + halt_step_logprob * active_mask_float + token_step_logprob * just_halted_mask_float
        new_total_entropy = carry.total_entropy + halt_step_entropy * active_mask_float + token_step_entropy * just_halted_mask_float

        with torch.no_grad():
            just_halted_mask = just_halted.unsqueeze(-1)
            
            new_final_actions = torch.where(just_halted_mask, torch.argmax(logits, dim=-1), carry.final_actions)
            new_final_halt_actions = torch.where(just_halted, (q_halt_logits >= 0).long(), carry.final_halt_actions)

        new_carry = TinyRecursiveReasoningModel_GRPOCarry(
            inner_carry=new_inner_carry,
            steps=new_steps,
            halted=new_halted,
            current_data=new_current_data,
            total_logprob=new_total_logprob,
            total_entropy=new_total_entropy,
            final_actions=new_final_actions,
            final_halt_actions=new_final_halt_actions,
        )

        return new_carry, outputs
