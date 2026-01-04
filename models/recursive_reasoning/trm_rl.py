# models/recursive_reasoning/trm_rl.py

from typing import Dict, Tuple
import torch
from torch import nn

from .trm import (
    TinyRecursiveReasoningModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1Carry,
)


class TinyRecursiveReasoningModel_RL(TinyRecursiveReasoningModel_ACTV1):
    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> TinyRecursiveReasoningModel_ACTV1Carry:
        device = batch["inputs"].device
        batch_size = batch["inputs"].shape[0]

        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.zeros((batch_size,), dtype=torch.bool, device=device),
            current_data={k: v.clone() for k, v in batch.items()},
        )

    def forward(
        self,
        carry: TinyRecursiveReasoningModel_ACTV1Carry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[TinyRecursiveReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:
        inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)

        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(
            inner_carry,
            batch,
        )

        outputs = {
            "logits": logits,  # (B, L, V)
            "q_halt_logits": q_halt_logits,        # (B,)
            "q_continue_logits": q_continue_logits # (B,)
        }

        new_carry = TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=new_inner_carry,
            steps=carry.steps,
            halted=carry.halted,
            current_data=batch,
        )

        return new_carry, outputs
