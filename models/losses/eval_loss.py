from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
from models.losses.loss_fn import IGNORE_LABEL_ID


class EvalLossHead(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            # Preds
            if hasattr(new_carry, "final_actions"):
                outputs["preds"] = new_carry.final_actions
            else:
                outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            if hasattr(new_carry, "final_halt_actions"):
                outputs["q_halt_preds"] = new_carry.final_halt_actions
            else:
                outputs["q_halt_preds"] = (outputs["halt_logits"] >= 0).long()

            # Correctness
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (outputs["preds"] == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                
                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & (outputs["q_halt_preds"] == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, None, metrics, detached_outputs, new_carry.halted.all()
