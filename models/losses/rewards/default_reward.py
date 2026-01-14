import torch
from pydantic import BaseModel


class DefaultRewardConfig(BaseModel):
    correct_reward: float = 1.0
    len_penalty: float = 0.1


class DefaultReward:
    def __init__(self, config_dict: dict):
        self.config = DefaultRewardConfig(**config_dict)
    
    def compute(
        self,
        seq_is_correct: torch.Tensor,
        final_steps: torch.Tensor,
        max_steps: int
    ) -> torch.Tensor:
        # Correctness reward
        r_correct = seq_is_correct * self.config.correct_reward
        
        # Length penalty (only for correct sequences)
        r_len = torch.where(
            seq_is_correct,
            -self.config.len_penalty * (final_steps / float(max(1, max_steps))),
            torch.zeros_like(seq_is_correct)
        )
        
        return r_correct + r_len