from pydantic import BaseModel
import torch


class V1RewardConfig(BaseModel):
    name: str
    correct_reward: float = 1.0
    len_penalty: float = 0.1
    p_cutoff: float = 0.9


class V1Reward:
    def __init__(self, config: V1RewardConfig):
        self.config = config

    def compute(self, seq_is_correct: torch.Tensor, final_steps: torch.Tensor, max_steps: int) -> torch.Tensor:
        rewards = torch.zeros_like(seq_is_correct, dtype=torch.float32)
        rewards[seq_is_correct] = self.config.correct_reward
        rewards = rewards * (1 - self.config.len_penalty * (final_steps / max_steps))
        return rewards