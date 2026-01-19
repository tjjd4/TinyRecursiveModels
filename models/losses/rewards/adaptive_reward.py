from pydantic import BaseModel
import torch

class AdaptiveRewardConfig(BaseModel):
    num_generations: int
    correct_reward: float
    len_penalty: float
    len_reward: float
    p_cutoff: float

class AdaptiveReward:
    def __init__(self, config_dict: dict):
        self.config = AdaptiveRewardConfig(**config_dict)

    def compute(
        self, 
        seq_is_correct: torch.Tensor, 
        final_steps: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        N = seq_is_correct.shape[0]
        G = self.config.num_generations
        
        if N % G != 0:
            raise ValueError(f"Total size {N} not divisible by num_generations {G}")
        
        B = N // G
        
        r_correct = seq_is_correct.float() * self.config.correct_reward

        correct_mask_2d = seq_is_correct.view(B, G).float()
        steps_2d = final_steps.view(B, G).float()

        p_correct = correct_mask_2d.mean(dim=1, keepdim=True)  # (B, 1)

        mean_len = steps_2d.mean(dim=1, keepdim=True)  # (B, 1)
        max_len = steps_2d.max(dim=1, keepdim=True).values  # (B, 1)
        min_len = steps_2d.min(dim=1, keepdim=True).values  # (B, 1)

        normalized_len = (steps_2d - mean_len) / (max_len - min_len).clamp(min=1e-6)

        is_easy = p_correct >= self.config.p_cutoff

        len_contribution = torch.where(
            is_easy,
            -self.config.len_penalty * normalized_len,  # Easy: Penalize long
            self.config.len_reward * normalized_len     # Hard: Reward long
        )

        len_contribution = len_contribution.view(-1)
        
        rewards = r_correct + len_contribution * seq_is_correct.float()

        return rewards