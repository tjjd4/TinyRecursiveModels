import numpy as np
import torch

data = np.load("checkpoints/Sudoku-extreme-1k-aug-1000-trace-torch/z_analysis_pretrain_mlp_t_sudoku/z_analysis_step_0/z_raw.npz", allow_pickle=True)
trajs  = list(data["trajectories"])   # list of (T, D) — T steps, D hidden
flags  = list(data["correct_flags"])  # bool

# 取每條 correct trajectory 的最後一個 step
correct_finals = np.stack([
    t[-1] for t, f in zip(trajs, flags) if f
])  # (N_correct, D)

centroid = correct_finals.mean(axis=0)  # (D,)
print("centroid shape:", centroid.shape)
print("centroid norm:", np.linalg.norm(centroid))