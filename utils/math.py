import numpy as np
import torch

def linear_cka(X, Y):
    """X: (N, D), Y: (N, D) -> scalar"""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    YtX = Y.T @ X          # (D, D)
    XtX = X.T @ X          # (D, D)
    YtY = Y.T @ Y          # (D, D)
    return float(np.linalg.norm(YtX, 'fro') ** 2 /
                 (np.linalg.norm(XtX, 'fro') * np.linalg.norm(YtY, 'fro')))


def softmax_entropy(logits):
    shifted_logits = logits - logits.max(axis=-1, keepdims=True)
    log_sum_exp = np.log(np.exp(shifted_logits).sum(axis=-1, keepdims=True))
    log_probs = shifted_logits - log_sum_exp
    probs = np.exp(log_probs)
    return -(probs * log_probs).sum(axis=-1)

def logit_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: (B, seq_len, n_classes)
    return: (B, seq_len)  — entropy in nats
    """
    log_probs = torch.log_softmax(logits, dim=-1)   # numerically stable
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)