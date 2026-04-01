import numpy as np

def linear_cka(X, Y):
    """X: (N, D), Y: (N, D) -> scalar"""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    YtX = Y.T @ X          # (D, D)
    XtX = X.T @ X          # (D, D)
    YtY = Y.T @ Y          # (D, D)
    return float(np.linalg.norm(YtX, 'fro') ** 2 /
                 (np.linalg.norm(XtX, 'fro') * np.linalg.norm(YtY, 'fro')))