import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here\
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    N, D = X.shape
    w = np.zeros(D)
    b = 0.0
    for _ in range(steps):
        Z = X @ w + b
        p = _sigmoid(Z)
        L = (-1/N) * (y*np.log(p) + (1-y)* np.log(1-p))
        dw = X.T @ (p - y) / N
        db = np.mean(p-y)
        w = w - lr * dw
        b = b - lr * db
    return (w,b)