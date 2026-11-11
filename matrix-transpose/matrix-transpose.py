import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    # Write code here
    ROWS = len(A)
    COLS = len(A[0])
    res = [[0] * ROWS for _ in range(COLS)]
    for r in range(ROWS):
        for c in range(COLS):
            res[c][r] = A[r][c]
    return np.asarray(res)