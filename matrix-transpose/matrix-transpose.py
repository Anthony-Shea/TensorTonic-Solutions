import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    # Write code here
    A = np.asarray(A)
    ROWS, COLS = A.shape
    T = np.zeros((COLS,ROWS), dtype=A.dtype)
    for r in range(ROWS):
        for c in range(COLS):
            T[c,r] = A[r,c]
    return T