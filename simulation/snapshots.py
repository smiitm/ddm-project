"""Impulse and adjoint snapshots, the data used to build POD and BPOD models."""

import numpy as np
from scipy.linalg import expm

T_FACTOR = 5.0             # T = T_FACTOR * slowest time constant (over both graphs)
N_SNAP = 500               # impulse snapshots per input / output
T_SNAP_MIN = 0.01          # first nonzero snapshot time [s]; the grid is log-spaced up to T


def snapshot_grid(T):
    """Snapshot times (t = 0, then log-spaced up to T) and their trapezoid weights."""
    t = np.concatenate([[0.0], np.geomspace(T_SNAP_MIN, T, N_SNAP - 1)])
    w = np.empty_like(t)
    w[0], w[-1] = (t[1] - t[0]) / 2, (t[-1] - t[-2]) / 2
    w[1:-1] = (t[2:] - t[:-2]) / 2
    return t, w


def impulse_snapshots(A, B, C, t, w):
    """Weighted forward and adjoint impulse snapshots.

    X columns: sqrt(w_k) e^{A t_k} b_j,          so X X^T is approximately W_c.
    Y columns: sqrt(w_k) e^{A^T t_k} c_i^T,      so Y Y^T is approximately W_o.
    Columns are grouped by input (or output): all times of the first, then all of the second.
    """
    n, m, p = A.shape[0], B.shape[1], C.shape[0]
    X = np.empty((n, m, len(t)))
    Y = np.empty((n, p, len(t)))
    for k in range(len(t)):
        E = expm(A * t[k])
        X[:, :, k] = np.sqrt(w[k]) * (E @ B)
        Y[:, :, k] = np.sqrt(w[k]) * (E.T @ C.T)
    return X.reshape(n, m * len(t)), Y.reshape(n, p * len(t))
