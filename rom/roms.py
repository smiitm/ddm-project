"""
POD-Galerkin, balanced POD and exact balanced truncation (BT).

Each builder returns bases Phi, Psi (columns ordered by importance) and the singular values s
that ordered them. The r-state ROM keeps the first r columns.
"""

import numpy as np
from scipy.linalg import solve_continuous_lyapunov
from scipy.signal import lsim


def pod_basis(X):
    """Phi = Psi = left singular vectors of X."""
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    return U, U, s


def bpod_basis(X, Y, r_max):
    """Y^T X = U S V^T, Phi = X V S^-1/2, Psi = Y U S^-1/2 (Rowley 2005, eq. 24).

    Only the first r_max columns are formed, since 1/sqrt(s) blows up for the noise-level values.
    """
    U, s, Vt = np.linalg.svd(Y.T @ X, full_matrices=False)
    U_r, s_r, V_r = U[:, :r_max], s[:r_max], Vt[:r_max].T
    return X @ V_r / np.sqrt(s_r), Y @ U_r / np.sqrt(s_r), s


def gramian_factor(W):
    """Z with Z Z^T = W. Eigendecomposition, as W is too close to singular for Cholesky."""
    lam, V = np.linalg.eigh(W)
    return V * np.sqrt(np.clip(lam, 0, None))


def bt_basis(A, B, C):
    """Square-root BT: Zo^T Zc = U S V^T, Phi = Zc V S^-1/2, Psi = Zo U S^-1/2.

    Keeps the Hankel singular values above the noise level sqrt(n eps ||Wc|| ||Wo||).
    """
    Wc = solve_continuous_lyapunov(A, -B @ B.T)
    Wo = solve_continuous_lyapunov(A.T, -C.T @ C)
    Zc, Zo = gramian_factor((Wc + Wc.T) / 2), gramian_factor((Wo + Wo.T) / 2)
    U, s, Vt = np.linalg.svd(Zo.T @ Zc)
    noise = np.sqrt(A.shape[0] * np.finfo(float).eps * np.linalg.norm(Wc, 2) * np.linalg.norm(Wo, 2))
    k = int((s > noise).sum())
    return Zc @ Vt[:k].T / np.sqrt(s[:k]), Zo @ U[:, :k] / np.sqrt(s[:k]), s[:k]


def project(A, B, C, Phi, Psi, r):
    """The r-state ROM matrices (Ar, Br, Cr)."""
    P, Q = Phi[:, :r], Psi[:, :r]
    return Q.T @ A @ P, Q.T @ B, C @ P


def simulate(Ar, Br, Cr, u, t):
    """ROM output for input u, with the same first-order hold step as the full model."""
    D = np.zeros((Cr.shape[0], Br.shape[1]))
    return lsim((Ar, Br, Cr, D), u, t)[1]
