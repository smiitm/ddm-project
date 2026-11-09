"""Error measures used to compare a ROM with the full model."""

import numpy as np


def relative_error(y, y_rom):
    """||y - y_rom||_L2 / ||y||_L2 in %, over all outputs and times (the time grid is uniform)."""
    return 100 * np.linalg.norm(y - y_rom) / np.linalg.norm(y)


def max_error(y, y_rom):
    """Largest absolute error over all outputs and times [m]."""
    return np.abs(y - y_rom).max()


def steady_state_error(y, y_rom):
    """Largest absolute error over all outputs at the last time [m]."""
    return np.abs(y[-1] - y_rom[-1]).max()


def frequency_response(A, B, C, omega):
    """G(i w) = C (i w I - A)^-1 B at each frequency, shape (len(omega), p, m)."""
    I = np.eye(A.shape[0])
    return np.array([C @ np.linalg.solve(1j * w * I - A, B) for w in omega])


def gain(G):
    """Largest singular value of G at each frequency."""
    return np.linalg.norm(G, 2, axis=(1, 2))
