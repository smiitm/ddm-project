"""Drone positions, the sensing graphs and the state-space model (A, B, C).

Sensing matrix S: S[i, j] = 1 if drone i senses drone j (information flows j -> i).
Laplacian L = diag(S 1) - S. Followers obey  x_i' = kappa * sum_j S_ij (x_j - x_i).
"""

import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist

from . import parameters as params


def place_drones():
    """Random drone positions, and which drones are leaders, followers and monitors."""
    pos = np.random.default_rng(params.SEED_POSITIONS).uniform(0.0, params.L_BOX, size=(params.N, 2))
    leaders = [np.argmin(np.linalg.norm(pos - target, axis=1)) for target in params.LEADER_TARGETS]
    monitors = [np.argmin(np.linalg.norm(pos - target, axis=1)) for target in params.MONITOR_TARGETS]
    followers = [k for k in range(params.N) if k not in leaders]
    return pos, leaders, followers, monitors


def undirected_graph(pos):
    """Smallest radius R with mean degree >= target and a connected graph."""
    D = cdist(pos, pos)
    np.fill_diagonal(D, np.inf)
    for R in params.R_SEARCH:
        S = (D <= R).astype(float)
        if S.sum(1).mean() >= params.TARGET_MEAN_DEGREE and nx.is_connected(nx.from_numpy_array(S)):
            return S, float(R)


def thin(S_U, leaders, alpha):
    """Keep forward links; drop the others with probability alpha.

    A link "i senses j" is forward if j is fewer hops from a leader than i. With alpha = 0
    nothing is dropped, which gives the undirected graph back.
    """
    d = nx.multi_source_dijkstra_path_length(nx.from_numpy_array(S_U), set(leaders))
    hops = np.array([d[k] for k in range(params.N)])
    not_forward = hops[None, :] >= hops[:, None]      # [i, j]: j is not closer to the leaders than i
    U = np.random.default_rng(params.SEED_THINNING).uniform(size=S_U.shape)
    return S_U * ~(not_forward & (U < alpha))


def state_space(S, leaders, followers, monitors):
    """A = -kappa L_FF, B = -kappa L_FL, C picks the monitored followers.  Proposal eq. (2)"""
    L = np.diag(S.sum(1)) - S
    A = -params.KAPPA * L[np.ix_(followers, followers)]
    B = -params.KAPPA * L[np.ix_(followers, leaders)]
    C = np.zeros((len(monitors), len(followers)))
    for k, drone in enumerate(monitors):
        C[k, followers.index(drone)] = 1.0
    return A, B, C
