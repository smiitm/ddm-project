"""Numbers and random seeds for the swarm, graphs and dynamics."""

import numpy as np

# swarm
N = 300                    # number of drones
M_LEADERS = 3              # leaders = inputs
P_MONITORS = 5             # monitored followers = outputs
L_BOX = 1000.0             # side of the deployment square [m]
SEED_POSITIONS = 1         # drone positions
SEED_THINNING = 2          # random numbers used to drop links in the directed graph


# leaders: drones nearest to these points on the edge x = 0
LEADER_TARGETS = [(0.0, L_BOX / 4), (0.0, L_BOX / 2), (0.0, 3 * L_BOX / 4)]
# monitored drones: drones nearest to evenly spaced points on the edge x = L_BOX
MONITOR_TARGETS = [(L_BOX, L_BOX * k / (P_MONITORS + 1)) for k in range(1, P_MONITORS + 1)]


# sensing graphs
GRAPHS = {"undirected": 0.0, "directed": 0.9}   # graph name: alpha (probability of dropping a non-forward link)
TARGET_MEAN_DEGREE = 10.0                       # undirected graph: smallest R with mean degree >= this and connected
R_SEARCH = np.arange(60.0, 200.0, 0.5)          # candidate radii [m]


# dynamics
KAPPA = 1.0                # consensus gain [1/s]


# time grids
T_ROUND = 50.0             # round T and the step-test length up to a multiple of this [s]
