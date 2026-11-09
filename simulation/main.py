"""Run every simulation and save the results in simulation_results/ as CSV files."""

from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.signal import lsim

from . import parameters as params
from .inputs import DT, RAND_GAIN_RATIO, RAND_SEEDS, STEP_RAMP, STEP_SETTLE, random_input, step_input
from .snapshots import T_FACTOR, impulse_snapshots, snapshot_grid
from .swarm import place_drones, state_space, thin, undirected_graph

RESULTS = Path(__file__).resolve().parents[1] / "simulation_results"


def slowest_time_constant(A):
    return 1 / (-np.linalg.eigvals(A).real).min()


def band_limit(A, B, C):
    """Frequency where the gain from leader 1 to the monitors first falls to RAND_GAIN_RATIO
    of its zero-frequency value."""
    n, b = A.shape[0], B[:, 0]
    gain = lambda w: np.linalg.norm(C @ np.linalg.solve(1j * w * np.eye(n) - A, b))
    target = RAND_GAIN_RATIO * np.linalg.norm(C @ np.linalg.solve(-A, b))
    grid = np.logspace(-5, 2, 200)
    k = int(np.argmax([gain(w) < target for w in grid]))
    return brentq(lambda w: gain(w) - target, grid[k - 1], grid[k], xtol=1e-12)


def save_csv(path, table, header=""):
    """Save numbers with full precision. A header is the first line of the file, if given."""
    np.savetxt(path, table, delimiter=",", fmt="%.17g", header=header, comments="")


if __name__ == "__main__":
    # 1. the drones and both sensing graphs
    pos, leaders, followers, monitors = place_drones()
    S_U, R = undirected_graph(pos)
    systems = {}
    for name, alpha in params.GRAPHS.items():
        S = thin(S_U, leaders, alpha)
        systems[name] = (S, *state_space(S, leaders, followers, monitors))

    # 2. time grids and the random-input band limit, the same for both graphs
    tau = {name: slowest_time_constant(A) for name, (S, A, B, C) in systems.items()}
    tau_max = max(tau.values())
    T = params.T_ROUND * np.ceil(T_FACTOR * tau_max / params.T_ROUND)
    step_end = params.T_ROUND * np.ceil((STEP_RAMP + STEP_SETTLE * tau_max) / params.T_ROUND)
    omega_c = band_limit(*systems["undirected"][1:])

    t_snap, w_snap = snapshot_grid(T)
    t_step = np.arange(0.0, step_end + DT / 2, DT)
    t_rand = np.arange(0.0, T + DT / 2, DT)

    # 3. the four test inputs
    tests = {"step": (t_step, step_input(t_step))}
    for seed in RAND_SEEDS:
        tests[f"random_{seed}"] = (t_rand, random_input(t_rand, omega_c, seed))

    # 4. simulate each graph and save its results in its own folder
    for name, (S, A, B, C) in systems.items():
        folder = RESULTS / name
        folder.mkdir(parents=True, exist_ok=True)
        for matrix_name, matrix in (("S", S), ("A", A), ("B", B), ("C", C)):
            save_csv(folder / f"{matrix_name}.csv", matrix)

        X, Y = impulse_snapshots(A, B, C, t_snap, w_snap)
        save_csv(folder / "X.csv", X)
        save_csv(folder / "Y.csv", Y)
        save_csv(folder / "snapshot_times.csv", np.column_stack([t_snap, w_snap]), "t,w")

        D = np.zeros((C.shape[0], B.shape[1]))
        for test, (t, u) in tests.items():
            _, y, _ = lsim((A, B, C, D), u, t)     # first-order hold: u is a straight line between samples
            save_csv(folder / f"{test}.csv", np.column_stack([t, u, y]), "t,u1,u2,u3,y1,y2,y3,y4,y5")
        print(f"{name}: saved")

    # 5. what is shared by both graphs
    roles = ["follower"] * params.N
    for k, drone in enumerate(leaders):
        roles[drone] = f"leader_{k + 1}"
    for k, drone in enumerate(monitors):
        roles[drone] = f"monitor_{k + 1}"
    with open(RESULTS / "drones.csv", "w") as f:
        f.write("drone,x,y,role\n")
        for k in range(params.N):
            f.write(f"{k},{pos[k, 0]:.17g},{pos[k, 1]:.17g},{roles[k]}\n")
    with open(RESULTS / "summary.csv", "w") as f:
        f.write("name,value\n")
        for key, value in (("R", R), ("T", T), ("step_duration", step_end), ("omega_c", omega_c),
                           ("tau_undirected", tau["undirected"]), ("tau_directed", tau["directed"])):
            f.write(f"{key},{value:.17g}\n")
    print(f"Done. Results are in {RESULTS}")
