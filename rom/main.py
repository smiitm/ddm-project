"""Build the ROMs on both graphs, sweep r, and save the metrics in rom_results/ as CSV files."""

from pathlib import Path
from time import perf_counter

import numpy as np

from simulation.parameters import P_MONITORS
from simulation.snapshots import impulse_snapshots

from .metrics import frequency_response, gain, max_error, relative_error, steady_state_error
from .roms import bpod_basis, bt_basis, pod_basis, project, simulate

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "simulation_results"
RESULTS = ROOT / "rom_results"

R_MAX = 30                 # largest ROM size in the sweep
N_SAVE = 40                # singular values saved per method
R_RUN = 5                  # ROM size of the saved step run, small enough for the errors to show in a plot
TESTS = ("step", "random_11", "random_12", "random_13")
METHODS = ("pod", "bpod", "bt")
OMEGA = np.logspace(-4, 2, 400)   # frequencies of the frequency response [rad/s]
HSV_RATIO = 1e-3           # recommended r: smallest r with sigma_(r+1) / sigma_1 below this
ERROR_TARGET = 1.0         # recommended r: smallest r with error below this on every test [%]
N_TIMING = 5               # each timing keeps the fastest of this many runs


def load(folder, name, skiprows=0):
    return np.loadtxt(folder / f"{name}.csv", delimiter=",", skiprows=skiprows)


def save_csv(path, table, header=""):
    """Save numbers with full precision. A header is the first line of the file, if given."""
    np.savetxt(path, table, delimiter=",", fmt="%.17g", header=header, comments="")


def timed(function, *args):
    """The result of function(*args) and its fastest wall-clock time over N_TIMING runs [s]."""
    seconds = []
    for _ in range(N_TIMING):
        start = perf_counter()
        result = function(*args)
        seconds.append(perf_counter() - start)
    return result, min(seconds)


def smallest_r_hsv(s):
    """Smallest r with s[r] / s[0] below HSV_RATIO (s[r] is sigma_(r+1))."""
    return int(np.argmax(s[1:] / s[0] < HSV_RATIO)) + 1


if __name__ == "__main__":
    for name in ("undirected", "directed"):
        A, B, C, X, Y = (load(DATA / name, m) for m in ("A", "B", "C", "X", "Y"))
        times = load(DATA / name, "snapshot_times", skiprows=1)
        runs = {test: load(DATA / name, test, skiprows=1) for test in TESTS}
        folder = RESULTS / name
        folder.mkdir(parents=True, exist_ok=True)

        # 1. the bases and their offline cost
        _, snapshot_time = timed(impulse_snapshots, A, B, C, times[:, 0], times[:, 1])
        (Phi_pod, Psi_pod, s_pod), pod_time = timed(pod_basis, X)
        (Phi_bpod, Psi_bpod, s_bpod), bpod_time = timed(bpod_basis, X, Y, R_MAX)
        (Phi_bt, Psi_bt, s_bt), bt_time = timed(bt_basis, A, B, C)
        bases = {"pod": (Phi_pod, Psi_pod), "bpod": (Phi_bpod, Psi_bpod), "bt": (Phi_bt, Psi_bt)}

        # BT keeps only the values above noise; the rest of its column is nan
        s_bt_column = np.concatenate([s_bt, np.full(N_SAVE, np.nan)])
        table = np.column_stack([np.arange(1, N_SAVE + 1), s_pod[:N_SAVE], s_bpod[:N_SAVE],
                                 s_bt_column[:N_SAVE]])
        save_csv(folder / "singular_values.csv", table, "k,pod,bpod,bt")

        # 2. the full model on the step test and in frequency
        step = runs["step"]
        _, fom_time = timed(simulate, A, B, C, step[:, 1:4], step[:, 0])
        G = frequency_response(A, B, C, OMEGA)
        G_gain = gain(G)

        # 3. the sweep; a size a method does not reach stays nan
        relative = np.full((R_MAX, len(METHODS), len(TESTS)), np.nan)
        maximum = np.full((R_MAX, len(METHODS), len(TESTS)), np.nan)
        steady = np.full((R_MAX, len(METHODS)), np.nan)
        online = np.full((R_MAX, len(METHODS)), np.nan)
        max_real = np.full((R_MAX, len(METHODS)), np.nan)
        h_inf = np.full((R_MAX, len(METHODS)), np.nan)
        freq_error = np.full((len(OMEGA), len(METHODS) * R_MAX), np.nan)
        eigenvalues = []

        for i, method in enumerate(METHODS):
            Phi, Psi = bases[method]
            for r in range(1, min(R_MAX, Phi.shape[1]) + 1):
                Ar, Br, Cr = project(A, B, C, Phi, Psi, r)

                for j, test in enumerate(TESTS):
                    t, u, y = runs[test][:, 0], runs[test][:, 1:4], runs[test][:, 4:9]
                    y_rom = simulate(Ar, Br, Cr, u, t)
                    relative[r - 1, i, j] = relative_error(y, y_rom)
                    maximum[r - 1, i, j] = max_error(y, y_rom)
                    if test == "step":
                        steady[r - 1, i] = steady_state_error(y, y_rom)

                _, online[r - 1, i] = timed(simulate, Ar, Br, Cr, step[:, 1:4], step[:, 0])

                G_error = gain(G - frequency_response(Ar, Br, Cr, OMEGA))
                freq_error[:, i * R_MAX + r - 1] = G_error / G_gain
                h_inf[r - 1, i] = G_error.max() / G_gain.max()

                lam = np.linalg.eigvals(Ar)
                max_real[r - 1, i] = lam.real.max()
                eigenvalues += [(method, r, value.real, value.imag) for value in lam]

        # 4. save the metrics
        sizes = np.arange(1, R_MAX + 1)
        by_method = "r," + ",".join(METHODS)
        by_method_test = "r," + ",".join(f"{m}_{test}" for m in METHODS for test in TESTS)
        save_csv(folder / "relative_error.csv", np.column_stack([sizes, relative.reshape(R_MAX, -1)]),
                 by_method_test)
        save_csv(folder / "max_error.csv", np.column_stack([sizes, maximum.reshape(R_MAX, -1)]),
                 by_method_test)
        save_csv(folder / "steady_state_error.csv", np.column_stack([sizes, steady]), by_method)
        save_csv(folder / "online_time.csv", np.column_stack([sizes, online]), by_method)
        save_csv(folder / "max_real_eigenvalue.csv", np.column_stack([sizes, max_real]), by_method)
        save_csv(folder / "h_inf_error.csv", np.column_stack([sizes, h_inf]), by_method)
        save_csv(folder / "frequency_error.csv", np.column_stack([OMEGA, G_gain, freq_error]),
                 "omega,gain," + ",".join(f"{m}_r{r}" for m in METHODS for r in sizes))

        with open(folder / "eigenvalues.csv", "w") as f:
            f.write("method,r,real,imag\n")
            for method, r, real, imag in eigenvalues:
                f.write(f"{method},{r},{real:.17g},{imag:.17g}\n")

        with open(folder / "offline_time.csv", "w") as f:
            f.write("name,seconds\n")
            for key, value in (("snapshots", snapshot_time), ("pod_basis", pod_time), ("bpod_basis", bpod_time),
                               ("bt_basis", bt_time), ("full_model_step_run", fom_time)):
                f.write(f"{key},{value:.17g}\n")

        # smallest r below the error target on every test; nan if no size reaches it
        below = relative.max(axis=2) < ERROR_TARGET
        with open(folder / "recommended_r.csv", "w") as f:
            f.write("name,r\n")
            f.write(f"hsv_bpod,{smallest_r_hsv(s_bpod)}\n")
            f.write(f"hsv_bt,{smallest_r_hsv(s_bt)}\n")
            for i, method in enumerate(METHODS):
                r = sizes[below[:, i]].min() if below[:, i].any() else np.nan
                f.write(f"error_{method},{r}\n")

        # 5. the step test at one ROM size, for plotting
        t, u = step[:, 0], step[:, 1:4]
        roms = {m: project(A, B, C, *bases[m], R_RUN) for m in METHODS}
        header = ",".join(["t"] + [f"{m}_y{k + 1}" for m in METHODS for k in range(P_MONITORS)])
        table = np.column_stack([t] + [simulate(*roms[m], u, t) for m in METHODS])
        save_csv(folder / f"step_r{R_RUN}.csv", table, header)
        print(f"{name}: saved")
    print(f"Done. Results are in {RESULTS}")
