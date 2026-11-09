# Balanced POD on a leader-follower drone swarm

Code for the AS5401 project by Smit Dhameliya (AE23B107), IIT Madras.

A swarm of 300 drones follows 3 leaders through a linear consensus law. The followers form a
linear system `x' = A x + B u`, `y = C x` with n = 297 states, m = 3 inputs (leader positions)
and p = 5 outputs (five monitored drones on the far side of the swarm). We reduce this model with
POD-Galerkin and balanced POD (Rowley 2005), and compare both with exact balanced truncation (BT).
The same drones are used with two sensing graphs:

| Name | alpha | Meaning |
|---|---|---|
| `undirected` | 0 | every link is two-way |
| `directed` | 0.9 | most backward and sideways links are dropped, so information mostly flows away from the leaders |

## Requirements

Python 3.12 with the package versions in `requirements.txt` (NumPy, SciPy, networkx, Matplotlib):

```bash
pip install -r requirements.txt
```

## How to run

Run the three steps in this order from inside code folder. Each step reads what the one before it saved.

```bash
python -m simulation.main
python -m rom.main
python -m plots.main
```

| Step | Time | What it does |
|---|---|---|
| `simulation.main` | about 1 min | builds the full model on both graphs, runs the test inputs and saves the snapshots |
| `rom.main` | about 5.5 min | builds the ROMs, sweeps r from 1 to 30 and saves every metric |
| `plots.main` | under 10 s | makes the figures |

The timings in `rom_results/*/offline_time.csv` and `online_time.csv` depend on the computer, so
they change from run to run. Every other number comes out the same each time.

The repository leaves out the large files (about 83 MB): the matrices `A`, `S`, `X`, `Y`, the
full-model test runs and `step_r5.csv`. Running the three steps makes them again.

## Code

| File | Contents |
|---|---|
| `simulation/parameters.py` | Numbers and seeds for the swarm, graphs and dynamics. |
| `simulation/swarm.py` | Places the drones, builds the graphs, and forms `A`, `B`, `C`. |
| `simulation/inputs.py` | The test inputs: a smooth 50 m step, and random sums of cosines (seeds 11, 12, 13). |
| `simulation/snapshots.py` | The weighted impulse snapshots `X` (forward) and `Y` (adjoint), and their time grid. |
| `simulation/main.py` | Runs the full model and saves `simulation_results/`. |
| `rom/roms.py` | The POD, BPOD and BT bases, the projection to an r-state model, and the ROM simulation. |
| `rom/metrics.py` | The error measures and the frequency response. |
| `rom/main.py` | Builds the ROMs on both graphs and saves `rom_results/`. |
| `plots/main.py` | Makes the figures in `figures/`. |

The plotting code was written with help from an AI assistant.

## Results

Files without a header line: `A`, `B`, `C`, `S`, `X` and `Y`. All the others have one. To load them:

```python
import numpy as np
A = np.loadtxt("simulation_results/directed/A.csv", delimiter=",")
step = np.loadtxt("simulation_results/directed/step.csv", delimiter=",", skiprows=1)
t, u, y = step[:, 0], step[:, 1:4], step[:, 4:9]
```

### `simulation_results/`

`drones.csv` and `summary.csv` are shared by both graphs. The rest is in one folder per graph,
`undirected/` and `directed/`, with the same files in each.

| File | Rows × columns | Meaning |
|---|---|---|
| `drones.csv` | 300 × 4 | `drone,x,y,role`: position [m] and role (`leader_1`..`leader_3`, `monitor_1`..`monitor_5` or `follower`). Drones are numbered 0 to 299. Monitors are also followers. |
| `summary.csv` | 6 × 2 | sensing radius `R` [m], snapshot horizon `T` [s], `step_duration` [s], random-input band limit `omega_c` [rad/s], and the slowest time constant of each graph `tau_undirected`, `tau_directed` [s] |
| `A.csv` | 297 × 297 | state k is the k-th follower, counting up in drone number and skipping the leaders |
| `B.csv` | 297 × 3 | column j is `leader_j+1`, counting from 0 |
| `C.csv` | 5 × 297 | row i is `monitor_i+1`, counting from 0 |
| `S.csv` | 300 × 300 | `S[i, j] = 1` if drone i senses drone j |
| `X.csv` | 297 × 1500 | forward impulse snapshots, already weighted: the 500 times of input 1, then input 2, then input 3 |
| `Y.csv` | 297 × 2500 | adjoint impulse snapshots, already weighted: the 500 times of output 1, then output 2, and so on |
| `snapshot_times.csv` | 500 × 2 | snapshot time `t` [s] and trapezoid weight `w` |
| `step.csv` | 35001 × 9 | `t,u1,u2,u3,y1,...,y5`: time [s], leader and monitored drone displacements [m] |
| `random_11.csv` | 9001 × 9 | random test with seed 11, same columns as `step.csv` (also `random_12`, `random_13`) |

### `rom_results/`

One folder per graph, with the same files in each. Most tables have one row per ROM size r = 1
to 30. A `nan` means the method does not reach that size: BT keeps only the Hankel singular values
above round-off noise, so its columns can stop early.

| File | Columns | Meaning |
|---|---|---|
| `singular_values.csv` | `k,pod,bpod,bt` | the first 40 singular values: of `X` for POD, of `Y^T X` for BPOD, and the exact Hankel singular values for BT |
| `relative_error.csv` | `r,pod_step,pod_random_11,...` | relative output error [%] for each method and test run |
| `max_error.csv` | same as above | largest absolute output error [m] |
| `steady_state_error.csv` | `r,pod,bpod,bt` | largest output error at the end of the step test [m] |
| `h_inf_error.csv` | `r,pod,bpod,bt` | relative H-infinity error, taken over the frequency grid |
| `frequency_error.csv` | `omega,gain,pod_r1,...,bt_r30` | full-model gain, then the relative frequency-response error of each ROM, at 400 frequencies |
| `max_real_eigenvalue.csv` | `r,pod,bpod,bt` | largest real part of the ROM eigenvalues (above 0 means unstable) |
| `eigenvalues.csv` | `method,r,real,imag` | every eigenvalue of every ROM |
| `online_time.csv` | `r,pod,bpod,bt` | time to run the step test with the ROM [s], fastest of 5 runs |
| `offline_time.csv` | `name,seconds` | time to build the snapshots and each basis, and to run the full model on the step test |
| `recommended_r.csv` | `name,r` | smallest r with `sigma_(r+1)/sigma_1 < 1e-3` (BPOD and BT), and smallest r with error below 1% on all four test runs |
| `step_r5.csv` | `t,pod_y1,...,bt_y5` | output of each r = 5 ROM on the step test |

### `figures/`
LLMs were used to plot figures and improve their asthetics.
Each figure is saved as PDF and PNG.

| Figure | Shows |
|---|---|
| `layout` | the drones and both sensing graphs |
| `responses` | full-model responses to the step test and one random test |
| `singular_values` | normalized singular values of each method |
| `error_vs_r` | worst relative output error against r |
| `error_ratio` | POD error divided by BPOD error against r |
| `frequency_error` | relative frequency-response error at r = 10 |
| `step_rom` | step response of one monitored drone with the r = 5 ROMs |
