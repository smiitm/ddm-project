"""Plot the saved results and save the figures in figures/ as PDF and PNG files."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rom.main import HSV_RATIO, METHODS, N_SAVE, R_MAX, R_RUN, TESTS
from simulation.parameters import P_MONITORS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "simulation_results"
RESULTS = ROOT / "rom_results"
FIGURES = ROOT / "figures"

GRAPHS = ("undirected", "directed")
R_FREQ = 10                # ROM size in the frequency-response figure
OMEGA_MAX = 10.0           # the frequency figure stops here [rad/s]; the full-model gain is about 1e-9 or less by then
STEP_END = {"undirected": 450.0, "directed": 100.0}   # x-axis of the ROM step figure [s]
MONITOR = 2                # monitored drone in the ROM step figure, counting from 0
WIDTH = 7.16               # full page width of an IEEE paper [in]
SMALL = {"font.size": 7, "axes.titlesize": 7, "legend.fontsize": 6.5}   # fonts for one-column figures

INK = "#0b0b0b"
GREY = "#8a8985"
LABEL = {"pod": "POD-Galerkin", "bpod": "BPOD", "bt": "Exact BT"}
METHOD_STYLE = {"pod": dict(color="#2a78d6", ls="--", marker="s", ms=2.2, lw=1.0),
                "bpod": dict(color="#eb6834", ls="none", marker="x", ms=3.5, mew=1.0),
                "bt": dict(color="#1baf7a", ls="-", lw=1.3)}
GRAPH_STYLE = {"undirected": dict(color="#6250d6", ls="-", lw=1.5),
               "directed": dict(color="#eda100", ls="--", lw=1.5)}

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"],
                     "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "legend.frameon": False, "savefig.dpi": 200})


def load(folder, name, skiprows=0):
    return np.loadtxt(folder / f"{name}.csv", delimiter=",", skiprows=skiprows)


def worst_error(graph):
    """Largest relative error over the test runs, shape (R_MAX, number of methods)."""
    table = load(RESULTS / graph, "relative_error", skiprows=1)
    return table[:, 1:].reshape(R_MAX, len(METHODS), len(TESTS)).max(axis=2)


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png")
    plt.close(fig)


def legend_on_top(fig, axis, ncol):
    fig.legend(*axis.get_legend_handles_labels(), loc="outside upper center", ncol=ncol, handlelength=1.5,
               columnspacing=0.8, handletextpad=0.4)


def column_pair(sharey):
    """Two panels side by side that fit in one column of the paper."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH / 2, 2.0), sharey=sharey, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.04, wspace=0.02)
    for ax in axes:
        ax.tick_params(length=2, pad=1.5)
    return fig, axes


def plot_layout():
    drones = np.genfromtxt(DATA / "drones.csv", delimiter=",", skip_header=1, dtype=None, encoding=None)
    pos = np.array([[d[1], d[2]] for d in drones])
    role = np.array([d[3] for d in drones])
    leaders = np.char.startswith(role, "leader")
    monitors = np.char.startswith(role, "monitor")

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH / 2, 2.15), sharey=True, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.04, wspace=0.02)
    for ax, graph in zip(axes, GRAPHS):
        S = load(DATA / graph, "S") > 0
        i, j = np.nonzero(S & S.T & np.triu(np.ones_like(S), 1).astype(bool))
        two_way = np.column_stack([pos[i, 0], pos[j, 0], np.full(len(i), np.nan)]).ravel(), \
            np.column_stack([pos[i, 1], pos[j, 1], np.full(len(i), np.nan)]).ravel()
        ax.plot(*two_way, color="#c3c2b7", lw=0.35, label="Two-way link")

        one_way = S & ~S.T
        one_way[leaders] = False         # leaders follow their own paths, so what they sense has no effect
        i, j = np.nonzero(one_way)       # drone i senses drone j, so information flows j -> i
        if len(i):
            # each arrow stops short of its drone so that the head stays visible
            color = GRAPH_STYLE["directed"]["color"]
            ax.quiver(pos[j, 0], pos[j, 1], 0.85 * (pos[i, 0] - pos[j, 0]), 0.85 * (pos[i, 1] - pos[j, 1]),
                      angles="xy", scale_units="xy", scale=1, color=color, width=0.003, headwidth=4, headlength=5)
            ax.plot([], [], color=color, lw=0.8, marker=">", ms=3, label="One-way link")

        ax.scatter(*pos.T, s=1.5, color=GREY, zorder=3, lw=0)
        ax.scatter(*pos[leaders].T, s=18, marker="^", color=INK, zorder=4, label="Leader")
        ax.scatter(*pos[monitors].T, s=13, marker="s", facecolor="white", edgecolor=INK, lw=0.8, zorder=4,
                   label="Monitored drone")
        links = int(S[~leaders].sum())   # sensing links that end at followers, as in the proposal
        ax.set(title=f"{graph}, {links} links", xlabel="$x$ [m]", aspect="equal",
               xlim=(-20, 1020), ylim=(-20, 1020), xticks=[0, 500, 1000], yticks=[0, 500, 1000])
        ax.tick_params(length=2, pad=1.5)
        ax.grid(False)
    axes[0].set_ylabel("$y$ [m]", labelpad=1)
    legend_on_top(fig, axes[1], 4)
    save(fig, "layout")


def plot_singular_values():
    fig, axes = column_pair(sharey=True)
    for ax, graph in zip(axes, GRAPHS):
        table = load(RESULTS / graph, "singular_values", skiprows=1)
        for k, method in enumerate(METHODS):
            ax.semilogy(table[:, 0], table[:, k + 1] / table[0, k + 1], label=LABEL[method],
                        **METHOD_STYLE[method])
        ax.axhline(HSV_RATIO, color=GREY, ls=":", lw=1)
        ax.text(N_SAVE, 1.5 * HSV_RATIO, "$10^{-3}$", ha="right", color="#52514e")
        ax.set(title=graph, xlabel="index $k$")
    axes[0].set_ylabel(r"$\sigma_k/\sigma_1$")
    legend_on_top(fig, axes[0], 3)
    save(fig, "singular_values")


def plot_responses():
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.6), layout="constrained")
    for ax, test, title in zip(axes, ("step", "random_11"), ("step test", "random test, seed 11")):
        for graph in GRAPHS:
            run = load(DATA / graph, test, skiprows=1)
            keep = run[:, 0] <= 450.0
            lines = ax.plot(run[keep, 0], run[keep, 4:9], **GRAPH_STYLE[graph])
            lines[0].set_label(f"Monitored drones, {graph}")
        lines = ax.plot(run[keep, 0], run[keep, 1:4], color=INK, ls="-.", lw=0.8)
        lines[0].set_label("Leaders")
        ax.set(title=title, xlabel="time [s]", xlim=(0, 450))
    axes[0].set_ylabel("displacement [m]")
    legend_on_top(fig, axes[0], 3)
    save(fig, "responses")


def plot_error_vs_r():
    fig, axes = column_pair(sharey=True)
    sizes = np.arange(1, R_MAX + 1)
    for ax, graph in zip(axes, GRAPHS):
        error = worst_error(graph)
        for k, method in enumerate(METHODS):
            ax.semilogy(sizes, error[:, k], label=LABEL[method], zorder=3 if method == "bpod" else 2,
                        **METHOD_STYLE[method])
        ax.axhline(1.0, color=GREY, ls=":", lw=1)
        ax.text(R_MAX, 1.3, "1%", ha="right", color="#52514e")
        ax.set(title=graph, xlabel="ROM size $r$", xlim=(0, R_MAX + 1))
    axes[0].set_ylabel("worst relative output error [%]")
    legend_on_top(fig, axes[0], 3)
    save(fig, "error_vs_r")


def plot_error_ratio():
    fig, ax = plt.subplots(figsize=(WIDTH / 2, 2.0), layout="constrained")
    ax.tick_params(length=2, pad=1.5)
    sizes = np.arange(1, R_MAX + 1)
    pod, bpod = METHODS.index("pod"), METHODS.index("bpod")
    # the curves are graphs, not methods, so they are black and grey rather than the method colours
    style = {"undirected": dict(color=INK, ls="-", lw=1.0, marker="o", ms=2.2),
             "directed": dict(color=GREY, ls="--", lw=1.0, marker="^", ms=2.6)}
    for graph in GRAPHS:
        error = worst_error(graph)
        ax.semilogy(sizes, error[:, pod] / error[:, bpod], label=graph, **style[graph])
    ax.set(xlabel="ROM size $r$", ylabel="POD error / BPOD error", xlim=(0, R_MAX + 1))
    legend_on_top(fig, ax, 2)
    save(fig, "error_ratio")


def plot_frequency_error():
    summary = np.genfromtxt(DATA / "summary.csv", delimiter=",", skip_header=1, dtype=None, encoding=None)
    omega_c = dict(summary.tolist())["omega_c"]
    fig, axes = column_pair(sharey=True)
    for ax, graph in zip(axes, GRAPHS):
        table = load(RESULTS / graph, "frequency_error", skiprows=1)
        keep = table[:, 0] <= OMEGA_MAX
        for k, method in enumerate(METHODS):
            style = {**METHOD_STYLE[method], "marker": "x" if method == "bpod" else None,
                     "markevery": 12, "ms": 3}
            ax.loglog(table[keep, 0], table[keep, 2 + k * R_MAX + R_FREQ - 1], label=LABEL[method], **style)
        ax.set(xticks=[1e-4, 1e-2, 1e0], yticks=[1e-3, 1e1, 1e5])   # every second decade, so the labels fit
        ax.axvline(omega_c, color=GREY, ls=":", lw=1)
        ax.text(1.2 * omega_c, 1e4, r"$\omega_c$", color="#52514e")
        ax.set(title=f"{graph}, $r$ = {R_FREQ}", xlabel=r"frequency $\omega$ [rad/s]")
    axes[0].set_ylabel("relative frequency-response error")
    legend_on_top(fig, axes[0], 3)
    save(fig, "frequency_error")


def plot_step_rom():
    fig, axes = column_pair(sharey=False)
    for ax, graph in zip(axes, GRAPHS):
        full = load(DATA / graph, "step", skiprows=1)
        rom = load(RESULTS / graph, f"step_r{R_RUN}", skiprows=1)
        keep = full[:, 0] <= STEP_END[graph]
        t = full[keep, 0]
        ax.plot(t, full[keep, 4 + MONITOR], color=INK, lw=2.2, label="Full model")
        for k, method in enumerate(METHODS):
            style = {**METHOD_STYLE[method], "markevery": 700 if graph == "undirected" else 100}
            ax.plot(t, rom[keep, 1 + k * P_MONITORS + MONITOR], label=LABEL[method], **style)
        ax.set(title=f"{graph}, $r$ = {R_RUN}", xlabel="time [s]", xlim=(0, STEP_END[graph]))
    axes[0].set_ylabel(f"monitored drone {MONITOR + 1} [m]")
    legend_on_top(fig, axes[0], 4)
    save(fig, "step_rom")


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    plot_responses()
    with plt.rc_context(SMALL):   # these go in one column of the paper
        plot_layout()
        plot_error_ratio()
        plot_singular_values()
        plot_error_vs_r()
        plot_frequency_error()
        plot_step_rom()
    print(f"Done. Figures are in {FIGURES}")
