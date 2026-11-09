"""Test inputs (never used to build ROMs). Each returns leader displacements u [m], shape (len(t), 3)."""

import numpy as np
from .parameters import M_LEADERS

DT = 0.05                  # time step of the test runs [s]

# step test: all leaders move STEP_AMP along a smooth ramp, then hold
STEP_AMP = 50.0            # shift [m]
STEP_RAMP = 20.0           # ramp time [s]
STEP_SETTLE = 20.0         # run lasts STEP_RAMP + STEP_SETTLE * slowest time constant

# random test: each leader follows a random smooth sum of cosines
RAND_RMS = 20.0            # rms [m]
RAND_NFREQ = 20            # number of frequencies per leader
RAND_RAMP = 20.0           # ramp-in time [s]
RAND_GAIN_RATIO = 0.5      # band limit omega_c: where the gain from leader 1 to the monitors
                           # falls to this fraction of its zero-frequency value
RAND_BAND = (0.05, 1.0)    # frequencies are uniform in this range times omega_c
RAND_SEEDS = (11, 12, 13)


def quintic(s):
    """Smooth 0 -> 1 ramp with zero velocity and acceleration at both ends."""
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def step_input(t):
    """All leaders move STEP_AMP along a quintic ramp over STEP_RAMP, then hold."""
    return STEP_AMP * quintic(t / STEP_RAMP)[:, None] * np.ones(M_LEADERS)


def random_input(t, omega_c, seed):
    """Per leader, a sum of RAND_NFREQ cosines with random frequencies in RAND_BAND * omega_c
    and random phases, scaled to RAND_RMS and ramped in over RAND_RAMP."""
    rng = np.random.default_rng(seed)
    freq = rng.uniform(*RAND_BAND, size=(M_LEADERS, RAND_NFREQ)) * omega_c
    phase = rng.uniform(0, 2 * np.pi, size=(M_LEADERS, RAND_NFREQ))
    amp = RAND_RMS * np.sqrt(2.0 / RAND_NFREQ)    # rms of the sum = RAND_RMS
    cosines = np.cos(t[:, None, None] * freq + phase).sum(-1)
    return amp * quintic(t / RAND_RAMP)[:, None] * cosines
