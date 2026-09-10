"""Frequency decoding that must hold at MORE THAN ONE duty cycle.

``evolve_fixed_duty`` tests fast-vs-slow at a single 50% duty cycle. A motif can pass that
by exploiting something specific to 50% duty and still not generalise -- and "responds to
temporal pattern independently of dose" is the property the reviewers actually care about.

Here the same fast/slow discrimination must hold at **25% and 75% duty**. Within each duty
the two regimes are exactly dose-matched; across duties the dose differs 3-fold. A motif
that reads dose scores zero because the within-duty contrast is dose-neutral; a motif that
reads duty-specific timing scores poorly because it must work at both.

Four independent simulations per candidate (2 duties x 2 periods), each from the same
initial state after an identical rest period -- independent runs, for the reason given in
``evolve_fixed_duty`` (consecutive windows let NSGA-II find a timer instead).

Pulse width = duty x period must be a whole number of time units, since ``sim_signet``
holds the input constant over each unit step:

    25% duty:  fast = period 4  (1 on / 3 off)    slow = period 40 (10 on / 30 off)
    75% duty:  fast = period 4  (3 on / 1 off)    slow = period 40 (30 on / 10 off)
"""

import numpy as np

from protosignet.evolve_fixed_duty import sim_tight

REST_N = 40
WIN_N = 200  # long enough for 5 cycles of the slow (period-40) regime
REST = slice(0, REST_N)
WIN = slice(REST_N, REST_N + WIN_N)
DUTIES = (0.25, 0.75)
PERIODS = {"fast": 4, "slow": 40}


def make_input(duty, kind):
    """Rest, then a WIN_N-unit window at the given duty cycle and period."""
    period = PERIODS[kind]
    width = int(round(duty * period))
    tt = np.arange(0, REST_N + WIN_N + 1, 1.0)
    uu = np.zeros_like(tt)
    for i in range(REST_N, REST_N + WIN_N):
        uu[i] = 1.0 if (i - REST_N) % period < width else 0.0
    return tt, uu


INPUTS = {(d, k): make_input(d, k) for d in DUTIES for k in PERIODS}
for _d in DUTIES:  # fast and slow are dose-matched within each duty
    _f = INPUTS[(_d, "fast")][1][WIN].sum()
    _s = INPUTS[(_d, "slow")][1][WIN].sum()
    assert _f == _s == _d * WIN_N, f"duty {_d} not dose-matched: {_f} vs {_s}"


def obj_func_multiduty(candidate, n_nodes):
    """Maximize simplicity and fast/slow discrimination held at BOTH duty cycles.

    Args:
        candidate (1 x N+N+N*N array): parameters for simulating the signet model with N nodes

    Returns:
        obj_scores (1 x 2 array): [simplicity, performance], both maximized, both in [0, 1]
    """

    def rescale_01(x, xmin, xmax):
        return (x - xmin) / (xmax - xmin)

    if len(candidate) != 2 * n_nodes + n_nodes**2:
        raise ValueError("param_space")
    obj1 = np.count_nonzero(candidate == 0)
    obj1 = rescale_01(x=obj1, xmin=0, xmax=len(candidate) - n_nodes)

    candidate = candidate.reshape(n_nodes, -1)
    kr, ku, kX = candidate[:, 0], candidate[:, 1], candidate[:, 2:]
    per_duty = []
    for duty in DUTIES:
        X = {}
        for kind in PERIODS:
            tt, uu = INPUTS[(duty, kind)]
            _, Xm = sim_tight(tt, uu, kr, ku, kX)
            if np.max(np.abs(Xm)) > 1.01:  # integration blew up -- do not score it
                return np.array([obj1, 0.0])
            X[kind] = Xm
        rest_X1 = np.mean(X["fast"][0][REST])
        rest_X2 = np.mean(X["fast"][1][REST])
        d1 = np.mean(X["fast"][0][WIN]) - np.mean(X["slow"][0][WIN]) - rest_X1
        d2 = np.mean(X["slow"][1][WIN]) - np.mean(X["fast"][1][WIN]) - rest_X2
        per_duty.append(d1 + d2)
    # the WORST duty cycle sets the score, so a motif must work at both, not average out
    obj2 = rescale_01(x=min(per_duty), xmin=-4, xmax=2)
    objectives = np.array([obj if 0 <= obj <= 1 else 0 for obj in (obj1, obj2)])
    return objectives
