"""Search for a motif that decodes THREE temporal patterns at a fixed duty cycle.

Answers R3 #1 ("the system currently supports only two frequencies … which limits its
potential for more complex, multi-target applications"). Our OptoPI screen already
established the negative half rigorously: kinetic tuning of the present two-module motif
reaches three channels only at a separation margin of ~0.38 (versus 1.0 for the cleanly
separable dense channel), so three channels are not achievable by retuning what we have.
This search asks the complementary question — does a motif class exist that *can*?

Three input regimes, all at **exactly 25% duty cycle**, differing only in period:

    fast    period 4   =  1 on /  3 off
    medium  period 20  =  5 on / 15 off
    slow    period 80  = 20 on / 60 off

25% duty is the regime the real system operates in: the manuscript's sparse input is
1 s ON / 3 s OFF, i.e. exactly period 4 at 25% duty, so `fast` here is the existing sparse
input and the search asks what could be layered on top of it.

Every regime therefore delivers the same number of ON time-units over the same window, so
nothing can be won by reading time-averaged dose. A three-channel decoder distinguished by
dose instead would be a thermometer code and near-trivial; this is the version worth
reporting.

Each regime runs as an INDEPENDENT simulation from the same initial state — scoring regimes
as consecutive windows of one run is gameable, and NSGA-II will return a timer instead of a
decoder (see the note in ``evolve_fixed_duty``).

Objective 1: simplicity = fraction of zero parameters.
Objective 2: the WORST of the three channels. X1 must prefer fast, X2 medium, X3 slow, and
the score is set by whichever channel separates least — so a motif that nails two channels
and drops the third scores no better than its weakest link.
"""

import numpy as np

from protosignet.evolve_fixed_duty import sim_tight

REST_N = 40
WIN_N = 400  # 5 cycles of the slowest (period-80) regime
REST = slice(0, REST_N)
WIN = slice(REST_N, REST_N + WIN_N)
PERIODS = {"fast": 4, "medium": 20, "slow": 80}
DUTY = 0.25
# which node is supposed to own which regime
OWNER = {"fast": 0, "medium": 1, "slow": 2}


def make_input(kind):
    """Rest, then a WIN_N-unit window at `kind`'s period and 25% duty."""
    period = PERIODS[kind]
    width = int(round(DUTY * period))
    tt = np.arange(0, REST_N + WIN_N + 1, 1.0)
    uu = np.zeros_like(tt)
    for i in range(REST_N, REST_N + WIN_N):
        uu[i] = 1.0 if (i - REST_N) % period < width else 0.0
    return tt, uu


INPUTS = {k: make_input(k) for k in PERIODS}
_on = {k: INPUTS[k][1][WIN].sum() for k in PERIODS}
assert len(set(_on.values())) == 1 and next(iter(_on.values())) == DUTY * WIN_N, f"regimes are not dose-matched: {_on}"


def obj_func_3decoder(candidate, n_nodes):
    """Maximize simplicity and three-way temporal-pattern separation at matched dose.

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

    means, rest = {}, {}
    for kind, (tt, uu) in INPUTS.items():
        _, Xm = sim_tight(tt, uu, kr, ku, kX)
        if np.max(np.abs(Xm)) > 1.01:  # integration blew up; do not score it
            return np.array([obj1, 0.0])
        means[kind] = [np.mean(Xm[i][WIN]) for i in range(3)]
        rest[kind] = [np.mean(Xm[i][REST]) for i in range(3)]

    # each channel: its own regime, minus the best competing regime, minus its resting level
    scores = []
    for kind, node in OWNER.items():
        own = means[kind][node]
        other = max(means[k][node] for k in PERIODS if k != kind)
        scores.append(own - other - rest[kind][node])
    # the worst channel sets the score -- no credit for two-out-of-three
    obj2 = rescale_01(x=min(scores), xmin=-2, xmax=1)
    objectives = np.array([obj if 0 <= obj <= 1 else 0 for obj in (obj1, obj2)])
    return objectives
