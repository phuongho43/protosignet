"""Search for a motif that decodes input FREQUENCY at a FIXED DUTY CYCLE.

Why this exists
---------------
The published dual-FM objective (``evolve_motifs.obj_func_dual_fm``) contrasts a "sparse"
input of 1-unit pulses every 10 units (10% duty) against a "dense" input that is
continuously on (100% duty). Frequency and duty cycle are therefore fully confounded by
construction, and the optimizer was never asked to separate them -- which is exactly what
the Exp-1 duty x frequency grid found experimentally: the decoders respond to duty cycle
(dose-rate), not to frequency at matched duty.

This objective removes the confound. Both regimes deliver **exactly the same number of ON
time-units over the same window** (50% duty, 50 ON units per 100-unit window) and differ
only in timescale:

    run A:  rest 0-40, then t = 40-140 at period 2  = 1 on / 1 off    (50% duty)
    run B:  rest 0-40, then t = 40-140 at period 20 = 10 on / 10 off  (50% duty)

The two runs are INDEPENDENT simulations from the same initial state -- see the note by
``REST_N`` below for why consecutive windows of a single simulation do not work.

A node whose output is a linear low-pass filter of the input has the same mean in both
runs, so any solution must use nonlinearity plus memory. That makes this a strictly harder
problem than the published one -- which is the point.

Objective 1 (unchanged): simplicity = fraction of zero parameters.
Objective 2: X1 high under FAST and low under SLOW/rest; X2 high under SLOW and low under
FAST/rest.
"""

import numpy as np
from scipy.integrate import ode


def sim_tight(tt, uu, kr, ku, kX, rtol=1e-8, atol=1e-8, max_step=0.05):
    """``model.sim_signet`` integrated to convergence.

    sim_signet uses rtol = atol = 1e-5 and max_step = 0.1. Those are not converged for
    stiff parameter sets: every term in the model carries a (1 - X) factor so a correct
    solution stays in [0, 1], yet the loose solve silently returns states as large as 13
    for some candidates -- and an optimizer will happily exploit that. These settings keep
    max(X) = 1.0000 and agree to <0.5% with lsoda and dopri5 at 1e-10, at ~2.1x the cost.
    """
    kr = np.asarray(kr, float)
    ku = np.asarray(ku, float)
    kX = np.asarray(kX, float)
    X0 = np.where(kr < 0, 0, 1).astype(float)
    ku = np.where(kr * ku > 0, -ku, ku)

    def model(t, X):
        Xr = np.where(kr < 0, X, 1 - X)
        Xu = np.where(ku < 0, X, 1 - X)
        Xy = np.where(kX.T < 0, X, 1 - X).T
        return kr * Xr + ku * Xu * u + np.sum(kX * Xy * X, axis=1)

    solver = ode(model)
    solver.set_integrator("vode", method="bdf", rtol=rtol, atol=atol, max_step=max_step)
    solver.set_initial_value(X0)
    sol_X = [X0]
    for i in range(1, len(tt)):
        u = uu[i]
        solver.integrate(tt[i])
        sol_X.append(solver.y.copy())
    return np.array(tt), np.array(sol_X).T


sim_signet = sim_tight

# Each regime is simulated in its OWN run, from the same initial state, preceded by the
# same rest period. Scoring the two regimes as consecutive windows of ONE simulation is
# gameable: NSGA-II then finds a *timer* -- a slow integrator that latches with elapsed
# time and gates the other node -- which scores well only because the regime order is
# fixed. Verified on such a solution: it reverses cleanly when the regime order is swapped
# (the second window is always the "slow-like" one, whichever regime it actually is) and
# shows almost no discrimination when each regime is presented alone. Independent runs
# remove the exploit, so any remaining difference is genuine frequency discrimination at
# matched dose.
REST_N = 40  # rest units before the stimulus
WIN_N = 100  # stimulus units per regime (50 ON units either way = 50% duty)
REST = slice(0, REST_N)
WIN = slice(REST_N, REST_N + WIN_N)


def make_input(kind):
    """One regime: rest, then a 100-unit window at exactly 50% duty.

    kind='fast' -> period 2  (1 on / 1 off);  kind='slow' -> period 20 (10 on / 10 off).
    """
    tt = np.arange(0, REST_N + WIN_N + 1, 1.0)
    uu = np.zeros_like(tt)
    for i in range(REST_N, REST_N + WIN_N):
        j = i - REST_N
        uu[i] = (1.0 if j % 2 == 0 else 0.0) if kind == "fast" else (1.0 if j % 20 < 10 else 0.0)
    return tt, uu


TT_FAST, UU_FAST = make_input("fast")
TT_SLOW, UU_SLOW = make_input("slow")
assert UU_FAST[WIN].sum() == UU_SLOW[WIN].sum() == WIN_N / 2, "regimes are not dose-matched"


def obj_func_fixed_duty(candidate, n_nodes):
    """Maximize simplicity and fixed-duty frequency discrimination.

    Args:
        candidate (1 x N+N+N*N array): parameters for simulating the signet model with N nodes

    Returns:
        obj_scores (1 x 2 array): [simplicity, performance], both maximized, both in [0, 1]
    """

    def rescale_01(x, xmin, xmax):
        return (x - xmin) / (xmax - xmin)

    if len(candidate) != 2 * n_nodes + n_nodes**2:
        raise ValueError("param_space")
    # Obj1: simplicity
    obj1 = np.count_nonzero(candidate == 0)
    obj1 = rescale_01(x=obj1, xmin=0, xmax=len(candidate) - n_nodes)
    # Obj2: fixed-duty frequency discrimination, from two INDEPENDENT simulations
    candidate = candidate.reshape(n_nodes, -1)
    kr = candidate[:, 0]
    ku = candidate[:, 1]
    kX = candidate[:, 2:]
    _, Xf = sim_signet(TT_FAST, UU_FAST, kr, ku, kX)
    _, Xs = sim_signet(TT_SLOW, UU_SLOW, kr, ku, kX)
    # Boundedness guard. Every term in the model carries a (1 - X) factor, so a correctly
    # integrated state stays in [0, 1]. sim_signet's default tolerances (rtol=atol=1e-5,
    # max_step=0.1) are not converged for stiff parameter sets and can silently return
    # states far outside that range (observed max(X) = 13.3), which inflates the score.
    # Reject such candidates rather than let the optimizer exploit integration failure.
    if max(np.max(np.abs(Xf)), np.max(np.abs(Xs))) > 1.01:
        return np.array([obj1, 0.0])
    rest_X1 = np.mean(Xf[0][REST])
    rest_X2 = np.mean(Xf[1][REST])
    obj2_X1 = np.mean(Xf[0][WIN]) - np.mean(Xs[0][WIN]) - rest_X1
    obj2_X2 = np.mean(Xs[1][WIN]) - np.mean(Xf[1][WIN]) - rest_X2
    obj2 = rescale_01(x=obj2_X1 + obj2_X2, xmin=-4, xmax=2)
    objectives = np.array([obj if 0 <= obj <= 1 else 0 for obj in (obj1, obj2)])
    return objectives
