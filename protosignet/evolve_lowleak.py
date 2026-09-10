"""Search for a dual-FM motif that decodes dense vs sparse with LOW CROSSTALK.

Why this exists
---------------
The published objective (``evolve_motifs.obj_func_dual_fm``) scores each decoder with a
**difference**::

    obj2_X1 = dense_X1  - sparse_X1 - resting_X1
    obj2_X2 = sparse_X2 - dense_X2  - resting_X2

A difference is indifferent to leak. A candidate reading sparse 0.9 / dense 0.2 scores
exactly as well as one reading sparse 0.7 / dense 0.0, so the optimizer had no reason to
suppress off-regime activation and spent its budget on signal amplitude instead. The
published FMDM shows the consequence (independent runs, converged tolerance):

    X1 dense decoder    rest 0.000  sparse 0.146  dense 0.909   ->  16.0% leak,  6.2x
    X2 sparse decoder   rest 0.000  sparse 0.830  dense 0.093   ->  11.2% leak,  8.9x

which matches the experimental observation that the sparse decoder is only strongly
repressed under high-intensity continuous light.

What changes
------------
The TASK is deliberately identical to the published one (sparse = 1 unit ON every 10,
dense = continuously ON) so that any difference in the result is attributable to the
scoring and the motifs are directly comparable. Only the SCORE changes:

1. **Normalized contrast instead of difference.** Per decoder,
   ``c = (on - off) / (on + off)``, bounded in [-1, 1]. Halving the leak now improves the
   score even when the gap is unchanged, which is exactly what the difference could not see.
2. **Worst channel, not the sum.** ``obj2 = min(c_dense, c_sparse)``. Nailing one decoder
   and letting the other leak earns nothing. Same design as ``evolve_3decoder``.
3. **``off`` is the worse of the cross-regime and resting means**, so a motif cannot hide
   leak by being noisy at rest.
4. **A minimum on-amplitude.** Without it, a dead motif (on = off = 0) scores 0.5 by
   symmetry. Candidates whose on-state never reaches MIN_ON score zero.

Independent simulations
-----------------------
The published objective scores both regimes as consecutive windows of ONE run with sparse
always first. That is gameable: on the fixed-duty task NSGA-II exploited exactly this and
returned a *timer* -- a slow integrator latching on elapsed time rather than a decoder.
Each regime here is therefore its own run from the same initial state after the same rest
period, as in ``evolve_fixed_duty``.

Objective 1 (unchanged): simplicity = fraction of zero parameters.
"""

import numpy as np

from protosignet.evolve_fixed_duty import sim_tight

REST_N = 40  # rest units before stimulus
WIN_N = 41  # stimulus units (matches the published dense window, t = 80..121)
REST = slice(0, REST_N)
WIN = slice(REST_N, REST_N + WIN_N)

MIN_ON = 0.05  # a decoder must actually turn on; below this the candidate scores zero
EPS = 1e-9


def make_input(kind):
    """rest, then a 41-unit window. 'sparse' = 1 on every 10; 'dense' = continuously on."""
    tt = np.arange(0, REST_N + WIN_N, 1.0)
    uu = np.zeros_like(tt)
    if kind == "sparse":
        uu[REST_N : REST_N + WIN_N : 10] = 1.0
    elif kind == "dense":
        uu[REST_N : REST_N + WIN_N] = 1.0
    else:
        raise ValueError(kind)
    return tt, uu


TT_SP, UU_SP = make_input("sparse")
TT_DN, UU_DN = make_input("dense")


def _contrast(on, off):
    """Normalized contrast in [-1, 1]; -1 when the decoder never turns on."""
    if on < MIN_ON:
        return -1.0
    return (on - off) / (on + off + EPS)


def obj_func_lowleak(candidate, n_nodes):
    """Maximize simplicity and worst-channel normalized contrast (i.e. minimize leak).

    Args:
        candidate (1 x N+N+N*N array): parameters for simulating the signet model with N nodes

    Returns:
        obj_scores (1 x 2 array): [simplicity, contrast], both maximized, both in [0, 1].
            A motif with no discrimination scores 0.5 on objective 2; the published FMDM
            scores ~0.86; a leak-free decoder approaches 1.0.
    """
    rescale_01 = lambda x, xmin, xmax: (x - xmin) / (xmax - xmin)
    if len(candidate) != 2 * n_nodes + n_nodes**2:
        raise ValueError("param_space")
    obj1 = np.count_nonzero(candidate == 0)
    obj1 = rescale_01(x=obj1, xmin=0, xmax=len(candidate) - n_nodes)

    candidate = candidate.reshape(n_nodes, -1)
    kr, ku, kX = candidate[:, 0], candidate[:, 1], candidate[:, 2:]
    _, Xsp = sim_tight(TT_SP, UU_SP, kr, ku, kX)
    _, Xdn = sim_tight(TT_DN, UU_DN, kr, ku, kX)

    # Every term in the model carries a (1 - X) factor, so a correct solution stays in
    # [0, 1]. Reject integration failure rather than let the optimizer select on it.
    if max(np.max(np.abs(Xsp)), np.max(np.abs(Xdn))) > 1.01:
        return np.array([obj1, 0.0])

    rest_X1 = max(np.mean(Xsp[0][REST]), np.mean(Xdn[0][REST]))
    rest_X2 = max(np.mean(Xsp[1][REST]), np.mean(Xdn[1][REST]))
    c_dense = _contrast(np.mean(Xdn[0][WIN]), max(np.mean(Xsp[0][WIN]), rest_X1))
    c_sparse = _contrast(np.mean(Xsp[1][WIN]), max(np.mean(Xdn[1][WIN]), rest_X2))

    obj2 = rescale_01(x=min(c_dense, c_sparse), xmin=-1.0, xmax=1.0)
    objectives = np.array([obj1, obj2])
    return np.array([o if 0 <= o <= 1 else 0 for o in objectives])
