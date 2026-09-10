"""Low-crosstalk dual-FM search, v2: tracking in the objective, real amplitude, FMDM seeded.

v1 (``evolve_lowleak``) established that the published FMDM's ~10-18% leak is a consequence
of scoring each decoder with a *difference*, and found a vetted motif at 0.2% leak. Three
weaknesses of that run are fixed here:

1. **Tracking is now scored, not only diagnosed.** v1 scored each regime in its own run,
   which removes the timer exploit but never asks the motif to *re-select* when regimes
   follow one another. Five of eleven v1 front points duly failed the ORDER or TRACKING
   diagnostic. A third simulation presents sparse/dense/sparse/dense and its contrast enters
   the score, so a latch cannot reach the front in the first place.

2. **MIN_ON raised 0.05 -> 0.25.** Two v1 front points passed every diagnostic with an
   on-state amplitude of 0.090 - discriminating, but producing almost no output. 0.25 is
   still well below the FMDM's 0.815 and rejects only the degenerate cases.

3. **The published FMDM is seeded into the initial population** (see ``modal_app.py``,
   ``seed_fmdm``). v1's front was dominated by the FMDM at simplicity 0.867 - the search
   failed to rediscover a point it should have beaten - so no claim about a complexity
   threshold was supportable. Seeding removes that failure mode: the front is then
   guaranteed to be at least as good as the published motif everywhere.

Objective 1 (unchanged): simplicity = fraction of zero parameters.
Objective 2: min over {X1 isolation, X2 isolation, X1 tracking, X2 tracking} of the
normalized contrast (on - off) / (on + off).
"""

import numpy as np

from protosignet.evolve_fixed_duty import sim_tight

REST_N, WIN_N = 40, 41
REST = slice(0, REST_N)
WIN = slice(REST_N, REST_N + WIN_N)

MIN_ON = 0.25  # v1 used 0.05, which admitted motifs with ~0.09 output
EPS = 1e-9


def pulse(kind, n=WIN_N):
    u = np.zeros(n)
    if kind == "sparse":
        u[::10] = 1.0
    elif kind == "dense":
        u[:] = 1.0
    elif kind != "rest":
        raise ValueError(kind)
    return u


def build(seq):
    uu = np.concatenate([np.zeros(REST_N)] + [pulse(k) for k in seq])
    return np.arange(0, len(uu), 1.0), uu


TT_SP, UU_SP = build(["sparse"])
TT_DN, UU_DN = build(["dense"])
TRK_SEQ = ["sparse", "dense", "sparse", "dense"]
TT_TRK, UU_TRK = build(TRK_SEQ)
TRK_WIN = [slice(REST_N + i * WIN_N, REST_N + (i + 1) * WIN_N) for i in range(len(TRK_SEQ))]


def _contrast(on, off):
    """Normalized contrast in [-1, 1]; -1 if the decoder never reaches MIN_ON."""
    if on < MIN_ON:
        return -1.0
    return (on - off) / (on + off + EPS)


def obj_func_lowleak2(candidate, n_nodes):
    """Maximize simplicity and worst-case normalized contrast across isolation + tracking.

    Args:
        candidate (1 x N+N+N*N array): parameters for simulating the signet model with N nodes

    Returns:
        obj_scores (1 x 2 array): [simplicity, contrast], both maximized, both in [0, 1].
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
    _, Xtk = sim_tight(TT_TRK, UU_TRK, kr, ku, kX)

    # Every term carries a (1 - X) factor, so a correct solution stays in [0, 1]. Reject
    # integration failure rather than let the optimizer select on it.
    if max(np.max(np.abs(Xsp)), np.max(np.abs(Xdn)), np.max(np.abs(Xtk))) > 1.01:
        return np.array([obj1, 0.0])

    # isolation: each regime alone, from the same rest state
    rest1 = max(np.mean(Xsp[0][REST]), np.mean(Xdn[0][REST]))
    rest2 = max(np.mean(Xsp[1][REST]), np.mean(Xdn[1][REST]))
    c1_iso = _contrast(np.mean(Xdn[0][WIN]), max(np.mean(Xsp[0][WIN]), rest1))
    c2_iso = _contrast(np.mean(Xsp[1][WIN]), max(np.mean(Xdn[1][WIN]), rest2))

    # tracking: the motif must re-select on every alternation, not latch once
    sp_w = [w for k, w in zip(TRK_SEQ, TRK_WIN, strict=False) if k == "sparse"]
    dn_w = [w for k, w in zip(TRK_SEQ, TRK_WIN, strict=False) if k == "dense"]
    c1_trk = _contrast(min(np.mean(Xtk[0][w]) for w in dn_w), max(np.mean(Xtk[0][w]) for w in sp_w))
    c2_trk = _contrast(min(np.mean(Xtk[1][w]) for w in sp_w), max(np.mean(Xtk[1][w]) for w in dn_w))

    obj2 = rescale_01(x=min(c1_iso, c2_iso, c1_trk, c2_trk), xmin=-1.0, xmax=1.0)
    objectives = np.array([obj1, obj2])
    return np.array([o if 0 <= o <= 1 else 0 for o in objectives])
