"""Analyze the fresh Modal production run: Pareto front + recover the high-perf/high-simplicity
motif programmatically (not the original run's hardcoded addresses) and confirm it FM-decodes.

    uv run python analyze_prod.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from protosignet.analyze_motifs import fetch_indiv, prep_obj_addr
from protosignet.model import sim_signet

BASE = Path("/home/phuong/projects/csc-revisions-2026/data/0--protosignet")
DATA = BASE / "prod-results"
OUT = BASE / "0--dual-fm" / "results"
OUT.mkdir(parents=True, exist_ok=True)

prep_obj_addr(OUT / "obj_addr.csv", DATA)
df = pd.read_csv(OUT / "obj_addr.csv")
par = df[df.is_pareto]
uniq = par.drop_duplicates(subset=["obj_0", "obj_1"]).sort_values("obj_1")
print(f"Total Pareto-optimal individuals: {int(df.is_pareto.sum())}  (unique (simplicity,perf) pairs: {len(uniq)})")
print(f"  simplicity (obj_0) range: {uniq.obj_0.min():.3f} - {uniq.obj_0.max():.3f}")
print(f"  performance (obj_1) range: {uniq.obj_1.min():.3f} - {uniq.obj_1.max():.3f}")
print("\nPareto front (unique points, simplicity / performance):")
for _, r in uniq.iterrows():
    print(f"  simp={r.obj_0:.3f}  perf={r.obj_1:.3f}   (rep {int(r.rep_i)}, gen {int(r.gen_j)}, pop {int(r.pop_k)})")

# high-perf/high-simplicity corner = max (obj_0 + obj_1) among Pareto
par = par.assign(score=par.obj_0 + par.obj_1).sort_values("score", ascending=False)
best = par.iloc[0]
addr = [int(best.rep_i), int(best.gen_j), int(best.pop_k)]
indiv = fetch_indiv(DATA, addr)
print(f"\nHigh-performance/high-simplicity motif @ rep{addr[0]} gen{addr[1]} pop{addr[2]}: " f"simplicity={best.obj_0:.3f}, performance={best.obj_1:.3f}")
np.set_printoptions(precision=3, suppress=True)
print("params [kr, ku, kX...] per node:\n", indiv)

# confirm FM decoding: dense (X1) high under dense input; sparse (X2) high under sparse input
kr, ku, kX = indiv[:, 0], indiv[:, 1], indiv[:, 2:]
tu = np.arange(0, 121, 1.0)
uu = np.zeros_like(tu)
uu[40:80:10] = 1.0  # sparse regime
uu[80:121:1] = 1.0  # dense regime
_, Xm = sim_signet(tu, uu, kr, ku, kX)
print("\nRegime-mean output (rest / sparse-input / dense-input):")
for name, X in [("Dense decoder  X1", Xm[0]), ("Sparse decoder X2", Xm[1])]:
    print(f"  {name}: {np.mean(X[:40]):.3f} / {np.mean(X[40:80]):.3f} / {np.mean(X[80:]):.3f}")
print("\nDecoding check: X1 should peak at DENSE input, X2 at SPARSE input.")
