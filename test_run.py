"""Scaled-down NSGA-II replication test — confirm the ProtoSigNet dual-FM optimization runs
and converges, and measure per-generation time to extrapolate the full-run cost.

Full production settings (evolve_motifs.py): pop_size=100, n_gens=250, repeats=5 (~1 day).
Here: small pop / few gens / 1 repeat, timed.

    uv run python test_run.py
"""

import os
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

from protosignet.evolve_motifs import obj_func_dual_fm
from protosignet.optimizer import NSGAII

OUT = Path("/home/phuong/projects/csc-revisions-2026/data/0--protosignet/1--replication-test")
OUT.mkdir(parents=True, exist_ok=True)

N_NODES = 5
POP = 40  # vs 100 production
GENS = 25  # vs 250 production
SEED = 42

kr = [-10, -1, -0.1, -0.01, 0.01, 0.1, 1, 10]
ku = [0, 0.01, 0.1, 1, 10]
kX = [[-10, -1, -0.1, 0.01, 0, 0.01, 0.1, 1, 10]] * N_NODES
param_space = [kr, ku, *kX] * N_NODES

print(f"CPUs available: {os.cpu_count()}  |  pop={POP} gens={GENS} nodes={N_NODES}")
opt = NSGAII(obj_func=obj_func_dual_fm, param_space=param_space, obj_func_kwargs={"n_nodes": N_NODES}, pop_size=POP, rng_seed=SEED)


# joblib+loky is broken on this venv's Python 3.14; run objective SEQUENTIALLY for the test
def _eval_seq(self, population):
    return np.array([self.obj_func(indiv, **self.obj_func_kwargs) for indiv in population])


opt.eval_objective = types.MethodType(_eval_seq, opt)
print("(running SEQUENTIALLY — single core — to bypass the joblib/py3.14 issue)")
t0 = time.time()
data = opt.evolve(n_gen=GENS)
dt = time.time() - t0

df = pd.DataFrame(data)
df.to_csv(OUT / "test_rep0.csv", index=False)

# convergence: best performance (obj2) per generation
best_obj2 = [max(o[1] for o in gen["objective"]) for gen in data]
best_obj1 = [max(o[0] for o in gen["objective"]) for gen in data]
print("\n=== RESULT ===")
print(f"{GENS} gens x pop {POP} in {dt:.1f}s = {dt / GENS:.2f} s/gen  ({dt / GENS / POP * 1000:.1f} ms/candidate)")
per_gen_full = (dt / GENS) * (100 / POP)  # scale to pop=100, single core
full_1rep_1core = per_gen_full * 250
full_5rep_1core = full_1rep_1core * 5
ncpu = os.cpu_count()
par = 0.7 * ncpu  # ~70% parallel efficiency estimate
print("Extrapolated production run (100pop x 250gen):")
print(f"  SINGLE CORE (as measured):  1 rep {full_1rep_1core / 3600:.1f} h | 5 reps {full_5rep_1core / 3600:.1f} h")
print(f"  PARALLEL ~{par:.0f} eff-cores (once joblib fixed): 1 rep {full_1rep_1core / 3600 / par:.1f} h | 5 reps {full_5rep_1core / 3600 / par:.1f} h")
print(f"best performance (obj2) per gen: {[round(b, 3) for b in best_obj2]}")
print(f"best simplicity  (obj1) per gen: {[round(b, 3) for b in best_obj1]}")
print(f"obj2 improved {best_obj2[0]:.3f} -> {best_obj2[-1]:.3f}")
print("saved", OUT / "test_rep0.csv")
