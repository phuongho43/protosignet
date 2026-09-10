"""Smoke test: confirm joblib/loky parallel eval works (run after pinning Python 3.12)."""

import os
import time

from protosignet.evolve_motifs import obj_func_dual_fm
from protosignet.optimizer import NSGAII

n = 5
kr = [-10, -1, -0.1, -0.01, 0.01, 0.1, 1, 10]
ku = [0, 0.01, 0.1, 1, 10]
kX = [[-10, -1, -0.1, 0.01, 0, 0.01, 0.1, 1, 10]] * n
ps = [kr, ku, *kX] * n

opt = NSGAII(obj_func=obj_func_dual_fm, param_space=ps, obj_func_kwargs={"n_nodes": n}, pop_size=48, rng_seed=1)
t = time.time()
opt.evolve(n_gen=3)  # uses the real parallel eval_objective (joblib/loky)
print(f"\nOK: joblib parallel ran 3 gens (pop 48) in {time.time() - t:.1f}s on {os.cpu_count()} cpus")
