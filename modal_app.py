"""Run the ProtoSigNet NSGA-II dual-FM optimization on Modal cloud CPU.

Modal is a deploy/run tool invoked from your laptop; it is NOT a runtime dependency of the
protosignet package. The container image below pins its own deps, so runs are reproducible and
independent of the local env (and free of the local Python-3.14/joblib issue).

One-time setup
--------------
    uv tool install modal      # or: pip install modal
    modal setup                # authenticate (opens browser)

Run
---
    modal run modal_app.py                              # 5 reps, production settings
    modal run modal_app.py --reps 3 --gens 100 --pop 80 --nodes 5 --cpu 16
    modal run modal_app.py --reps 1 --gens 10 --pop 40  # quick cloud smoke test

Each repeat runs in its own container on `cpu` cores (joblib parallelizes candidate
evaluation across them); the repeats run concurrently, so wall-time ~= one repeat.

Fetch results (one CSV of per-generation data per repeat)
--------------------------------------------------------
    modal volume get protosignet-data /output ./results
"""

import modal

app = modal.App("protosignet")

image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy", "scipy", "joblib", "pandas", "prettytable").add_local_python_source("protosignet")
data_vol = modal.Volume.from_name("protosignet-data", create_if_missing=True)
OUTPUT_DIR = "/data/output"


OBJECTIVES = {
    # published task: sparse (10% duty) vs dense (100% duty) -- frequency confounded with duty
    "dual_fm": ("protosignet.evolve_motifs", "obj_func_dual_fm", "0--dual-fm"),
    # harder task: fast vs slow at a FIXED 50% duty cycle -- dose-matched, frequency only
    "fixed_duty": ("protosignet.evolve_fixed_duty", "obj_func_fixed_duty", "1--fixed-duty"),
    # three channels (fast/medium/slow), all at 50% duty -- answers R3 #1 on scaling
    "three_decoder": ("protosignet.evolve_3decoder", "obj_func_3decoder", "2--three-decoder"),
    # published task, re-scored for LOW CROSSTALK: normalized contrast, worst channel
    "lowleak": ("protosignet.evolve_lowleak", "obj_func_lowleak", "3--lowleak"),
    # same, v2: tracking scored in the objective, MIN_ON 0.25, FMDM seeded into gen 0
    "lowleak2": ("protosignet.evolve_lowleak2", "obj_func_lowleak2", "4--lowleak2"),
}


@app.function(image=image, volumes={"/data": data_vol}, cpu=16, timeout=2 * 60 * 60)
def run_rep(rep_i: int, n_nodes: int = 5, pop: int = 100, gens: int = 250, seed: int | None = None, objective: str = "dual_fm", seed_fmdm: bool = False) -> dict:
    """Run one NSGA-II repeat (parallel candidate eval across the container's cores)."""
    import importlib
    import os
    import time
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from protosignet.optimizer import NSGAII

    mod_name, fn_name, subdir = OBJECTIVES[objective]
    obj_func = getattr(importlib.import_module(mod_name), fn_name)

    # parameter space (matches evolve_motifs.py)
    kr = [-10, -1, -0.1, -0.01, 0.01, 0.1, 1, 10]
    ku = [0, 0.01, 0.1, 1, 10]
    kX = [[-10, -1, -0.1, 0.01, 0, 0.01, 0.1, 1, 10]] * n_nodes
    param_space = [kr, ku, *kX] * n_nodes

    t0 = time.time()
    opt = NSGAII(obj_func=obj_func, param_space=param_space, obj_func_kwargs={"n_nodes": n_nodes}, pop_size=pop, rng_seed=seed)
    if seed_fmdm:
        # Place the published motif (dual-fm run, individual [3, 246, 45]) in generation 0.
        # Without it the v1 lowleak front was dominated by the FMDM at simplicity 0.867 --
        # the search never rediscovered a point it should have beaten -- which made any
        # claim about a complexity threshold unsupportable. Every value below is present in
        # param_space, so the seeded individual is reachable by mutation like any other.
        if n_nodes != 5:
            raise ValueError("seed_fmdm assumes the 5-node published motif")
        fmdm = np.array([[-1, 10, 0, 0, 0, 0, 0], [-0.1, 0.01, -10, 10, 0, 0, 0], [-10, 0, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0, 0], [-10, 0, 0, 0, 0, 0, 0]], dtype=float).flatten()
        for j, v in enumerate(fmdm):
            if v not in list(param_space[j]):
                raise ValueError(f"FMDM value {v} at param {j} is not in param_space")
        opt.population[0] = fmdm
    data = opt.evolve(n_gen=gens)
    df = pd.DataFrame(data)
    out_dir = Path(OUTPUT_DIR) / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{rep_i}.csv", index=False)
    data_vol.commit()

    final_obj = np.array(data[-1]["objective"])  # (pop x 2): [simplicity, performance]
    return {
        "rep": rep_i,
        "gens": gens,
        "pop": pop,
        "nodes": n_nodes,
        "cpus": os.cpu_count(),
        "best_simplicity": round(float(final_obj[:, 0].max()), 4),
        "best_performance": round(float(final_obj[:, 1].max()), 4),
        "seconds": round(time.time() - t0, 1),
    }


@app.local_entrypoint()
def main(reps: int = 5, nodes: int = 5, pop: int = 100, gens: int = 250, cpu: int = 16, objective: str = "dual_fm", seed_fmdm: bool = False):
    if objective not in OBJECTIVES:
        raise SystemExit(f"objective must be one of {list(OBJECTIVES)}")
    subdir = OBJECTIVES[objective][2]
    print(f"Launching {reps} NSGA-II repeat(s) on '{objective}': " f"nodes={nodes} pop={pop} gens={gens} ({cpu} cpu/container)")
    args = [(i, nodes, pop, gens, None, objective, seed_fmdm) for i in range(reps)]
    results = list(run_rep.starmap(args))
    for r in sorted(results, key=lambda x: x["rep"]):
        print(f"  rep {r['rep']}: perf={r['best_performance']} simp={r['best_simplicity']} " f"in {r['seconds']}s on {r['cpus']} cpus")
    print(f"Fetch:  modal volume get protosignet-data /output/{subdir} ./results")
