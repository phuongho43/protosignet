import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from protosignet.model import sim_signet
from protosignet.optimizer import NSGAII


def obj_func_dual_fm(candidate, n_nodes):
    """Objective function to optimize for a network motif that decodes dense vs sparse FM inputs.

    Args:
        candidate (1 x N+N+N*N array): parameters for simulating the signet model with N nodes

    Returns:
        obj_scores (1 x 2 array): objective scores
            NOTE: We assume a *maximization* scheme for every objective score.
            Objective 1: Maximize the count of zero parameters (simplicity)
            Objective 2: Maximize ave([X1]) during dense input and ave([X2]) during sparse input (performance)
    """
    rescale_01 = lambda x, xmin, xmax: (x - xmin) / (xmax - xmin)
    if len(candidate) != 2 * n_nodes + n_nodes**2:
        raise ValueError("param_space")
    # Obj1:
    obj1 = np.count_nonzero(candidate == 0)
    obj1 = rescale_01(x=obj1, xmin=0, xmax=len(candidate) - n_nodes)
    # Obj2:
    candidate = candidate.reshape(n_nodes, -1)
    kr = candidate[:, 0]
    ku = candidate[:, 1]
    kX = candidate[:, 2:]
    tt = np.arange(0, 121, 1.0)
    uu = np.zeros_like(tt)
    uu[40:80:10] = 1.0  # sparse input
    uu[80:121:1] = 1.0  # dense input
    tm, Xm = sim_signet(tt, uu, kr, ku, kX)
    X1 = Xm[0]  # dense decoder output
    X2 = Xm[1]  # sparse decoder output
    resting_X1 = np.mean(X1[:40])
    sparse_X1 = np.mean(X1[40:80])
    dense_X1 = np.mean(X1[80:])
    obj2_X1 = dense_X1 - sparse_X1 - resting_X1
    resting_X2 = np.mean(X2[:40])
    sparse_X2 = np.mean(X2[40:80])
    dense_X2 = np.mean(X2[80:])
    obj2_X2 = sparse_X2 - dense_X2 - resting_X2
    obj2 = obj2_X1 + obj2_X2
    obj2 = rescale_01(x=obj2, xmin=-4, xmax=2)
    objectives = [obj1, obj2]
    objectives = np.array([obj if obj >= 0 and obj <= 1 else 0 for obj in objectives])
    return objectives


def main():
    # directory path to save results
    save_dp = Path("/home/phuong/data/protosignet/dual_fm/data")
    if save_dp.exists():
        input("Directory already exists. Overwrite?...Press ENTER to continue or Ctrl-C to abort.")
    save_dp.mkdir(parents=True, exist_ok=True)
    # save a copy of this script alongside the results
    this_fp = Path(__file__).resolve()
    copy_fp = save_dp.parent / "settings.txt"
    shutil.copyfile(this_fp, copy_fp)

    ## SETTINGS ##
    # number of model nodes
    n_nodes = 5
    # population size
    pop_size = 100
    # number of evolutionary generations
    n_gens = 250
    # number of program runs/repeats
    repeats = 5
    # rng seed for numpy
    rng_seed = None
    # parameter space to draw values from for each candidate
    # e.g. for a 3 node system, a list containing 15 lists corresponding to each parameter:
    # kr1, ku1, kX11, kX12, kX13,
    # kr2, ku2, kX21, kX22, kX23,
    # kr3, ku3, kX31, kX32, kX33
    kr = [-10, -1, -0.1, -0.01, 0.01, 0.1, 1, 10]
    ku = [0, 0.01, 0.1, 1, 10]
    kX = [[-10, -1, -0.1, 0.01, 0, 0.01, 0.1, 1, 10]] * n_nodes
    param_space = [kr, ku, *kX] * n_nodes

    # run NSGA-II and save data
    for rep_i in range(repeats):
        optimizer = NSGAII(obj_func=obj_func_dual_fm, param_space=param_space, obj_func_kwargs={"n_nodes": n_nodes}, pop_size=pop_size, rng_seed=rng_seed)
        data = optimizer.evolve(n_gen=n_gens)
        data = pd.DataFrame(data)
        data.to_csv(save_dp / f"{rep_i}.csv", index=False)


if __name__ == "__main__":
    main()
