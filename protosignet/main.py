import os
import shutil

import numpy as np
import pandas as pd

from protosignet.model import NSGAII, sim_signet
from protosignet.utils import makedirs, rescale_01

# directory path to save results
SAVE_DIR = "/home/phuong/data/protosignet/dual_fm"
# number of model nodes
N_NODES = 3
# population size
POP_SIZE = 100
# number of evolutionary generations
N_GENS = 250
# number of program runs/repeats
RUNS = 5
# rng seed for numpy
RNG_SEED = None
# parameter space to draw values from for each candidate
# e.g. for a 3 node system, a list containing 15 lists corresponding to each parameter:
# kr1, ku1, kX11, kX12, kX13,
# kr2, ku2, kX21, kX22, kX23,
# kr3, ku3, kX31, kX32, kX33
kr = [-10, -1, -0.1, -0.01, 0.01, 0.1, 1, 10]
ku = [0, 0.01, 0.1, 1, 10]
kX = [[-10, -1, -0.1, 0.01, 0, 0.01, 0.1, 1, 10]] * N_NODES
PARAM_SPACE = [kr, ku, *kX] * N_NODES


def obj_func_dual_fm(candidate):
    """Objective function to optimize for a network motif that decodes dense vs sparse FM inputs.

    Args:
        candidate (1 x N+N+N*N array): parameters for simulating the signet model with N nodes

    Returns:
        obj_scores (1 x 2 array): objective scores
            NOTE: We assume a *maximization* scheme for every objective score.
            Objective 1: Maximize the count of zero parameters (simplicity)
            Objective 2: Maximize ave([X1]) during dense input and ave([X2]) during sparse input (performance)
    """
    if len(candidate) != 2 * N_NODES + N_NODES**2:
        raise ValueError("PARAM_SPACE")
    # Obj1:
    obj1 = np.count_nonzero(candidate == 0)
    obj1 = rescale_01(obj1, xmin=0, xmax=len(candidate))
    # Obj2:
    candidate = candidate.reshape(N_NODES, -1)
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
    obj2 = rescale_01(obj2, xmin=-4, xmax=2)
    return np.array([obj1, obj2])


def main():
    if os.path.exists(SAVE_DIR):
        input("Directory already exists. Overwrite?...Press ENTER to continue, Ctrl-C to abort.")
    makedirs(SAVE_DIR)
    shutil.copyfile(__file__, os.path.join(SAVE_DIR, "settings.txt"))

    for run_i in range(RUNS):
        optimizer = NSGAII(obj_func=obj_func_dual_fm, param_space=PARAM_SPACE, pop_size=POP_SIZE, rng_seed=RNG_SEED)
        data = optimizer.evolve(n_gen=N_GENS)
        data = pd.DataFrame(data)
        data.to_csv(os.path.join(SAVE_DIR, f"{run_i}.csv"), index=False)


if __name__ == "__main__":
    main()
