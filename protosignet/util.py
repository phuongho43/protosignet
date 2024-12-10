import ast
from pathlib import Path

import numpy as np
import pandas as pd
from natsort import natsorted


def calc_hypervolume2D(pf_obj, ref):
    """Calculate the hypervolume indicator for an optimization problem with 2 objectives.

    Args:
        pf_obj (m x 2 array): m nondominated front individuals by j=2 objectives
        ref (1 x 2 array): reference point for HV calculation.
            ref value 1: the minimum value possible for objective 1
            ref value 2: the minimum value possible for objective 2

    Returns:
        hv (float): hypervolume value
    """
    pf_obj = np.unique(pf_obj, axis=0)  # get unique obj scores and sort by obj 1
    df1 = np.diff([ref[0], *list(pf_obj[:, 0])])  # rectangle widths
    df2 = np.abs(pf_obj[:, 1] - ref[1])  # rectangle heights
    hv = (df1 * df2).sum()
    return hv


def tag_objectives(data_dp):
    """Tag every objective score set with an address (repeat index, generation index, population index).

    Args:
        data_dp (str): absolute path to data directory

    Returns:
        df (DataFrame): reorganized data with columns ["obj1", "obj2", "rep_i", "gen_j", "pop_k"]
    """
    obj_scores = np.empty((0, 5))
    for i, csv_fp in enumerate(natsorted(Path(data_dp).glob("*.csv"))):
        df_rep = pd.read_csv(csv_fp)
        os_rep = df_rep["objective"].values
        for j in range(len(os_rep)):
            os_gen = np.array(ast.literal_eval(os_rep[j]))
            rep_i = i * np.ones(os_gen.shape[0])
            gen_j = j * np.ones(os_gen.shape[0])
            pop_k = np.arange(os_gen.shape[0])
            address = np.column_stack((rep_i, gen_j, pop_k))
            os_ijk = np.concatenate((os_gen, address), axis=1)
            obj_scores = np.vstack((obj_scores, os_ijk))
    df = pd.DataFrame(data=np.array(obj_scores), columns=["obj1", "obj2", "rep_i", "gen_j", "pop_k"])
    return df


def dominates(p_obj, q_obj):
    """Evaluates whether individual p dominates individual q.

    Individual p dominates individual q if p is no worse than q in all objectives and p is
    strictly better than q in at least one objective.

    Args:
        p_obj (1D array-like): array of j objective scores corresponding to individual p
        q_obj (1D array-like): array of j objective scores corresponding to individual q

    Returns:
        True if p dominates q else False
    """
    return np.all(p_obj >= q_obj) and np.any(p_obj > q_obj)


def eval_pareto(objectives):
    pop_idx = range(len(objectives))
    dom_count = [0 for i in pop_idx]
    is_pareto = [0 for i in pop_idx]
    for p in pop_idx:
        for q in pop_idx:
            if dominates(objectives[q], objectives[p]):
                dom_count[p] += 1
        if dom_count[p] == 0:
            is_pareto[p] = 1
    return np.array(is_pareto)


def calc_n_nodes(n_params):
    x = 0
    y = 1
    while y != 0:
        x += 1
        y = x**2 + 2 * x - n_params
        if x > 10:
            raise ValueError("n_nodes")
    return x
