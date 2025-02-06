import ast
from pathlib import Path

import numpy as np
import pandas as pd
from natsort import natsorted

from protosignet.model import sim_signet


def eval_is_pareto(objectives):
    is_pareto = np.ones(objectives.shape[0], dtype=bool)
    for i, obj in enumerate(objectives):
        if is_pareto[i]:
            is_pareto[is_pareto] = np.any(objectives[is_pareto] > obj, axis=1)
            is_pareto[i] = True
    return is_pareto


def prep_obj_addr(save_csv_fp, data_dp):
    """Reorganize the objective scores and address for each individual:

    [obj_0, obj_1, ..., repeat_index, generation_index, population_index, is_pareto]

    Args:
        save_csv_fp (str): absolute path for saving obj_addr data as csv file
        data_dp (str): absolute path to data directory containing optimization repeat csv files

    Returns:
        df (DataFrame): data with columns [obj_0, obj_1, ..., rep_i, gen_j, pop_k, is_pareto]
    """
    save_csv_fp = Path(save_csv_fp)
    save_csv_fp.parent.mkdir(parents=True, exist_ok=True)
    obj_addr_df = []
    for rep_i, rep_csv_fp in enumerate(natsorted(Path(data_dp).glob("*.csv"))):
        rep_df = pd.read_csv(rep_csv_fp)
        for _, row in rep_df.iterrows():
            gen_j = row["generation"]
            obj_arr = np.array(ast.literal_eval(row["objective"]))
            obj_df = pd.DataFrame(obj_arr, columns=[f"obj_{s}" for s in range(obj_arr.shape[1])])
            obj_df["rep_i"] = np.ones(len(obj_df), dtype=int) * rep_i
            obj_df["gen_j"] = np.ones(len(obj_df), dtype=int) * gen_j
            obj_df["pop_k"] = np.arange(len(obj_df), dtype=int)
            obj_addr_df.append(obj_df)
    obj_addr_df = pd.concat(obj_addr_df)
    obj_cols = [col for col in obj_addr_df if col.startswith("obj")]
    all_objs = obj_addr_df[obj_cols].to_numpy()
    obj_addr_df["is_pareto"] = eval_is_pareto(all_objs)
    obj_addr_df.to_csv(save_csv_fp, index=False)


def calc_n_nodes(n_params):
    x = 0
    y = 1
    while y != 0:
        x += 1
        y = x**2 + 2 * x - n_params
        if x > 10:
            raise ValueError("n_nodes")
    return x


def fetch_indiv(data_dp, address):
    rep_i, gen_j, pop_k = address
    df_rep = pd.read_csv(data_dp / f"{int(rep_i)}.csv")
    pop_rep = df_rep["population"].values
    pop_gen = np.array(ast.literal_eval(pop_rep[int(gen_j)]))
    n_params = pop_gen.shape[1]
    n_nodes = calc_n_nodes(n_params)
    indiv = pop_gen[int(pop_k)].reshape(int(n_nodes), -1)
    return indiv


def sim_fm_motif(save_csv_fp, indiv):
    save_csv_fp = Path(save_csv_fp)
    save_csv_fp.parent.mkdir(parents=True, exist_ok=True)
    kr = indiv[:, 0]
    ku = indiv[:, 1]
    kX = indiv[:, 2:]
    tu = np.arange(0, 121, 1.0)
    uu = np.zeros_like(tu)
    uu[40:80:10] = 1.0  # sparse input
    uu[80:121:1] = 1.0  # dense input
    tm, Xm = sim_signet(tu, uu, kr, ku, kX)
    df_denser = pd.DataFrame({"t": tm, "y": Xm[0], "c": np.zeros(len(tm), dtype=int)})
    df_sparser = pd.DataFrame({"t": tm, "y": Xm[1], "c": np.ones(len(tm), dtype=int)})
    df = pd.concat([df_denser, df_sparser])
    df.to_csv(save_csv_fp, index=False)


def calc_fm_response(save_csv_fp, indiv):
    save_csv_fp = Path(save_csv_fp)
    save_csv_fp.parent.mkdir(parents=True, exist_ok=True)
    kr = indiv[:, 0]
    ku = indiv[:, 1]
    kX = indiv[:, 2:]
    fm_ave_df = []
    tu = np.arange(0, 121, 1)
    periods = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 36, 40, 50, 60, 70, 80, 90, 100, 121]
    for period in periods:
        freq = 1 / period
        uu = np.zeros_like(tu)
        uu[period:121:period] = 1
        tm, Xm = sim_signet(tu, uu, kr, ku, kX)
        y_ave_denser = np.mean(Xm[0])
        y_ave_sparser = np.mean(Xm[1])
        fm_ave_df.append({"c": 0, "t": freq, "y": y_ave_denser})
        fm_ave_df.append({"c": 1, "t": freq, "y": y_ave_sparser})
    pd.DataFrame(fm_ave_df).to_csv(save_csv_fp, index=False)


def main():
    ## Reorganize Optimization Results ##
    expt_dp = Path("/home/phuong/data/phd-project/0--protosignet/0--dual-fm/")
    data_dp = expt_dp / "data"
    results_dp = expt_dp / "results"
    results_dp.mkdir(parents=True, exist_ok=True)
    obj_addr_csv_fp = results_dp / "obj_addr.csv"
    prep_obj_addr(obj_addr_csv_fp, data_dp)

    ## Fetch Pareto Optimal Results ##
    expt_dp = Path("/home/phuong/data/phd-project/0--protosignet/0--dual-fm/")
    results_dp = expt_dp / "results"
    obj_addr_csv_fp = results_dp / "obj_addr.csv"
    obj_addr_df = pd.read_csv(obj_addr_csv_fp)
    pareto_df = obj_addr_df.loc[obj_addr_df["is_pareto"]]
    print("Pareto Optimal Indiv Motifs:")
    print(pareto_df)

    ## Fetch and Sim Some Pareto Motifs ##
    expt_dp = Path("/home/phuong/data/phd-project/0--protosignet/0--dual-fm/")
    data_dp = expt_dp / "data"
    results_dp = expt_dp / "results"
    # High Simplicity & Low Performance
    save_csv_fp = results_dp / "fm_motif_high_low.csv"
    address = [3, 225, 57]
    indiv = fetch_indiv(data_dp, address)
    sim_fm_motif(save_csv_fp, indiv)
    # Low Simplicity & High Performance
    save_csv_fp = results_dp / "fm_motif_low_high.csv"
    address = [1, 217, 56]
    indiv = fetch_indiv(data_dp, address)
    sim_fm_motif(save_csv_fp, indiv)
    # High Simplicity & High Performance
    save_csv_fp = results_dp / "fm_motif_high_high.csv"
    address = [3, 246, 45]
    indiv = fetch_indiv(data_dp, address)
    print(f"Params for Indiv Motif: {address}")
    print(indiv)
    sim_fm_motif(save_csv_fp, indiv)

    ## Calc FM Ave Response for Dense and Sparse Decoders ##
    expt_dp = Path("/home/phuong/data/phd-project/0--protosignet/0--dual-fm/")
    data_dp = expt_dp / "data"
    results_dp = expt_dp / "results"
    save_csv_fp = results_dp / "fm_ave_response.csv"
    address = [3, 246, 45]
    indiv = fetch_indiv(data_dp, address)
    calc_fm_response(save_csv_fp, indiv)

    ## Explore Impact of Modified Reactions ##
    expt_dp = Path("/home/phuong/data/phd-project/0--protosignet/0--dual-fm/")
    data_dp = expt_dp / "data"
    results_dp = expt_dp / "results"
    address = [3, 246, 45]
    kk_ref = fetch_indiv(data_dp, address)
    kk_test1 = kk_ref.copy()
    kk_test1[1, 3] = 0.0
    kk_test2 = kk_ref.copy()
    kk_test2[1, 1] = 10.0
    kk_test3 = kk_ref.copy()
    kk_test3[1, 3] = 0.0
    kk_test3[1, 1] = 10.0
    save_csv_fp = results_dp / "fm_motif_no_x2_self_activ.csv"
    sim_fm_motif(save_csv_fp, kk_test1)
    save_csv_fp = results_dp / "fm_motif_high_x2_induc.csv"
    sim_fm_motif(save_csv_fp, kk_test2)
    save_csv_fp = results_dp / "fm_motif_no_x2_self_activ_high_x2_induc.csv"
    sim_fm_motif(save_csv_fp, kk_test3)


if __name__ == "__main__":
    main()
