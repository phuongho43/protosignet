import ast
from pathlib import Path

import numpy as np
import pandas as pd
from natsort import natsorted

CUSTOM_PALETTE = ["#648FFF", "#2ECC71", "#8069EC", "#EA822C", "#D143A4", "#F1C40F", "#34495E"]

CUSTOM_STYLE = {
    "image.cmap": "turbo",
    "figure.figsize": (24, 16),
    "text.color": "#212121",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelpad": 12,
    "axes.labelcolor": "#212121",
    "axes.labelweight": 600,
    "axes.linewidth": 6,
    "axes.edgecolor": "#212121",
    "grid.linewidth": 1,
    "xtick.major.pad": 12,
    "ytick.major.pad": 12,
    "lines.linewidth": 10,
    "axes.labelsize": 72,
    "xtick.labelsize": 56,
    "ytick.labelsize": 56,
    "legend.fontsize": 56,
}


def plot_figure_1d(data_dp, save_dp):
    """Generate scatterplot of obj 1 (simplicity) vs obj 2 (performance) over all runs/repeats.

    Args:
        data_dp (str): absolute path to data directory
        save_dp (str): absolute path to save directory
    """
    obj_scores = np.empty((0, 5))
    for i, csv_fp in enumerate(natsorted(Path(data_dp).glob("*.csv"))):
        df_rep = pd.read_csv(csv_fp)
        os_rep = df_rep["objective"].values
        for j in range(len(os_rep)):
            os_gen = np.array(ast.literal_eval(os_rep[j]))
            os_gen[:, 0] = os_gen[:, 0]
            os_gen[:, 1] = os_gen[:, 1]
            csv_i = i * np.ones(os_gen.shape[0])
            gen_j = j * np.ones(os_gen.shape[0])
            pop_k = np.arange(os_gen.shape[0])
            address = np.column_stack((csv_i, gen_j, pop_k))
            os_ijk = np.concatenate((os_gen, address), axis=1)
            obj_scores = np.vstack((obj_scores, os_ijk))
    df = pd.DataFrame(data=np.array(obj_scores), columns=["obj1", "obj2", "rep_i", "gen_j", "pop_k"])
    print(df.shape)


def main():
    data_dp = Path("/home/phuong/data/protosignet/dual_fm/data/")
    save_dp = Path("/home/phuong/data/protosignet/dual_fm/figs/")
    plot_figure_1d(data_dp, save_dp)


if __name__ == "__main__":
    main()
