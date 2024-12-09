import ast
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from protosignet.model import sim_signet
from protosignet.util import calc_n_nodes, eval_pareto, tag_objectives

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
    "legend.fontsize": 48,
}


def plot_figure_1d(data_dp, fig_fp):
    """Generate scatterplot of obj 1 (simplicity) vs obj 2 (performance) over all runs/repeats.

    Args:
        data_dp (str): absolute path to data directory
        fig_fp (str): absolute path for saving generated figure
    """
    df = tag_objectives(data_dp)
    df_gen_001 = df.loc[df["gen_j"] == 0]
    df_gen_010 = df.loc[df["gen_j"] == 9]
    df_gen_100 = df.loc[df["gen_j"] == 99]
    df_top = df.iloc[df.groupby("obj1")["obj2"].idxmax().values].copy()
    df_top["is_pareto"] = eval_pareto(df_top[["obj1", "obj2"]].to_numpy())
    df_pareto = df_top.loc[df_top["is_pareto"] == 1]
    print(df_pareto)
    with plt.style.context(("seaborn-v0_8-whitegrid", CUSTOM_STYLE)):
        fig, ax = plt.subplots(figsize=(24, 20))
        sns.scatterplot(data=df_gen_001, x="obj1", y="obj2", edgecolor="#212121", facecolor="#2ECC71", alpha=0.8, linewidth=2, s=600)
        sns.scatterplot(data=df_gen_010, x="obj1", y="obj2", edgecolor="#212121", facecolor="#F1C40F", alpha=0.8, linewidth=2, s=600)
        sns.scatterplot(data=df_gen_100, x="obj1", y="obj2", edgecolor="#212121", facecolor="#EA822C", alpha=0.8, linewidth=2, s=600)
        sns.scatterplot(data=df_pareto, x="obj1", y="obj2", edgecolor="#212121", facecolor="#D143A4", alpha=1.0, linewidth=2, s=600)
        handles = [
            mpl.lines.Line2D([], [], color="#2ECC71", marker="o", markersize=8, linewidth=0),
            mpl.lines.Line2D([], [], color="#F1C40F", marker="o", markersize=8, linewidth=0),
            mpl.lines.Line2D([], [], color="#EA822C", marker="o", markersize=8, linewidth=0),
            mpl.lines.Line2D([], [], color="#D143A4", marker="o", markersize=8, linewidth=0),
        ]
        group_labels = ["Gen 1", "Gen 10", "Gen 100", "Best (Pareto)"]
        ax.legend(
            handles,
            group_labels,
            loc="best",
            markerscale=4,
            frameon=True,
            shadow=False,
            handletextpad=0.4,
            borderpad=0.2,
            labelspacing=0.2,
            handlelength=1,
        )
        ax.set_xlabel("Simplicity")
        ax.set_ylabel("Performance")
        ax.xaxis.set_ticks(np.arange(0, 1.1, 0.2))
        ax.set_xlim(-0.1, 1.1)
        ax.yaxis.set_ticks(np.arange(0, 1.1, 0.2))
        ax.set_ylim(-0.1, 1.1)
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(fig_fp, pad_inches=0.3, dpi=200, bbox_inches="tight", transparent=False)
    plt.close("all")


def plot_figure_1e(address, data_dp, fig_fp):
    """Simulate dynamics for a specified motif.

    Args:
        address (list): [rep_i, gen_j, pop_k]
        data_dp (str): absolute path to data directory
        fig_fp (str): absolute path for saving generated figure
    """
    rep_i, gen_j, pop_k = address
    df_rep = pd.read_csv(data_dp / f"{int(rep_i)}.csv")
    pop_rep = df_rep["population"].values
    pop_gen = np.array(ast.literal_eval(pop_rep[int(gen_j)]))
    n_params = pop_gen.shape[1]
    n_nodes = calc_n_nodes(n_params)
    indiv = pop_gen[int(pop_k)].reshape(int(n_nodes), -1)
    print(indiv)
    kr = indiv[:, 0]
    ku = indiv[:, 1]
    kX = indiv[:, 2:]
    tu = np.arange(0, 121, 1.0)
    uu = np.zeros_like(tu)
    uu[40:80:10] = 1.0  # sparse input
    uu[80:121:1] = 1.0  # dense input
    tm, Xm = sim_signet(tu, uu, kr, ku, kX)
    X1_df = pd.DataFrame({"t": tm, "y": Xm[0], "h": np.ones_like(tm) * 0})
    X2_df = pd.DataFrame({"t": tm, "y": Xm[1], "h": np.ones_like(tm) * 1})
    Xm_df = pd.concat([X1_df, X2_df], ignore_index=True)
    with plt.style.context(("seaborn-v0_8-whitegrid", CUSTOM_STYLE)):
        fig, ax = plt.subplots(figsize=(24, 20))
        sns.lineplot(data=Xm_df, x="t", y="y", hue="h", ax=ax, palette=["#8069EC", "#EA822C"], zorder=2.2)
        ymin, ymax = ax.get_ylim()
        for t in tu[uu > 0]:
            ax.axvspan(t, t + 1, color="#648FFF", alpha=0.5, linewidth=0, zorder=2.1)
        ax.set_ylim(ymin, ymax)
        handles = [
            mpl.lines.Line2D([], [], color="#648FFF", linewidth=16, alpha=0.5),
            mpl.lines.Line2D([], [], color="#8069EC", linewidth=16),
            mpl.lines.Line2D([], [], color="#EA822C", linewidth=16),
        ]
        group_labels = ["Input", "Dense Decoder", "Sparse Decoder"]
        ax.legend(
            handles,
            group_labels,
            loc="best",
            markerscale=4,
            frameon=True,
            shadow=False,
            framealpha=1.0,
            handletextpad=0.4,
            borderpad=0.2,
            labelspacing=0.2,
            handlelength=1,
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("AU")
        ax.locator_params(axis="x", nbins=10)
        ax.locator_params(axis="y", nbins=10)
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(fig_fp, pad_inches=0.3, dpi=200, bbox_inches="tight", transparent=False)
    plt.close("all")


def main():
    data_dp = Path("/home/phuong/data/protosignet/dual_fm/data/")
    save_dp = Path("/home/phuong/data/protosignet/dual_fm/figs/")
    save_dp.mkdir(parents=True, exist_ok=True)

    # fig_fp = save_dp / "fig_1d.png"
    # plot_figure_1d(data_dp, fig_fp)

    for a, address in enumerate([[0, 241, 93], [1, 235, 83], [3, 246, 45]]):
        fig_fp = save_dp / f"fig_1e_{a}.png"
        plot_figure_1e(address, data_dp, fig_fp)


if __name__ == "__main__":
    main()
