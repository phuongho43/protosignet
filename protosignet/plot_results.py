from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from protosignet.util import eval_pareto, tag_objectives

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
    # print(df_pareto)
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


def main():
    data_dp = Path("/home/phuong/data/protosignet/dual_fm/data/")
    save_dp = Path("/home/phuong/data/protosignet/dual_fm/figs/")
    save_dp.mkdir(parents=True, exist_ok=True)
    fig_fp = save_dp / "fig_1d.png"
    plot_figure_1d(data_dp, fig_fp)


if __name__ == "__main__":
    main()
