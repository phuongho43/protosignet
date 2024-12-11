import ast
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from protosignet.model import sim_signet
from protosignet.style import RC_PARAMS
from protosignet.util import calc_n_nodes, eval_pareto, tag_objectives


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
    with sns.axes_style("whitegrid"), mpl.rc_context(RC_PARAMS):
        fig, ax = plt.subplots(figsize=(24, 20))
        sns.scatterplot(data=df_gen_001, x="obj1", y="obj2", edgecolor="#212121", facecolor="#2ECC71", alpha=0.8, linewidth=2, s=600, antialiased=True)
        sns.scatterplot(data=df_gen_010, x="obj1", y="obj2", edgecolor="#212121", facecolor="#F1C40F", alpha=0.8, linewidth=2, s=600, antialiased=True)
        sns.scatterplot(data=df_gen_100, x="obj1", y="obj2", edgecolor="#212121", facecolor="#EA822C", alpha=0.8, linewidth=2, s=600, antialiased=True)
        sns.scatterplot(data=df_pareto, x="obj1", y="obj2", edgecolor="#212121", facecolor="#D143A4", alpha=1.0, linewidth=2, s=600, antialiased=True)
        handles = [
            mpl.lines.Line2D([], [], color="#2ECC71", marker="o", markersize=32, linewidth=0),
            mpl.lines.Line2D([], [], color="#F1C40F", marker="o", markersize=32, linewidth=0),
            mpl.lines.Line2D([], [], color="#EA822C", marker="o", markersize=32, linewidth=0),
            mpl.lines.Line2D([], [], color="#D143A4", marker="o", markersize=32, linewidth=0),
        ]
        group_labels = ["Gen 1", "Gen 10", "Gen 100", "Best (Pareto)"]
        ax.legend(handles, group_labels, loc="best")
        ax.set_xlabel("Simplicity")
        ax.set_ylabel("Performance")
        ax.xaxis.set_ticks(np.arange(0, 1.1, 0.2))
        ax.set_xlim(-0.1, 1.1)
        ax.yaxis.set_ticks(np.arange(0, 1.1, 0.2))
        ax.set_ylim(-0.1, 1.1)
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(fig_fp)
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
    df_denser = pd.DataFrame({"t": tm, "y": Xm[0]})
    df_sparser = pd.DataFrame({"t": tm, "y": Xm[1]})
    rc_f1e = dict(RC_PARAMS)
    rc_f1e.update({
        "lines.linewidth": 16,
        "axes.labelsize": 96,
        "axes.labelpad": 16,
        "axes.linewidth": 12,
        "xtick.major.pad": 16,
        "ytick.major.pad": 16,
        "xtick.labelsize": 72,
        "ytick.labelsize": 72,
        "xtick.major.size": 24,
        "ytick.major.size": 24,
        "xtick.major.width": 12,
        "ytick.major.width": 12,
        "legend.fontsize": 72,
        "legend.handletextpad": 0.4,
        "legend.labelspacing": 0.4,
        "legend.handlelength": 1,
        "grid.linewidth": 2,
    })
    with sns.axes_style("whitegrid"), mpl.rc_context(rc_f1e):
        fig, ax = plt.subplots(figsize=(24, 20))
        for t in tu[uu > 0]:
            ax.axvspan(t - 1, t, color="#648FFF", alpha=0.5, lw=0)
        sns.lineplot(data=df_sparser, x="t", y="y", ax=ax, color="#EA822C")
        sns.lineplot(data=df_denser, x="t", y="y", ax=ax, color="#8069EC", lw=12)
        ax.yaxis.set_ticks(np.arange(0, 1.1, 0.2))
        ax.set_ylim(-0.1, 1.1)
        handles = [
            mpl.lines.Line2D([], [], color="#648FFF", lw=24, alpha=0.5, solid_capstyle="projecting"),
            mpl.lines.Line2D([], [], color="#8069EC", lw=24),
            mpl.lines.Line2D([], [], color="#EA822C", lw=24),
        ]
        group_labels = ["Input", "Dense\nDecoder", "Sparse\nDecoder"]
        ax.legend(handles, group_labels, loc="upper left")
        ax.set_xlabel("Time")
        ax.set_ylabel("Output")
        ax.locator_params(axis="x", nbins=10)
        ax.locator_params(axis="y", nbins=10)
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(fig_fp)
    plt.close("all")


def plot_figure_1f(address, data_dp, fig_fp):
    """Simulate FM response for a specified motif.

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
    kr = indiv[:, 0]
    ku = indiv[:, 1]
    kX = indiv[:, 2:]
    tu = np.arange(0, 121, 1)
    periods = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 36, 40, 50, 60, 70, 80, 90, 100, 121]
    periods = np.array(periods)
    freqs = 1 / periods
    ave_denser = []
    ave_sparser = []
    for period in periods:
        uu = np.zeros_like(tu)
        uu[period:121:period] = 1
        tm, Xm = sim_signet(tu, uu, kr, ku, kX)
        ave_denser.append(np.mean(Xm[0]))
        ave_sparser.append(np.mean(Xm[1]))
    ave_denser = np.array(ave_denser)
    ave_sparser = np.array(ave_sparser)
    print(f"Dense Decoder FM Peak: {freqs[np.argmax(ave_denser)]} Hz")
    print(f"Sparse Decoder FM Peak: {freqs[np.argmax(ave_sparser)]} Hz")
    ave_denser_df = pd.DataFrame({"t": freqs, "y": ave_denser, "h": np.ones_like(freqs) * 0})
    ave_sparser_df = pd.DataFrame({"t": freqs, "y": ave_sparser, "h": np.ones_like(freqs) * 1})
    ave_df = pd.concat([ave_denser_df, ave_sparser_df], ignore_index=True)
    with sns.axes_style("whitegrid"), mpl.rc_context(RC_PARAMS):
        fig, ax = plt.subplots(figsize=(24, 20))
        palette = ["#8069EC", "#EA822C"]
        sns.lineplot(data=ave_df, x="t", y="y", hue="h", lw=16, ax=ax, palette=palette)
        ax.set_xlabel("FM Input (Hz)")
        ax.set_ylabel("Mean Ouput (AU)")
        ax.set_xscale("log")
        group_labels = ["Dense Decoder", "Sparse Decoder"]
        ax.yaxis.set_ticks(np.arange(0, 1.1, 0.2))
        ax.set_ylim(-0.1, 1.1)
        handles = [
            mpl.lines.Line2D([], [], color="#8069EC", linewidth=16, alpha=1.0),
            mpl.lines.Line2D([], [], color="#EA822C", linewidth=16, alpha=1.0),
        ]
        ax.legend(handles, group_labels, loc="upper left")
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(fig_fp)
    plt.close("all")


def plot_figure_1g(kk_ref, kk_test, fig_fp):
    """Study FM decoder motif with modified reactions.

    Args:
        kk_ref (1 x N+N+N*N array): parameters for regular FM decoder as reference
        kk_test (1 x N+N+N*N array): parameters for modified FM decoder to compare against the reference
        fig_fp (str): absolute path for saving generated figure
    """
    kk_ref = np.array(kk_ref)
    n_nodes = calc_n_nodes(len(kk_ref))
    kk_ref = kk_ref.reshape(int(n_nodes), -1)
    kk_test = kk_test.reshape(int(n_nodes), -1)
    kr_ref = kk_ref[:, 0]
    ku_ref = kk_ref[:, 1]
    kX_ref = kk_ref[:, 2:]
    kr_test = kk_test[:, 0]
    ku_test = kk_test[:, 1]
    kX_test = kk_test[:, 2:]
    tu = np.arange(0, 121, 1.0)
    uu = np.zeros_like(tu)
    uu[40:80:10] = 1.0
    uu[80:121:1] = 1.0
    tm_ref, Xm_ref = sim_signet(tu, uu, kr_ref, ku_ref, kX_ref)
    tm_test, Xm_test = sim_signet(tu, uu, kr_test, ku_test, kX_test)
    df_ref = pd.DataFrame({"t": tm_ref, "y": Xm_ref[1]})
    df_test = pd.DataFrame({"t": tm_test, "y": Xm_test[1]})
    rc_f1g = dict(RC_PARAMS)
    rc_f1g.update({
        "lines.linewidth": 16,
        "axes.labelsize": 96,
        "axes.labelpad": 16,
        "axes.linewidth": 12,
        "xtick.major.pad": 16,
        "ytick.major.pad": 16,
        "xtick.labelsize": 72,
        "ytick.labelsize": 72,
        "xtick.major.size": 24,
        "ytick.major.size": 24,
        "xtick.major.width": 12,
        "ytick.major.width": 12,
        "legend.fontsize": 72,
        "legend.handletextpad": 0.4,
        "legend.labelspacing": 0.4,
        "legend.handlelength": 1,
        "grid.linewidth": 2,
    })
    with sns.axes_style("whitegrid"), mpl.rc_context(rc_f1g):
        fig, ax = plt.subplots(figsize=(24, 20))
        for t in tu[uu > 0]:
            ax.axvspan(t - 1, t, color="#648FFF", alpha=0.5, lw=0)
        sns.lineplot(data=df_ref, x="t", y="y", ax=ax, color="#EA822C")
        sns.lineplot(data=df_test, x="t", y="y", ax=ax, color="#34495E", lw=12)
        ax.yaxis.set_ticks(np.arange(0, 1.1, 0.2))
        ax.set_ylim(-0.1, 1.1)
        handles = [
            mpl.lines.Line2D([], [], color="#648FFF", lw=24, alpha=0.5, solid_capstyle="projecting"),
            mpl.lines.Line2D([], [], color="#EA822C", lw=24),
            mpl.lines.Line2D([], [], color="#34495E", lw=24),
        ]
        group_labels = ["Input", "Regular", "Modified"]
        ax.legend(handles, group_labels, loc="upper left")
        ax.set_xlabel("Time")
        ax.set_ylabel("Output")
        ax.locator_params(axis="x", nbins=10)
        ax.locator_params(axis="y", nbins=10)
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(fig_fp)
    plt.close("all")


def main():
    data_dp = Path("/home/phuong/data/protosignet/dual_fm/data/")
    save_dp = Path("/home/phuong/data/protosignet/dual_fm/figs/")
    save_dp.mkdir(parents=True, exist_ok=True)

    fig_fp = save_dp / "fig_1d.png"
    plot_figure_1d(data_dp, fig_fp)

    for a, address in enumerate([[3, 225, 57], [1, 217, 56], [3, 246, 45]]):
        fig_fp = save_dp / f"fig_1e_{a}.png"
        plot_figure_1e(address, data_dp, fig_fp)

    fig_fp = save_dp / "fig_1f.png"
    address = [3, 246, 45]
    plot_figure_1f(address, data_dp, fig_fp)

    kk_ref = [  # regular FM decoder motif
        -1, 10, 0, 0, 0, 0, 0,
        -0.1, 0.01, -10, 10, 0, 0, 0,
        -10, 0, 0, 0, 0, 0, 0,
        -1, 0, 0, 0, 0, 0, 0,
        -10, 0, 0, 0, 0, 0, 0,
    ]  # fmt: skip
    kk_test1 = [  # No [X2] self-activation
        -1, 10, 0, 0, 0, 0, 0,
        -0.1, 0.01, -10, 0, 0, 0, 0,
        -10, 0, 0, 0, 0, 0, 0,
        -1, 0, 0, 0, 0, 0, 0,
        -10, 0, 0, 0, 0, 0, 0,
    ]  # fmt: skip
    kk_test2 = [  # High [X2] induction
        -1, 10, 0, 0, 0, 0, 0,
        -0.1, 10, -10, 10, 0, 0, 0,
        -10, 0, 0, 0, 0, 0, 0,
        -1, 0, 0, 0, 0, 0, 0,
        -10, 0, 0, 0, 0, 0, 0,
    ]  # fmt: skip
    kk_test3 = [  # both
        -1, 10, 0, 0, 0, 0, 0,
        -0.1, 10, -10, 0, 0, 0, 0,
        -10, 0, 0, 0, 0, 0, 0,
        -1, 0, 0, 0, 0, 0, 0,
        -10, 0, 0, 0, 0, 0, 0,
    ]  # fmt: skip
    for i, kk_test in enumerate([kk_test1, kk_test2, kk_test3]):
        fig_fp = save_dp / f"fig_1g_{i}.png"
        plot_figure_1g(kk_ref, kk_test, fig_fp)


if __name__ == "__main__":
    main()
