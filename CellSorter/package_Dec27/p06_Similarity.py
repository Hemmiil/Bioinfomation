from package_Dec03 import tax_filter
import pandas as pd
import numpy as np
import os
import math
from scipy.stats import t
import matplotlib.pyplot as plt
import seaborn as sns
import json
import datetime


def heatmap(with_noise=False, threashold=0):
    if with_noise:
        tax_path = "../data/contig-composition_with_noise.csv"
        dir_path = "../output_Nov16/heatmaps_r_significant_with_noise"
    else:
        tax_path = "../data/contig-composition.csv"
        dir_path = "../output_Nov16/heatmaps_r_significant"

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for tax_key in ["domain", "phylum", "order"]:
        for n_conpinents in [8,16,32]:
            df_tax = tax_filter.tax_filter(tax_key, threashold=threashold, save_path=tax_path)
            
            df_gmm = tax_filter.gmm(n_components=n_conpinents)
            
            df_result, df_r, df_p = tax_filter.analysis_corr(df_tax=df_tax, df_gmm=df_gmm)

            df_r = pd.pivot_table(
                df_result,
                index="Groups",
                columns="gmm_clusters",
                values="r-statics",
                aggfunc="sum"
            )

            df_pvalues = -pd.pivot_table(
                df_result,
                index="Groups",
                columns="gmm_clusters",
                values="p-values",
                aggfunc="sum",
            ).apply(np.log10)

            fig = plt.figure(figsize=(12,4))
            axes = fig.subplots(1,2)
            sns.heatmap(df_r, cmap="bwr", vmin=-1, vmax=1, ax=axes[0])
            axes[0].set_title("r-statics")
            
            sns.heatmap(df_pvalues,  vmin=0, vmax=8, ax=axes[1])
            axes[1].set_title("p-values")

            axes[1].set_yticks([])
            fig.suptitle(f"Tax: {tax_key.upper()}, {n_conpinents} Clusters")
            plt.tight_layout()
            
            plt.savefig(f"{dir_path}/{tax_key}-{n_conpinents}clusters.png")
            plt.close()
            print(f"Tax: {tax_key.upper()}, {n_conpinents} Clusters")

def heatmap_manual(df_result, tax_key, n_conpinents, dir_path="tmp"):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    df_r = pd.pivot_table(
        df_result,
        index="Groups",
        columns="gmm_clusters",
        values="r-statics",
        aggfunc="sum"
    )

    df_pvalues = -pd.pivot_table(
        df_result,
        index="Groups",
        columns="gmm_clusters",
        values="p-values",
        aggfunc="sum"
    ).apply(np.log10)

    fig = plt.figure(figsize=(12,4))
    axes = fig.subplots(1,2)
    sns.heatmap(df_r, cmap="bwr", vmin=-1, vmax=1, ax=axes[0])
    axes[0].set_title("r-statics")
    
    sns.heatmap(df_pvalues,  vmin=0, vmax=8, ax=axes[1])
    axes[1].set_title("p-values")

    axes[1].set_yticks([])
    fig.suptitle(f"Tax: {tax_key.upper()}, {n_conpinents} Clusters")
    plt.tight_layout()
    
    plt.savefig(f"{dir_path}/{tax_key}-{n_conpinents}clusters.png")
    plt.close()
    print(f"Tax: {tax_key.upper()}, {n_conpinents} Clusters")


def qqplot(with_noise=False, threashold=0):

    if with_noise:
        dir_path = "../output_Nov16/TaxVsCluster_with_noise"
    else:
        dir_path = "../output_Nov16/TaxVsCluster"

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for tax_key in [
        "domain",
        "phylum", 
        "order"
                    ]:
        for n_conpinents in [
            8,
            16,
            32
            ]:
            df_tax = tax_filter.tax_filter(tax_key, threashold, with_noise=with_noise)
            df_gmm = tax_filter.gmm(n_components=n_conpinents)
            df_result, p_m, r_m = tax_filter.analysis_corr(df_tax=df_tax, df_gmm=df_gmm)

            df_r = pd.pivot_table(
                df_result,
                index="Groups",
                columns="gmm_clusters",
                values="r-statics",
                aggfunc="sum"
            )

            n = 7
            r = df_result["r-statics"].fillna(0).replace(1, 1-1e-6)

            f = lambda r, n: r*math.sqrt(n-2) / math.sqrt(1-r**2)

            t_obs = [ f(r_,7) for r_ in r ]

            t_exp = t.rvs(df=5, size=len(t_obs), random_state=1)
            # グラフ描画
            fig, axes = plt.subplots(1, 2, figsize=(12,4))

            # QQプロット
            axes[0].scatter(
                np.sort(t_exp),
                np.sort(t_obs),
                label="QQ plot",
                alpha=0.7
            )
            axes[0].plot(
                np.sort(t_exp),
                np.sort(t_exp),
                label="Theoretical",
                color="black"
            )
            axes[0].set_xlabel("Theoretical")
            axes[0].set_ylabel("Observed")
            axes[0].legend()
            axes[0].set_title("QQ Plot")

            # 残差プロット
            axes[1].scatter(
                np.sort(t_obs),
                np.sort(t_obs) - np.sort(t_exp),
                label="Residuals",
                alpha=0.7
            )
            axes[1].hlines(
                y=0,
                xmin=min(t_obs),
                xmax=max(t_obs),
                color="black"
            )
            axes[1].set_xlabel("Observed")
            axes[1].set_ylabel("Observed - Theoretical")
            axes[1].legend()
            axes[1].set_title("Residuals Plot")

            fig.suptitle(f"Tax: {tax_key.upper()}, {n_conpinents} Clusters", fontsize=20, fontweight="bold")
            plt.tight_layout()
            
            plt.savefig(f"{dir_path}/{tax_key}-{n_conpinents}clusters.png")
            plt.close()
            print(f"Tax: {tax_key.upper()}, {n_conpinents} Clusters")

def qqplot_manual(df_result, tax_key, n_conpinents, dir_path="tmp"):
    r = df_result["r-statics"].fillna(0).replace(1, 1-1e-6)

    f = lambda r, n: r*math.sqrt(n-2) / math.sqrt(1-r**2)

    t_obs = [ f(r_,7) for r_ in r ]

    t_exp = t.rvs(df=5, size=len(t_obs), random_state=1)
    # グラフ描画
    fig, axes = plt.subplots(1, 2, figsize=(12,4))

    # QQプロット
    axes[0].scatter(
        np.sort(t_exp),
        np.sort(t_obs),
        label="QQ plot",
        alpha=0.7
    )
    axes[0].plot(
        np.sort(t_exp),
        np.sort(t_exp),
        label="Theoretical",
        color="black"
    )
    axes[0].set_xlabel("Theoretical")
    axes[0].set_ylabel("Observed")
    axes[0].legend()
    axes[0].set_title("QQ Plot")

    # 残差プロット
    axes[1].scatter(
        np.sort(t_obs),
        np.sort(t_obs) - np.sort(t_exp),
        label="Residuals",
        alpha=0.7
    )
    axes[1].hlines(
        y=0,
        xmin=min(t_obs),
        xmax=max(t_obs),
        color="black"
    )
    axes[1].set_xlabel("Observed")
    axes[1].set_ylabel("Observed - Theoretical")
    axes[1].legend()
    axes[1].set_title("Residuals Plot")

    fig.suptitle(f"Tax: {tax_key.upper()}, {n_conpinents} Clusters", fontsize=20, fontweight="bold")
    plt.tight_layout()
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.savefig(f"{dir_path}/{tax_key}-{n_conpinents}clusters.png")
    plt.close()
    print(f"Tax: {tax_key.upper()}, {n_conpinents} Clusters")


def main(with_noise=False, threashold=0):
    setting(with_noise=with_noise)
    heatmap(with_noise=with_noise, threashold=threashold)
    qqplot(with_noise=with_noise, threashold=threashold)

def main_manual(df_result, tax_key, n_conpinents, dir_path="tmp"):
    heatmap_manual(df_result, tax_key, n_conpinents, dir_path=dir_path+"/heatmaps_r_significant")
    qqplot_manual(df_result, tax_key, n_conpinents, dir_path=dir_path+"/TaxVsCluster")