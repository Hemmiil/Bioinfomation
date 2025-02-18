# GMM適用
def clustering(df, n_componetns, ward):
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=n_componetns, random_state=0)

    # wards = ["2025-01-s1", "2025-01-s4", "2025-01-s8"]

    # ward = wards[0]

    df_filtered = df[df["ward"]==ward]

    X = df_filtered.select_dtypes("float")
    clusters = gmm.fit_predict(X)

    X_clusters = X.copy()

    X_clusters["cluster"] = clusters

    return X_clusters

import matplotlib.pyplot as plt
import numpy as np
import os

def vis(X_clusters, save_dir):
    for cluster in np.sort(X_clusters["cluster"].unique()):
        condition = X_clusters["cluster"]==cluster
    
        plt.scatter(
            X_clusters["FSC-A"],
            X_clusters["APC-A"],
            label=cluster,
            s=1,
            color="gray"
        )

        plt.scatter(
            X_clusters["FSC-A"][condition],
            X_clusters["APC-A"][condition],
            label=cluster,
            s=1,
            color="darkred"
        )

        plt.title(cluster, weight="bold", size=20)
        plt.xlabel("FSC-A", weight="bold", fontsize=20)
        plt.ylabel("APC-A", weight="bold", fontsize=20)
        plt.tight_layout()
        # plt.savefig(f"output_Feb17/images/{ward}/clusters_{num_clusters}/{cluster}.png")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(f"{save_dir}/{cluster}.png")

# ビーズ条件
import pandas as pd
def profile(X_clusters, save_filename_base):
    # クラスタープロファイルを作成
    clusters_profile = X_clusters.groupby("cluster").agg(["mean", "count"])
    df_ratio_dict = {
        "cluster_bigger": [],
        "cluster_smaller": [],
        "ratio": [],
        "FSC_judge": []
    }
    # サイズ比を列挙
    for i, cluster_i_tmp in enumerate(clusters_profile.index[:-1]):
        for j, cluster_j_tmp in enumerate(clusters_profile.index[i+1:]):
            ratio = clusters_profile["FSC-A"]["count"][cluster_i_tmp] / clusters_profile["FSC-A"]["count"][cluster_j_tmp]
            cluster_i = cluster_i_tmp
            cluster_j = cluster_j_tmp
            if ratio < 1:
                ratio = 1/ratio
                cluster_k = cluster_i
                cluster_i = cluster_j
                cluster_j = cluster_k

            FSC_judge = clusters_profile["FSC-A"]["mean"][cluster_i] > clusters_profile["FSC-A"]["mean"][cluster_j]
            
            df_ratio_dict["cluster_bigger"].append(cluster_i)
            df_ratio_dict["cluster_smaller"].append(cluster_j)
            df_ratio_dict["ratio"].append(ratio)
            df_ratio_dict["FSC_judge"].append(FSC_judge)

    df_ratio = pd.DataFrame(df_ratio_dict)
    
    X_clusters[["cluster"]].to_csv(f"{save_filename_base}_clusters.csv")
    df_ratio.to_csv(f"{save_filename_base}_ratio.csv")
    clusters_profile.to_csv(f"{save_filename_base}_profile.csv")
            
from package_Feb14 import p00_CellSorter_Data_fixed
def main():
    df = p00_CellSorter_Data_fixed.main()
    wards = ["2025-01-s1", "2025-01-s4", "2025-01-s8"]
    for ward in wards[:1]:
        for n_components in [32, 64, 128]:
            X_clusters = clustering(df.copy(), n_components, ward)
            vis(X_clusters=X_clusters, save_dir=f"output_Feb17/images/{ward}/clusters_{n_components}")
            if not os.path.exists(f"output_Feb17/data/{ward}/clusters_{n_components}"):
                os.makedirs(f"output_Feb17/data/{ward}/clusters_{n_components}")
            profile(X_clusters, save_filename_base=f"output_Feb17/data/{ward}/clusters_{n_components}")

