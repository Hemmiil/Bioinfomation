from umap import UMAP
from sklearn.mixture import GaussianMixture
from package_Feb14 import p00_CellSorter_Data_fixed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.mixture import GaussianMixture


class Exp():
    def __init__(self):
        self.df = p00_CellSorter_Data_fixed.main()

    def data_setup(self, random_state):
        df = self.df
        # UMAPデータのダウンロード（ward + ビーズオンリー）
        ward = "2025-01-s1"
        X_umap = pd.read_csv(f"output_Feb17/data/umap_data/{ward}/random_state_{random_state}.csv", index_col=0)

        # GMMクラスタリング
        X_bio = df[df["ward"]==ward].select_dtypes("float")
        clusters = pd.read_csv(f"output_Feb17/data/{ward}/clusters_64_clusters.csv", index_col=0)["cluster"]

        # ビーズデータのクラスタリング
        clustering_beads = GaussianMixture(random_state=0, n_components=2)

        X_beads = df[df["ward"]=="2025-01-s999"].select_dtypes("float")
        clusters_beads = clustering_beads.fit_predict(X_beads)

        X_beads_clusters = X_beads.copy()
        X_beads_clusters["cluster"] = clusters_beads
        clusters_beads = pd.Series(clusters_beads)
        X_beads_profile = X_beads_clusters.groupby("cluster").mean()

        # 0をbeads_10mu, 1をbeads_3muにしたい
        if X_beads_profile["FSC-A"][0] > X_beads_profile["FSC-A"][1]:
            X_beads_clusters = X_beads_clusters.replace(0, "beads_10mu")
            X_beads_clusters = X_beads_clusters.replace(1, "beads_3mu")

        else:
            X_beads_clusters = X_beads_clusters.replace(1, "beads_10mu")
            X_beads_clusters = X_beads_clusters.replace(0, "beads_3mu")

        clusters_beads = X_beads_clusters["cluster"]
        labels = pd.concat([clusters_beads,clusters]).reset_index(drop=True)

        return X_umap, labels

    # 可視化
    def f(self, clusters, labels, X_umap):
        plt.scatter(
            X_umap["0"],
            X_umap["1"],
            color="gray",
            s=0.1
        )

        for cluster in clusters:
            condition = labels == cluster
            if cluster == 0:
                plt.scatter(
                    X_umap["0"][condition],
                    X_umap["1"][condition],
                    s=1,
                    label=f"{cluster}: control"
                )
            else:
                plt.scatter(
                    X_umap["0"][condition],
                    X_umap["1"][condition],
                    s=1,
                    label=cluster
                )
            
        legend = plt.legend(scatterpoints=5, markerscale=5)
        plt.title(f"UMAP: Detecting Beads", weight="bold", fontsize=20)
        plt.show()

    def g(self, clusters, labels):
        df = self.df
        X_bio = df[df["ward"]=="2025-01-s1"].select_dtypes("float")
        X_beads = df[df["ward"]=="2025-01-s999"].select_dtypes("float")
        X_concat = pd.concat([X_beads, X_bio]).reset_index(drop=True)

        plt.scatter(
            X_concat["FSC-A"],
            X_concat["APC-A"],
            color="gray",
            s=0.1
        )

        for cluster in clusters:
            condition = labels == cluster
            plt.scatter(
                X_concat["FSC-A"][condition],
                X_concat["APC-A"][condition],
                s=1,
                label=f"{cluster}",
            )

        legend = plt.legend(scatterpoints=5, markerscale=5)
        plt.xlabel("FSC-A", weight="bold", fontsize=20)
        plt.ylabel("APC-A", weight="bold", fontsize=20)
        plt.title(f"Rawdata: Detecting Beads", weight="bold", fontsize=20)
        plt.show()

    def main(self):
        # 第一引数は適宜調整すること。時間があったらconfigからロードする方式に
        X_umap, labels = self.data_setup(random_state=42)
        self.f([1, 4, 32, "beads_10mu", "beads_3mu", 0], labels=labels, X_umap=X_umap)
        self.g([1, 4, 32, "beads_10mu", "beads_3mu", 0], labels=labels, )

def create_embedded_data():
    df = p00_CellSorter_Data_fixed.main()
    for random_state in [42, 43, 44]:
        for ward in ["2025-01-s1"]:
            df_filtered = df[
                np.logical_or(df["ward"]=="2025-01-s999", df["ward"]==ward)
            ]
            X = df_filtered.select_dtypes("float").reset_index(drop=True) #.iloc[:100]
            embedding = UMAP(random_state=random_state, )
            X_umap = pd.DataFrame(embedding.fit_transform(X))
            X_umap["ward"] = df_filtered["ward"].values#.iloc[:100]
            save_dir = f"output_Feb17/data/umap_data/{ward}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            pd.DataFrame(
                X_umap,
            ).to_csv(f"output_Feb17/data/umap_data/{ward}/random_state_{random_state}.csv")