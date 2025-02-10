from package_Dec03 import p01_CellSorter_Data
import pandas as pd
from package_Feb05 import p01_Normtest
import json
import os
from sklearn .decomposition import PCA
from sklearn.mixture import GaussianMixture
import time
import matplotlib.pyplot as plt

class Experiment():
    def __init__(self):
        self.df_filtered = pd.DataFrame()

    def get_data(self):
        df = p01_CellSorter_Data.main()
        self.df_filtered = df[df["ward"]=="2024-12-s1"].reset_index(drop=True)

    def experiment(self, config):
        df_filtered = self.df_filtered
        pca = PCA(random_state=config["PCA_random_state"])

        gmm = GaussianMixture(
            n_components=config["GMM_num_clusters"], 
            random_state=config["GMM_random_state"],
            )

        ward = df_filtered["ward"]

        X = df_filtered.select_dtypes("float")
        X_pca = pd.DataFrame(
            pca.fit_transform(X), columns=[f"PC_{i}" for i in range(1, len(X.columns)+1)]
        )

        clusters = gmm.fit_predict(X_pca)

        X_cluster = X_pca.copy()
        X_cluster["cluster"] = clusters
        X_cluster["ward"] = ward

        X_raw_cluster = df_filtered.select_dtypes("float").copy()
        X_raw_cluster["cluster"] = clusters

        return X_cluster, X_raw_cluster
    
    def output_X(self, X_cluster, X_raw_cluster, config):
        GMM_num_clusters, random_state = config["GMM_num_clusters"], config["GMM_random_state"]

        save_dir = f"output_Feb07/r02_X_PCA/{GMM_num_clusters}/{random_state}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        X_cluster.to_csv(f"{save_dir}/X_pca.csv", index=0)

        save_dir = f"output_Feb07/r03_X_RAW/{GMM_num_clusters}/{random_state}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        X_raw_cluster.to_csv(f"{save_dir}/X_raw.csv", index=0)

    def output_statistics(self, X_cluster, config):
        result_dict = {}

        for cluster in range(config["GMM_num_clusters"]):
            result_dict[cluster] = {}
            for PC_n in ["PC_1", "PC_2"]:
                result = p01_Normtest.norm_statistics(
                    X_cluster[X_cluster["cluster"]==cluster][PC_n]
                )
                result_dict[cluster][PC_n] = result
            result_dict[cluster]["cluster_size"] = len(X_cluster[X_cluster["cluster"]==cluster][PC_n])
        
        GMM_num_clusters, random_state = config["GMM_num_clusters"], config["GMM_random_state"]
        
        save_dir = f"output_Feb07/r01_statistics/{GMM_num_clusters}/{random_state}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        with open(f"{save_dir}/result_dict.json", "w") as f:
            json.dump(result_dict, f, indent=2)
