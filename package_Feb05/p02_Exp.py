from package_Dec03 import p01_CellSorter_Data
import pandas as pd
from package_Feb14 import p01_Normtest
import json
import os
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import time
import matplotlib.pyplot as plt

class Experiment():
    def __init__(self):
        self.df = pd.DataFrame()
        self.df_filtered = pd.DataFrame()
        

    def get_data(self, ward):
        self.df = p01_CellSorter_Data.main()
        self.df_filtered = self.df[self.df["ward"]==ward].reset_index(drop=True)

    def experiment(self, config, root_save_dir):
        import numpy as np
        df_filtered = self.df_filtered

        gmm = GaussianMixture(
            n_components=config["GMM_num_clusters"], 
            random_state=config["GMM_random_state"],
            )

        ward = df_filtered["ward"]

        X = df_filtered.select_dtypes("float")

        clusters = gmm.fit_predict(X)

        X_pca_cluster = pd.DataFrame()
        num_ignore_cluster = 0
        num_ignore_events = 0
        explained_variance_ratio = []
        for cluster in np.sort(np.unique(clusters)):
            X_filtered = X[clusters==cluster].copy()
            if len(X_filtered)>=8:
                pca = PCA(
                    random_state=config["GMM_random_state"]
                    )
                X_pca_sub = pd.DataFrame(
                    pca.fit_transform(X_filtered),
                    index=X_filtered.index,
                    columns=[f"PC_{i}" for i in range(1, len(X_filtered.columns)+1)]
                )
                explained_variance_ratio_sub = pca.explained_variance_ratio_
                X_pca_sub["cluster"] = [cluster]*len(X_pca_sub)
                X_pca_cluster = pd.concat([X_pca_cluster, X_pca_sub], axis=0)
            else:
                num_ignore_cluster += 1
                num_ignore_events += len(X_filtered)
                explained_variance_ratio_sub = np.array([None for i in range(len(X_filtered.columns))])
            explained_variance_ratio.append(explained_variance_ratio_sub)
        X_pca_cluster = X_pca_cluster.sort_index()
        X_pca_cluster["ward"] = ward

        X_raw_cluster = df_filtered.select_dtypes("float").copy()
        X_raw_cluster["cluster"] = clusters

        return X_pca_cluster, X_raw_cluster, num_ignore_cluster, num_ignore_events, explained_variance_ratio
    
    def output_X(self, X_pca_cluster, X_raw_cluster, config, root_save_dir):
        GMM_num_clusters, random_state = config["GMM_num_clusters"], config["GMM_random_state"]

        save_dir = f"{root_save_dir}/r02_X_PCA/{GMM_num_clusters}/{random_state}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        X_pca_cluster.to_csv(f"{save_dir}/X_pca.csv", index=0)

        save_dir = f"{root_save_dir}/r03_X_RAW/{GMM_num_clusters}/{random_state}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        X_raw_cluster.to_csv(f"{save_dir}/X_raw.csv", index=0)

    def output_statistics(self, X_pca_cluster, config, num_ignore_cluster, num_ignore_events, explained_variance_ratio, root_save_dir):
        result_dict = {}

        for cluster in range(config["GMM_num_clusters"]):
            result_dict[cluster] = {}
            for PC_n_id, PC_n in enumerate(["PC_1", "PC_2"]):
                result = p01_Normtest.main(
                    X_pca_cluster[X_pca_cluster["cluster"]==cluster][PC_n]
                )

                result_dict[cluster][PC_n] = result
                result_dict[cluster][PC_n]["explained_variance_ratio"] = explained_variance_ratio[cluster][PC_n_id]
            result_dict[cluster]["cluster_size"] = len(X_pca_cluster[X_pca_cluster["cluster"]==cluster][PC_n])
            result_dict["num_ignore_cluster"] = num_ignore_cluster
            result_dict["num_ignore_events"] = num_ignore_events
        
        GMM_num_clusters, random_state = config["GMM_num_clusters"], config["GMM_random_state"]
        
        save_dir = f"{root_save_dir}/r01_statistics/{GMM_num_clusters}/{random_state}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        with open(f"{save_dir}/result_dict.json", "w") as f:
            json.dump(result_dict, f, indent=2)