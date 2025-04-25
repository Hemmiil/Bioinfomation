import pandas as pd
import scipy.stats as stats
from sklearn.decomposition import PCA
import os



class NormTest():
    def __init__(self):
        self.rawdata, self.cluster = self.get_data()
        self.pca_rst = 0
        self.exp_IDs = [0,1,2,3,4]
        # self.exp_IDs = [0]
        self.exp_ids = self.get_exp_id()

    def get_data(self):
        rawdata = pd.read_csv("output_Apr04/07__FCM_rawdata.csv", index_col=0).reset_index(drop=True)

        cluster = pd.read_csv(f"output_Apr04/clusters.csv", index_col=0).reset_index(drop=True)
        return rawdata, cluster
    
    def get_exp_id(self):
        return [f"exp_{ID}" for ID in self.exp_IDs]

    def norm_test(self, x_obs):
        ## p値が小さい＝正規性がある

        ### 歪度の検定
        skewness, skew_p_value = stats.skewtest(x_obs)

        ### 尖度の検定
        kurtosis, kurt_p_value = stats.kurtosistest(x_obs)
        return {
            "skewness": skewness,
            "skew_p_value": skew_p_value,
            "kurtosis": kurtosis,
            "kurt_p_value": kurt_p_value
        }

    def exp(self, exp_id):
        pca_rst = self.pca_rst
        norm_test_data = pd.DataFrame()
        X = self.rawdata.select_dtypes(float)
        for cluster_id in range(64):
            condition = self.cluster[exp_id]==cluster_id
            X_filtered = X[condition]
            embedding = PCA(random_state=pca_rst, n_components=3)
            
            X_pca_filtered = pd.DataFrame(
                embedding.fit_transform(X=X_filtered),
                index=X_filtered.index,
                columns=["PC1", "PC2", "PC3"]
            )

            for col in X_pca_filtered.columns:
                x_obs = X_pca_filtered[col]
                norm_test_data_row = self.norm_test(x_obs)
                norm_test_data_row["cluster"] = int(cluster_id)
                norm_test_data_row["PC_n"] = col
                norm_test_data_row["cluster_size"] = len(x_obs)
                norm_test_data[f"{cluster_id}_{col}"] = norm_test_data_row

        return norm_test_data.T

    def robust_makedir(self, dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    def main(self):
        for exp_id in self.exp_ids:
            norm_test_data = self.exp(exp_id)
            dir_path = f"output_Apr14/06__normtest"
            self.robust_makedir(dir_path)
            norm_test_data.to_csv(f"{dir_path}/rst_{exp_id}.csv")

def main():
    instance = NormTest()
    instance.main()

if __name__ == "__main__":
    main()



                