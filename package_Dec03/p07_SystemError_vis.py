import pandas as pd
import numpy as np
from package_Dec03 import p01_CellSorter_Data
from sklearn.mixture import GaussianMixture
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class SystemError_vis():
    def __init__(self, is_df_default):

        if is_df_default:
            self.df = p01_CellSorter_Data.main()

        self.gmm_params = {
            "centers": [],
            "pred": []
        }

        self.df_centers = pd.DataFrame()
        self.df_distance = pd.DataFrame()
        self.wards = []
        self.linear_mean_dict = {}


    def clustering(self, n_components, random_state):
        # クラスタリングを実行して、gmm_paramsを登録させる
        df = self.df

        X = df.select_dtypes("float")
        self.wards = df["ward"]

        gmm = GaussianMixture(
            n_components=n_components, random_state=random_state
        )

        self.gmm_params["pred"] = gmm.fit_predict(X)
        self.gmm_params["centers"] = gmm.means_

    def cul_distance(self):
        df_new = self.df.select_dtypes("float")
        df_new["cluster"] = self.gmm_params["pred"]

        df_centers = df_new.copy()

        df_centers.loc[:, df_new.select_dtypes("float").columns] = np.array([self.gmm_params["centers"].loc[f"cluster:{cluster}"].values for cluster in df_new["cluster"] ])

        cols = ['FSC-A', 'BSC-A', 'FITC-A-Compensated', 'PE-A-Compensated',
            'PI-A-Compensated', 'APC-A-Compensated', 'PerCP-Cy5.5-A-Compensated',
            'PE-Cy7-A-Compensated']

        df_distance = df_new[cols] - df_centers[cols]

        df_distance["ward"] = self.wards
        df_distance["cluster"] = df_new["cluster"]

        self.df_centers = df_centers
        self.df_distance = df_distance

    def violinplots(self, dir_path):

        import os
        import matplotlib.pyplot as plt

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # クラスタ数
        n_clusters = len(self.df_distance["cluster"].unique())

        for col in self.df_distance.select_dtypes("float").columns:
            fig = plt.figure(figsize=(5*4, 3*3))
            axes = fig.subplots(ncols=5, nrows=3)

            # **ここに追加**
            fig.subplots_adjust(hspace=0.5)

            for i, ward in enumerate(self.wards):
                data_filtered = self.df_distance[self.df_distance["ward"] == ward]
                data = [data_filtered[data_filtered["cluster"] == i][col].values.tolist() for i in data_filtered["cluster"].sort_values().unique()]
                ax = axes[i % 3][i // 3]
                ax.violinplot(
                    data,
                    positions=data_filtered["cluster"].sort_values().unique(),
                    vert=False,
                    widths=1,
                    showmeans=True
                )

                ax.vlines(
                    x=0,
                    ymin=-1,
                    ymax=n_clusters,
                    linestyles="dashed",
                    colors="darkred"
                )

                ax.set_yticks(
                    data_filtered["cluster"].sort_values().unique(),
                    [i for i in data_filtered["cluster"].sort_values().unique()]
                )
                ax.set_title(f"{ward}", weight="bold", fontsize=20)

            fig.suptitle(f"{col}", weight="bold", fontsize=20)
            fig.text(
                x=0, y=0.5, s="Clusters", rotation=90, va="center", ha="center", size=15, weight="bold"
            )
            fig.tight_layout()

            fig.savefig(f"{dir_path}/{col}.png")
            plt.close()




    def boxplots(self, dir_path):
        df_distance = self.df_distance
        
        cmap = plt.get_cmap('tab20')
        num = len(self.gmm_params["pred"]["cluster"].unique())
        color = [cmap(k/num) for k in range(num)]

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        df_linear_mean = self.df_distance.groupby("ward").mean()

        wards = df_distance["ward"].sort_values().unique()
        for col in df_distance.select_dtypes("float").columns:
            fig = plt.figure(figsize=(5*4,3*3))
            axes = fig.subplots(ncols=5, nrows=3)
            # **ここに追加**
            fig.subplots_adjust(hspace=0.5)
            
            widths_list = []

            for i, ward in enumerate(wards):
                ax = axes[i % 3][i // 3]
                data_filtered = df_distance[df_distance["ward"]==ward]   

                heights = data_filtered["cluster"].value_counts().sort_index()
                widths = data_filtered.select_dtypes(["float", "int"]).groupby("cluster").mean()[col]
                linear_mean = df_linear_mean.loc[ward, col]
                k = len(data_filtered)
                

                init_height = 0
                for i in heights.index:
                    width, height = widths[i], heights[i] / k
                    # fc = face color, ec = edge color
                    r = patches.Rectangle(
                        xy=(0, init_height), width=width, height=height, fc=color[i], ec="black", fill=True, zorder=2
                    )
                    ax.add_patch(r)

                    init_height += height

                ax.vlines(0, 0, init_height*1.25)

                ax.arrow(
                    x=0, y=init_height*1.25, dx=linear_mean*0.75, dy=0,
                    width=init_height*0.1,  # 矢印の線の太さ
                    head_width=np.mean(heights)/k,  # 矢印の頭の太さ
                    head_length=np.sign(linear_mean)*linear_mean*0.25,  # 矢印の頭の長さ
                    fc="white",  # 矢印の塗りつぶし色
                    ec="black",   # 矢印の外枠色,
                    zorder=2
                )

                yticks_copy = ax.get_yticks()[1:-1]
                ax.set_yticks(
                    yticks_copy,
                    [round(v) for v in yticks_copy*k]
                )

                ax.set_title(f"{ward}", weight="bold", fontsize=20)

                widths_list.extend(widths)
            
            widths_max = max([abs(v) for v in widths_list])
            widths_max_adj = round(0.01 * (widths_max//0.01 + 1), 3)

            for i in range(len(wards)):
                ax = axes[i % 3][i // 3]
                ax.set_xlim(-abs(widths_max_adj), abs(widths_max_adj))

            fig.suptitle(f"{col}", weight="bold", fontsize=20)
            fig.savefig(f"{dir_path}/{col}.png")
            plt.close()
        
        df_linear_mean.to_csv(f"{dir_path}/linear_mean.csv")

        

    def cal_2mean(data_filtered_cp, col):
        df_weighted_mean = pd.DataFrame(
            {
                "mean": data_filtered_cp.select_dtypes(["int", "float"]).groupby("cluster").mean()[col].sort_index(),
                "count": data_filtered_cp["cluster"].value_counts().sort_index(),
            }
        )

        df_weighted_mean["weight_linear"] = df_weighted_mean["count"]
        df_weighted_mean["weight_root"] = np.sqrt(df_weighted_mean["count"])
        df_weighted_mean["mean_weight_linear"] = df_weighted_mean["mean"] * df_weighted_mean["weight_linear"] 
        df_weighted_mean["mean_weight_root"] = df_weighted_mean["mean"] * df_weighted_mean["weight_root"] 

        linear_mean = df_weighted_mean["mean_weight_linear"].sum()/df_weighted_mean["weight_linear"].sum()
        sqrt_mean = df_weighted_mean["mean_weight_root"].sum()/df_weighted_mean["weight_root"].sum()

        return linear_mean, sqrt_mean, df_weighted_mean