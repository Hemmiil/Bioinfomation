import matplotlib.pyplot as plt
import pandas as pd

# 任意の種グループと関連がありそうなクラスタを可視化する
class agent():
    def __init__(self):
        self.rst = 0
        self.rawdata, self.cluster = None, None

        self.r_p_long = None

        self.r_p_long_filtered = None

    def get_data(self):
        rawdata = pd.read_csv("output_Apr04/07__FCM_rawdata.csv", index_col=0)
        cluster = pd.read_csv("output_Apr04/clusters.csv")
        data = pd.read_csv(f"output_Apr14/05__images/rst_{self.rst}/non_bs/p_table.csv")
        return data, rawdata, cluster

    def set_data(self):
        self.rawdata = pd.read_csv("output_Apr04/07__FCM_rawdata.csv", index_col=0)
        self.cluster = pd.read_csv("output_Apr04/clusters.csv")
        self.r_p_long = self.get_r_p_long()
        
    def filter_spgroup(self, sp_group):
        data = self.r_p_long
        data["is_significant"] = data["p_values"] < 0.01
        data_significant = data[data["is_significant"]]
        try:
            data_significant_filtered = data_significant.loc[sp_group]
            cluster_ids = data_significant_filtered.index
            self.r_p_long_filtered = data_significant_filtered

        except KeyError as e:
            cluster_ids = []

        return cluster_ids
    
        
    def get_r_p_long(self):
        # pvalue と r のlongデータの作成
        path_p = f"output_Apr14/01__p_table/{self.rst}.csv"
        path_r = f"output_Apr14/02__r_table/{self.rst}.csv"

        data_p = pd.read_csv(path_p, index_col=0)
        data_r = pd.read_csv(path_r, index_col=0)

        # どちらもロング型にする
        data_long_p = pd.DataFrame(data_p.stack(), columns=["p_values"])
        data_long_r = pd.DataFrame(data_r.stack(), columns=["r_values"])

        # 結合
        return pd.concat(
            [data_long_p, data_long_r], axis=1)
    
    def vis(self, cluster_ids, sp_group):
        fig = plt.figure(figsize=(12, 8))
        rawdata = self.rawdata

        for i, col in enumerate(rawdata.select_dtypes(float).columns[:8]):
            ax = fig.add_subplot(3, 3, i + 1)

            for j, cluster_id in enumerate(cluster_ids):
                condition = self.cluster[f"exp_{self.rst}"] == int(cluster_id)
                data_filtered = rawdata.reset_index()[condition]

                parts = ax.violinplot(
                    data_filtered[col],
                    positions=[j],
                    showmeans=False,
                    showmedians=True
                )
                for pc in parts['bodies']:
                    pc.set_facecolor("tab:blue")
                    pc.set_edgecolor("black")
                    pc.set_alpha(1.0)

                parts['cbars'].set_color("black")
                parts['cmaxes'].set_color("black")
                parts['cmins'].set_color("black")
                parts['cmedians'].set_color("black")
                

            # 全体分布
            parts_all = ax.violinplot(
                rawdata[col],
                positions=[-1],
                showmeans=False,
                showmedians=True
            )
            for pc in parts_all['bodies']:
                pc.set_facecolor("tab:orange")
                pc.set_edgecolor("black")
                pc.set_alpha(1.0)

            parts_all['cbars'].set_color("black")
            parts_all['cmaxes'].set_color("black")
            parts_all['cmins'].set_color("black")
            parts_all['cmedians'].set_color("black")


            ax.set_xticks([i for i in range(-1, len(cluster_ids))])
            ax.set_xticklabels(["All"] + [f"cls.{cluster_id}" for cluster_id in cluster_ids], weight="bold", fontsize=12)

            ax.set_title(col, weight="bold", fontsize=14)

        # 相関結果テキスト
        ax_r = fig.add_subplot(3, 3, 9)
 
        s = "\n".join(
            [
                f"cls.{self.r_p_long_filtered.index[i]} -> r: {round(self.r_p_long_filtered.iloc[i]['r_values'], 3)}, p_value: {round(self.r_p_long_filtered.iloc[i]['p_values'], 3)}" for i in range(len(self.r_p_long_filtered.index))
                ]
            )
        
        ax_r.axis("off")
        ax_r.text(
            0, 0,
            s=s, 
            fontsize=12,
            weight="bold",
            ha="center",
            va="center"
        )
        ax_r.set_xlim(-1, 1)
        ax_r.set_ylim(-1, 1)

        fig.suptitle(f"rst: {self.rst} {sp_group}", weight="bold", fontsize=20)

        fig.tight_layout()
        dir_path = f"output_Apr14/08_sig_cluster_features/{sp_group}"
        self.robust_mkdir(dir_path)
        file_path = f"{dir_path}/rst_{self.rst}.png"
        fig.savefig(file_path)
        plt.close()
        return parts

    def robust_mkdir(self, path):
        import os
        if not os.path.exists(path):
            os.makedirs(path)


    def main(self, sp_group):
        self.set_data()
        cluster_ids = self.filter_spgroup(sp_group)
        if len(cluster_ids)>0:
            self.vis(cluster_ids, sp_group)

def main():
    sp_groups = ['Metazoa:Bony fishes', 'Bacteria:alpha-proteobacteria',
        'Eukaryota:Other diatoms', 'Bacteria:Other bacteria',
        'Bacteria:CFB group', 'Eukaryota:Green algae', 'Metazoa:Copepods',
        'Eukaryota:Centric diatoms', 'Bacteria:Cyanobacteriota',
        'Eukaryota:Dinoflagellates', 'Bacteria:gamma-proteobacteria',
        'Eukaryota:Haptophytes', 'Metazoa:Other metazoa', 'Metazoa:Tunicates']
    instance = agent()
    for sp_group in sp_groups:
        for rst in range(5):
            instance.rst = rst
            instance.main(sp_group)
        print(f"{sp_group} done")
    

if __name__ == "__main__":
    main()