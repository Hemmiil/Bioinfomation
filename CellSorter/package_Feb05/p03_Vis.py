import json
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import math
import numpy as np
import os

class Vis():
    def __init__(self):
        self.root_dir = "output_Feb14"
        self.configs_ = {
            f"config_{i+1}": {
                    "PCA_random_state": [0]*6,
                    "GMM_random_state": [i]*6,
                    "GMM_num_clusters": [8,16,32,64,128,256],
                    "date": "2025/02/11",
                    "n_samples": 6
                        } for i in range(5)
        }
        self.keys = ["kurtosis", "skewness","cluster_size", "explained_variance_ratio"]
        self.summary_methods = ["non_weighted", "n weighted", "sqrt_n weighted"]


    def sign(self, num):
        if num > 0:
            return 1
        else:
            return -1

    def double_statistics_with_clustersize(self):
        sign = self.sign

        configs_ = self.configs_

        root_dir = self.root_dir
        multiple = 50

        x_statistics, y_statistics = "skewness", "explained_variance_ratio"
        for x_statistics, y_statistics in zip(
            ["skewness", "skewness"], 
            ["explained_variance_ratio", "kurtosis"]
        ):

            x_parent = []
            y_parent = []

            col, row = configs_["config_1"]["n_samples"], len(configs_.keys())
            plot_modes = ["all", "top_8"]

            for plot_mode in plot_modes:
                fig = plt.figure(figsize=(6*(col + 1), 4*row))
                axes = fig.subplots(row, col+1)
                for config_key_num, config_key in enumerate(configs_.keys()):
                    configs = configs_[config_key]

                    n = len(configs["GMM_num_clusters"])

                    for j in range(n):
                        n_clusters, random_state = configs["GMM_num_clusters"][j], configs["GMM_random_state"][j]

                        path = f"{root_dir}/r01_statistics/{n_clusters}/{random_state}/result_dict.json"
                        with open(path, "r") as f:
                            result = json.load(f)

                        x = [result[str(cluster)]["PC_1"][x_statistics] for cluster in range(n_clusters) if result[str(cluster)]["PC_1"]["cluster_size"] > 10]
                        y = [result[str(cluster)]["PC_1"][y_statistics] for cluster in range(n_clusters) if result[str(cluster)]["PC_1"]["cluster_size"] > 10]
                        s = [np.sqrt(result[str(cluster)]["PC_1"]["cluster_size"])*multiple for cluster in range(n_clusters) if result[str(cluster)]["PC_1"]["cluster_size"] > 10]
                        
                        if plot_mode == "all":
                            argsort_arr =np.argsort(s)[::-1]
                        elif plot_mode == "top_8":
                            argsort_arr =np.argsort(s)[::-1][:8]

                        x = np.array(x)[argsort_arr]
                        y = np.array(y)[argsort_arr]

                        x_parent.extend(x)
                        y_parent.extend(y)

                        ax = axes[config_key_num][j]
                        for i in range(len(x)-1):
                            ax.scatter(x=x[i:i+1], y=y[i:i+1], s=s[i:i+1], color="darkblue",alpha=0.5)

                        if random_state==4:
                            if n_clusters==8:
                                ax.set_xlabel(f"Num_Clusters:\n{n_clusters}", weight="bold", fontsize=40)
                            else:
                                ax.set_xlabel(f"{n_clusters}",weight="bold", fontsize=40)
                        if n_clusters==8:
                            if random_state==0:
                                ax.set_ylabel(f"Random:\n{random_state}", weight="bold", fontsize=40)
                            else:
                                ax.set_ylabel(f"{random_state}", weight="bold", fontsize=40)
                        #ax.set_yscale("log")
                        #ax.set_title(f"Random_state: {random_state}\nNum_clusters: {n_clusters}", weight="bold", fontsize=20)

                for config_key_num, config_key in enumerate(configs_.keys()):
                    configs = configs_[config_key]
                    for j in range(n):
                        ax = axes[config_key_num][j]
                        ax.set_xlim(1.1 * sign(min(x_parent))* abs(min(x_parent)), 1.1 * sign(max(x_parent))* abs(max(x_parent)))
                        if y_statistics == "explained_variance_ratio":
                            ax.set_ylim(0,1)
                            ax.set_yticks([0.25,0.5,0.75], ["25", "50", "75"])
                            ax.hlines(
                            y=[0.25,0.5,0.75], 
                            xmin=ax.get_xlim()[0],
                            xmax=ax.get_xlim()[1],
                            linestyles="dashed", colors="black"
                            )
                        elif y_statistics == "kurtosis":
                            ax.set_ylim(1.1 * sign(min(y_parent))* abs(min(y_parent)), 1.1 * sign(max(y_parent))* abs(max(y_parent)))
                        ax.vlines(
                            x=0, 
                            ymin=ax.get_ylim()[0],
                            ymax=ax.get_ylim()[1],
                            linestyles="dashed", colors="black"
                            )
                        ax.hlines(
                            y=[0], 
                            xmin=ax.get_xlim()[0],
                            xmax=ax.get_xlim()[1],
                            linestyles="dashed", colors="black"
                            )

                # legend 手動
                ax = axes[2][6]

                xlims, ylims = axes[0][0].get_xlim(), axes[0][0].get_ylim()
                ax.set_xlim(xlims[0], xlims[1])
                ax.set_ylim(ylims[0], ylims[1])

                scatter_x = xlims[0] + (xlims[1]-xlims[0])*0.25
                scatter_y = [ ylims[0] + (ylims[1]-ylims[0])*r for r in [0.25, 0.5, 0.75] ]
                label_x = xlims[0] + (xlims[1]-xlims[0])*0.5
                text_x = xlims[0] + (xlims[1]-xlims[0])*0.4
                text_y = ylims[0] + (ylims[1]-ylims[0])*0.9

                for i, size in zip(scatter_y, [10,100,1000,]):
                    ax.scatter([scatter_x], [i], s=np.sqrt(size)*multiple, color="darkblue")
                    ax.text(x=label_x, y=i, s=str(size), va="center", ha="center", weight="bold", fontsize=40)
                ax.text(
                    x=text_x, y=text_y, s="Cluser_size", va="bottom", ha="center", weight="bold", fontsize=40
                )

                for config_key_num in range(len(configs_.keys())):
                    ax = axes[config_key_num][6]
                    ax.axis("off")

                fig.text(
                    x = 0,
                    y = 0.5,
                    rotation=90,
                    s=y_statistics,
                    weight="bold",
                    fontsize="50",
                    va="center",
                    ha="right"
                    )

                fig.text(
                    x = 3/7,
                    y = 0,
                    s=x_statistics,
                    weight="bold",
                    fontsize="50",
                    va="top",
                    ha="center"
                    )
                # 余白を調整して見切れを防ぐ
                fig.tight_layout()
                fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

                # 画像を保存（見切れ防止）
                import os
                if not os.path.exists(f"{root_dir}/images"):
                    os.makedirs(f"{root_dir}/images")
                fig.savefig(f"{root_dir}/images/{x_statistics}_{y_statistics}_{plot_mode}.png", bbox_inches="tight")

                plt.close()

    def boxplots(self):
        configs_ = self.configs_

        keys = self.keys
        root_dir = self.root_dir

        for config_key in configs_.keys():
            configs = configs_[config_key]

            for key in keys:

                key_mean = []
                num_clusters = configs["GMM_num_clusters"]
                for i, num_cluster in enumerate(num_clusters):
                    num_cluster, random_state = configs["GMM_num_clusters"][i], configs["GMM_random_state"][i]
                    path = f"{root_dir}/r01_statistics/{num_cluster}/{random_state}/result_dict.json"
                    with open(path, "r") as f:
                        data = json.load(f)

                    skewness = [data[str(cluster)]["PC_1"][key] for cluster in range(num_cluster) if data[str(cluster)]["PC_1"]["cluster_size"]>10]
                    n = len(skewness)

                    skewness_mean = sum(skewness) / n
                    key_mean.append(skewness_mean)
                    skewness_se= sum([(v - skewness_mean)**2 for v in skewness]) / n-1

                    plt.scatter([i+1]*n, skewness, color="black", zorder=1)

                    plt.boxplot(skewness, positions=[i+1], widths=0.5, zorder=1)

                plt.plot(
                    [i for i in range(1, len(num_clusters)+1)], key_mean, 
                    marker="o", markeredgewidth=2.5, markeredgecolor="orange", markerfacecolor="white", color="orange", markersize=15, zorder=0,
                    label="mean"
                    )

                plt.xticks(
                    [i for i in range(1, len(num_clusters)+1)], num_clusters,
                    weight="bold",
                    fontsize=20
                )

                plt.hlines(0, 0.5, len(num_clusters)+0.5, color="black", linestyle="dashed")

                plt.title(key, fontsize=20, weight="bold")
                plt.xlabel("num_clusters", fontsize=15, weight="bold")
                plt.ylabel(key, fontsize=15, weight="bold")
                plt.legend()
                plt.tight_layout()

                if key=="cluster_size":
                    plt.yscale("log")
                    plt.yticks([1, 10, 100, 1000])
                
                import os
                save_dir = f"{root_dir}/images/boxplot/random_state_{random_state}"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                plt.savefig(f"{save_dir}/{key}.png")
                plt.close()

    def violineplots(self):
        import json
        import matplotlib.pyplot as plt
        import seaborn as sns

        configs_ = self.configs_

        keys = self.keys
        root_dir = self.root_dir

        for config_key in configs_.keys():
            configs = configs_[config_key]

            for key in keys:
                key_mean = []
                num_clusters = configs["GMM_num_clusters"]
                
                data_list = []
                cluster_labels = []
                
                for i, num_cluster in enumerate(num_clusters):
                    num_cluster, random_state = configs["GMM_num_clusters"][i], configs["GMM_random_state"][i]
                    path = f"{self.root_dir}/r01_statistics/{num_cluster}/{random_state}/result_dict.json"
                    with open(path, "r") as f:
                        data = json.load(f)

                    values = [data[str(cluster)]["PC_1"][key] for cluster in range(num_cluster) if data[str(cluster)]["PC_1"]["cluster_size"]>10]
                    n = len(values)

                    key_mean.append(sum(values) / n)
                    
                    # データをリストに追加
                    data_list.extend(values)
                    cluster_labels.extend([i + 1] * n)

                plt.figure(figsize=(8, 6))

                if key == "cluster_size":
                    sns.violinplot(x=cluster_labels, y=np.log10(data_list), inner="box", color="lightgray", linewidth=2)    
                else:
                    sns.violinplot(x=cluster_labels, y=data_list, inner="box", color="lightgray", linewidth=2)
                
                # 平均値プロット
                if key == "cluster_size":
                    plt.plot(
                        range(len(num_clusters)), np.log10(key_mean),
                        marker="o", markeredgewidth=2.5, markeredgecolor="orange", markerfacecolor="white",
                        color="orange", markersize=10, label="mean"
                    )
                else:
                    plt.plot(
                        range(len(num_clusters)), key_mean,
                        marker="o", markeredgewidth=2.5, markeredgecolor="orange", markerfacecolor="white",
                        color="orange", markersize=10, label="mean"
                    )


                plt.xticks(range(0, len(num_clusters)), num_clusters, fontsize=20, weight="bold")
                plt.hlines(0, -0.5, len(num_clusters) - 0.5, color="black", linestyle="dashed")

                plt.title(key, fontsize=25, weight="bold")
                plt.xlabel("num_clusters", fontsize=20, weight="bold")
                plt.ylabel(key, fontsize=20, weight="bold")
                plt.legend()
                plt.tight_layout()

                if key == "cluster_size":
                    plt.yticks([0,1,2,3], [1, 10, 100, 1000])
                
                elif key == "explained_variance_ratio":
                    plt.yticks([0, 0.25, 0.5, 0.75,1], [0,25, 50, 75,100])
                    plt.hlines([0, 0.25, 0.5, 0.75,1], -0.5, len(num_clusters) - 0.5, color="black", linestyle="dashed")


                import os
                save_dir = f"{root_dir}/images/violineplots/random_state_{random_state}"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                plt.savefig(f"{save_dir}/{key}.png")
                plt.close()

    def statistics_summary_mean(self):
        configs_ = self.configs_

        keys = self.keys
        mean_methods = self.summary_methods
        root_dir = self.root_dir

        for config_key in configs_.keys():
            configs = configs_[config_key]

            for mean_method in mean_methods:
                fig = plt.figure(figsize=(6*len(keys), 5))
                fig.subplots_adjust(hspace=0.5, wspace=0.5)
                axes = fig.subplots(1, len(keys))

                None_count_dict = []
                for configs_key in configs_.keys():
                    configs = configs_[configs_key]
                    for key_id, key in enumerate(keys):
                        key_mean = []
                        key_var = []
                        num_clusters = configs["GMM_num_clusters"]

                        ax = axes[key_id]
                        for i, num_cluster in enumerate(num_clusters):
                            num_cluster, random_state = configs["GMM_num_clusters"][i], configs["GMM_random_state"][i]
                            path = f"{root_dir}/r01_statistics/{num_cluster}/{random_state}/result_dict.json"
                            with open(path, "r") as f:
                                data = json.load(f)

                            statistics = [data[str(cluster)]["PC_1"][key] for cluster in range(num_cluster) if data[str(cluster)]["PC_1"]["cluster_size"]>10]
                            cluster_size = [data[str(cluster)]["PC_1"]["cluster_size"] for cluster in range(num_cluster) if data[str(cluster)]["PC_1"]["cluster_size"]>10]

                            if None in statistics:
                                None_count = sum([1 for v in cluster_size if v<10])
                                None_count_dict.append(
                                    {
                                        (num_cluster, configs_key): None_count
                                    }
                                )
                                statistics = [v for i, v in enumerate(statistics) if cluster_size[i]>=10]
                                cluster_size = [v for i, v in enumerate(cluster_size) if cluster_size[i]>=10]

                            n = len(statistics)

                            if mean_method == "non_weighted":
                                statistics_mean = sum(statistics) / n
                                statistics_var = sum([(v - statistics_mean)**2 for v in statistics]) / n
                            
                            elif mean_method == "n weighted":
                                # n 加重平均
                                statistics_mean = sum([st * n for st, n in zip(statistics, cluster_size)]) / sum([n for n in cluster_size])
                                statistics_var = sum([((v - statistics_mean)**2)*n for v, n in zip(statistics, cluster_size)]) / sum([n for n in cluster_size])

                            elif mean_method == "sqrt_n weighted":
                                # √n 加重平均
                                statistics_mean = sum([st * np.sqrt(n) for st, n in zip(statistics, cluster_size)]) / sum([np.sqrt(n) for n in cluster_size])
                                statistics_var = sum([((v - statistics_mean)**2)*np.sqrt(n) for v, n in zip(statistics, cluster_size)]) / sum([np.sqrt(n) for n in cluster_size])


                            key_mean.append(statistics_mean)
                            key_var.append(statistics_var)

                            # plt.scatter([i+1]*n, skewness, color="black", zorder=1)
                            #plt.scatter([i+1], skewness_mean, c="white", ec="orange",s=200)

                            # plt.boxplot(skewness, positions=[i+1], widths=0.5, zorder=1)

                        ax.plot(
                            [i for i in range(1, len(num_clusters)+1)], key_mean, 
                            marker="o", markeredgewidth=2.5, markerfacecolor="white", markersize=15, zorder=0,
                            label=configs_key
                            )

                        ax.set_xticks(
                            [i for i in range(1, len(num_clusters)+1)], num_clusters,
                            weight="bold",
                            fontsize=20
                        )

                        ax.hlines(0, 0.5, len(num_clusters)+0.5, color="black", linestyle="dashed")

                        ax.set_title(key, fontsize=20, weight="bold")
                        ax.set_xlabel("num_clusters", fontsize=15, weight="bold")
                        ax.set_ylabel(key, fontsize=15, weight="bold")
                        if key=="cluster_size":
                            ax.set_yscale("log")
                            ax.set_yticks([1e1, 1e2, 1e3])
                        if key=="explained_variance_ratio":
                            ax.set_yscale("log")
                            #ax.set_yticks([1e1, 1e2, 1e3])


                for i in range(len(keys)):
                    ax = axes[i]
                    ax.legend()
                fig.suptitle(f"Mean: {mean_method}", fontsize=25, weight="bold")
                fig.tight_layout()  

                import os
                save_dir = f"{root_dir}/images/statistics_summary/mean"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                plt.savefig(f"{save_dir}/{mean_method}.png")
                plt.close()

    def statistics_summary_var(self):
        configs_ = self.configs_

        keys = self.keys
        mean_methods = self.summary_methods
        root_dir = self.root_dir

        for config_key in configs_.keys():
            configs = configs_[config_key]

            for mean_method in mean_methods:
                fig = plt.figure(figsize=(6*len(keys), 5))
                fig.subplots_adjust(hspace=0.5, wspace=0.5)
                axes = fig.subplots(1, len(keys))

                None_count_dict = []

                for configs_key in configs_.keys():
                    configs = configs_[configs_key]
                    for key_id, key in enumerate(keys):
                        key_mean = []
                        key_var = []
                        num_clusters = configs["GMM_num_clusters"]

                        ax = axes[key_id]
                        for i, num_cluster in enumerate(num_clusters):
                            num_cluster, random_state = configs["GMM_num_clusters"][i], configs["GMM_random_state"][i]
                            path = f"{root_dir}/r01_statistics/{num_cluster}/{random_state}/result_dict.json"
                            with open(path, "r") as f:
                                data = json.load(f)

                            statistics = [data[str(cluster)]["PC_1"][key] for cluster in range(num_cluster) if data[str(cluster)]["PC_1"]["cluster_size"]>10]
                            cluster_size = [data[str(cluster)]["PC_1"]["cluster_size"] for cluster in range(num_cluster) if data[str(cluster)]["PC_1"]["cluster_size"]>10]

                            if None in statistics:
                                None_count = sum([1 for v in cluster_size if v<10])
                                None_count_dict.append(
                                    {
                                        (num_cluster, configs_key): None_count
                                    }
                                )
                                statistics = [v for i, v in enumerate(statistics) if cluster_size[i]>=10]
                                cluster_size = [v for i, v in enumerate(cluster_size) if cluster_size[i]>=10]

                            n = len(statistics)

                            if mean_method == "non_weighted":
                                statistics_mean = sum(statistics) / n
                                statistics_var = sum([(v - statistics_mean)**2 for v in statistics]) / n
                            
                            elif mean_method == "n weighted":
                                # n 加重平均
                                statistics_mean = sum([st * n for st, n in zip(statistics, cluster_size)]) / sum([n for n in cluster_size])
                                statistics_var = sum([((v - statistics_mean)**2)*n for v, n in zip(statistics, cluster_size)]) / sum([n for n in cluster_size])

                            elif mean_method == "sqrt_n weighted":
                                # √n 加重平均
                                statistics_mean = sum([st * np.sqrt(n) for st, n in zip(statistics, cluster_size)]) / sum([np.sqrt(n) for n in cluster_size])
                                statistics_var = sum([((v - statistics_mean)**2)*np.sqrt(n) for v, n in zip(statistics, cluster_size)]) / sum([np.sqrt(n) for n in cluster_size])

                            key_mean.append(statistics_mean)
                            key_var.append(statistics_var)

                            # plt.scatter([i+1]*n, skewness, color="black", zorder=1)
                            #plt.scatter([i+1], skewness_mean, c="white", ec="orange",s=200)

                            # plt.boxplot(skewness, positions=[i+1], widths=0.5, zorder=1)

                        ax.plot(
                            [i for i in range(1, len(num_clusters)+1)], key_var, 
                            marker="o", markeredgewidth=2.5, markerfacecolor="white", markersize=15, zorder=0,
                            label=configs_key
                            )

                        ax.set_xticks(
                            [i for i in range(1, len(num_clusters)+1)], num_clusters,
                            weight="bold",
                            fontsize=20
                        )

                        ax.hlines(0, 0.5, len(num_clusters)+0.5, color="black", linestyle="dashed")

                        ax.set_title(key, fontsize=20, weight="bold")
                        ax.set_xlabel("num_clusters", fontsize=15, weight="bold")
                        ax.set_ylabel(key, fontsize=15, weight="bold")
                        if key=="cluster_size":
                            ax.set_yscale("log")
                            ax.set_yticks([1e1, 1e2, 1e3])

                for i in range(len(keys)):
                    ax = axes[i]
                    ax.legend()
                fig.suptitle(f"Variance: {mean_method}", fontsize=25, weight="bold")
                plt.tight_layout()
                
                save_dir = f"{root_dir}/images/statistics_summary/var"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                plt.savefig(f"{save_dir}/{mean_method}.png")
                plt.close()

