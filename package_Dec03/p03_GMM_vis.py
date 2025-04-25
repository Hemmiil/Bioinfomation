
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def Sankey_diagram(cluster_before, cluster_after, dir_path="tmp", filename="tmp.png", is_save=False):
    # 距離行列の作成

    
    df_tmp = pd.DataFrame(
        {"Before": cluster_before.values,
        "After": cluster_after.values,
        "dummy": [1 for i in range(len(cluster_before))]}
    )

    df_move = pd.pivot_table(
        df_tmp,
        index="Before",
        columns="After",
        values="dummy",
        aggfunc="sum"
    )

    source = []
    target = []
    value = []

    for i in df_move.index:  # Before groups
        for j in df_move.columns:  # After groups
            if df_move.iloc[i, j] > 0:  # 0より大きい移動のみを追加
                source.append(f"B-{i}")
                target.append(f"A-{j}")  # After グループのインデックスは8から始まる
                value.append(df_move.iloc[i, j])

    nodes = np.sort(np.unique(target).tolist() + np.unique(source).tolist()).tolist()
    nodes_dict = {
        node: i for i, node in enumerate(nodes)
    }    

    fig = go.Figure(
        data=[go.Sankey(
        node = dict(
        pad = 10,
        thickness = 100,
        line = dict(color = "black", width = 0.5),
        label = np.sort(np.unique(target).tolist() + np.unique(source).tolist()).tolist(),
        color = "blue"
        ),
        link = dict(
        source = [nodes_dict[source_i] for source_i in source], # indices correspond to labels, eg A1, A2, A1, B1, ...
        target = [nodes_dict[target_i] for target_i in target],
        value = value
    ))])

    fig.update_layout(
        title_text="Basic Sankey Diagram", font_size=25,
        height = 1000,
        width = 1500,
    )
    if is_save:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        fig.savefig(f"{dir_path}/{filename}")
    plt.close()


def scatterplot(X):
    # 条件：カラムにcluster, FSC-A, APC-A-Compensatedがある、セルソータのデータ
    for cluster in [3,4,5]:
        condition = X["cluster"]==cluster
        condition.sum()
        plt.scatter(
            X["FSC-A"][condition], 
            X["APC-A-Compensated"][condition], 
            s=0.1,
            label=cluster,
            alpha=0.5
        )

    plt.legend()
    plt.xlabel("FSC-A", fontweight="bold", fontsize=20, )
    plt.ylabel("APC-A-Compensated", fontweight="bold", fontsize=20)
    plt.show()

# 差は縮まっているか？？
def Absolute_distance(diff_dict, dir_path="tmp", filename="tmp.png", is_save=False):
    for key in diff_dict[0]:
        diff_sub = [
            diff_dict[i].abs().mean()[key] for i in range(len(diff_dict.keys()))
        ]
        plt.plot(diff_sub, marker="o", label=key)
    plt.xticks(np.arange(len(diff_dict.keys())), np.arange(1,len(diff_dict.keys())+1))
    plt.legend()
    plt.yscale("log")
    plt.title("Difference Absolute Values", fontweight="bold")
    if is_save:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        plt.savefig(f"{dir_path}/{filename}")
    plt.close()


def System_error(diff_dict, dir_path="tmp", cols=["FSC-A", "APC-A-Compensated"],  is_save=False, ):
    diff_sum = pd.DataFrame(
        0, index=diff_dict[0].index, columns=diff_dict[0].columns 
    )

    for i in range(len(diff_dict)):
        diff_sum = diff_sum + diff_dict[i]
    diff_sum

    for key in cols:
        plt.barh(
            y = diff_sum[key].index,
            width = diff_sum[key].values
        )
        plt.title(key, size=20, fontweight="bold")
        if is_save:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            plt.savefig(f"{dir_path}/{key}.png")
        plt.close()


def PCA_Biplot(X, clusters, dir_path="tmp", filename="tmp.png", is_save=False):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    import numpy as np
    pca = PCA()
    df_pca = pca.fit_transform(X)

    for cluster in [3, 4, 5]:
        condition =  clusters == cluster
        plt.scatter(df_pca[condition, 0], df_pca[condition, 1], s=1, label=f"Cluster {cluster}", alpha=0.5)

    # バイプロット用の負荷ベクトルを追加
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    feature_names = X.columns

    for i, feature in enumerate(feature_names):
        plt.arrow(0, 0, loadings[i, 0]* 9, loadings[i, 1]*9, color='black', alpha=0.5)
        plt.text(loadings[i, 0]* 10, loadings[i, 1] * 10, feature, color='black', ha='center', va='center')

    plt.legend()
    plt.xlabel("PC1", fontweight="bold", fontsize=20)
    plt.ylabel("PC2", fontweight="bold", fontsize=20)
    plt.title("PCA Biplot")
    plt.grid(True)
    plt.axis('equal')  # 各軸を同じスケールに設定
    if is_save:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        plt.savefig(f"{dir_path}/{filename}")
    plt.close()

