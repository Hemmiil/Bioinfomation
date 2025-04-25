import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd


def get_data():
    rawdata = pd.read_csv("output_Apr14/07__FCM_rawdata.csv", index_col=0).reset_index(drop=True)
    cluster = pd.read_csv("output_Apr14/clusters.csv", index_col=0)
    return rawdata, cluster


def get_hsv_colormap(n=10):
    """
    matplotlibの'hsv'カラーマップからn色を均等にサンプリングし、16進カラーコードでリスト化。
    """
    cmap = cm.get_cmap('jet')
    hex_colors = [
        mcolors.to_hex(cmap(i / (n - 1))) for i in range(n)
    ]
    return hex_colors

def robust_mkdir(path):
    import os
    if not os.path.exists(path):
        os.makedirs(path)


def visualize(rawdata, cluster, dirpath):
    ## データ整形
    condition = rawdata["ward"]=="2024-04-O-s1"
    data_filtered = rawdata[condition]
    cluster_filtered = cluster["exp_0"][condition]
    clusters_sorted = cluster_filtered.value_counts(ascending=True)
    
    ## グラデーションになるようカラーコードを出力
    colors = get_hsv_colormap(n=len(clusters_sorted))
    fig = plt.figure(figsize=(40, 12))

    ## 図表描画
    ax = fig.add_subplot(1,3,2)
    for cluster_i, color in zip(clusters_sorted.index, colors):
        condition_cluster = cluster_filtered==cluster_i
        ax.scatter(
            data_filtered["FSC-A"][condition_cluster],
            data_filtered["APC-A"][condition_cluster],
            s=5,
            color=color
        )

    ax.set_ylabel("Chl-a+++\n665nm FL4", fontsize=20, weight="bold")
    ax.set_xlabel("Forward Scatter\n488nm", fontsize=20, weight="bold")

    ax.set_xticks([5.0, 5.5, 6.0], [5.0, 5.5, 6.0],
        fontsize=20, weight="bold")
    ax.set_yticks(
        [0.2, 0.4, 0.6, 0.8],
        [0.2, 0.4, 0.6, 0.8],
        fontsize=20, weight="bold")
    ax.set_title("After Clustering", fontsize=25, weight="bold")


    ax = fig.add_subplot(1,3,1)
    ax.scatter(
        data_filtered["FSC-A"],
        data_filtered["APC-A"],
        s=1,
        color="black"
    )

    ax.set_ylabel("Chl-a+++\n665nm FL4", fontsize=20, weight="bold")
    ax.set_xlabel("Forward Scatter\n488nm", fontsize=20, weight="bold")

    ax.set_xticks([5.0, 5.5, 6.0], [5.0, 5.5, 6.0],
        fontsize=20, weight="bold")
    ax.set_yticks(
        [0.2, 0.4, 0.6, 0.8],
        [0.2, 0.4, 0.6, 0.8],
        fontsize=20, weight="bold")

    ax.set_title("Before Clustering", fontsize=25, weight="bold")

    ax = fig.add_subplot(1,3,3)
    labels = [
        "{:.1%}".format(round(v, 3)) if v > 0.01 and i % 2 == 0 else None
        for i, v in enumerate(clusters_sorted.values / len(cluster_filtered))
    ]

    ax.pie(
        labels=labels,
        x=clusters_sorted.values,
        colors=colors,
        startangle=90,
        labeldistance=1.01,
        textprops={'fontsize': 20, 'weight': 'bold', "ha": "center"}  # ← ここでスタイル指定！
    )
    ax.axis('equal')
    ax.set_title("Clusters Composition", fontsize=25, weight="bold")
    fig.suptitle("Marine Plankton Flowcytometry", fontsize=45, weight="bold")
    
    ## グラフ保存
    robust_mkdir(dirpath)
    fig.savefig(f"{dirpath}/MarinePlanktonFlowcytometry.png")

def main():
    rawdata, cluster = get_data()
    visualize(rawdata, cluster, "output_GPEES")




