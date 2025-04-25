import pandas as pd
from package_Mar04 import p10_bootstrap
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities, modularity



def get_data():
    instance = p10_bootstrap.Exp()
    instance.crt_dataset()

    return instance


def crt_graphs(instance):
    Gs = {}
    for key in ["real_data", "sample_shuffle"]:
        # --- グラフ構築 ---
        G = nx.Graph()
        corr_matrix = instance.random_controls_corr[key]
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.index)):
                corr = corr_matrix.iloc[j, i]
                if corr > 0.1:
                    G.add_edge(corr_matrix.columns[i], corr_matrix.index[j], weight=max(0, corr))
        Gs[key] = G

    # モジュラリティの算出

    # --- コミュニティ検出 ---
    communities = {}
    for key in ["real_data", "sample_shuffle"]:
        communities[key] = list(greedy_modularity_communities(Gs[key], weight="weight"))


    # --- モジュラリティ算出（重み付き）---
    Q = {}
    for key in ["real_data", "sample_shuffle"]:
        Q[key] = modularity(Gs[key], communities[key], weight="weight")

    return Gs, Q

def visualize(Gs, dirpath):
# --- 可視化 ---
    for key, subtitile in zip(["real_data", "sample_shuffle"], ["Real Data", "Negative Control"]):
        G = Gs[key]
        colors_list = ["tab:blue", "tab:orange"]
        colors = []
        for node in G.nodes:
            if node in [str(i) for i in range(64)]:
                colors.append(colors_list[0])
            else:
                colors.append(colors_list[1])
                
        
        pos = nx.spring_layout(G, seed=42)
        edges = G.edges(data=True)
        weights = [abs(d['weight']) for (_, _, d) in edges]

        plt.figure(figsize=(12, 6))
        nx.draw(G, pos, with_labels=False, node_color=colors, width=weights)
        plt.title(subtitile, fontsize=20, weight="bold")
        plt.tight_layout()

        robust_mkdir(dirpath)
        plt.savefig(f"{dirpath}/{key}.png")

def robust_mkdir(dirpath):
    import os
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def main():
    instance = get_data()
    Gs, Q = crt_graphs(instance)
    print(Q)
    visualize(Gs, "output_tmp/networks/")

if __name__ == "__main__":
    main()




