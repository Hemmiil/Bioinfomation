from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import json
import os
from statsmodels.multivariate.manova import MANOVA



class rst_experiment():
    def __init__(self):
        self.ward = "2024-05-s1" # サンプルラベル
        self.n_clusters = 64 # クラスタ数
        self.random_states = [i for i in range(5)] # 比較するランダムステートの組
        self.c = 0 # 検定対象の主成分番号
        self.is_equal_var = True # 平均値の差における、等分散条件の有無
        self.threashold_percentage = 0.01
        self.is_pca_done = False

        pathes = [
            f"output_Feb25/{self.ward}/r03_X_RAW/{self.n_clusters}/{random_state_i}/X_raw.csv"
            for random_state_i in self.random_states
        ]
        self.data = {
            rst: pd.read_csv(path) for rst, path in zip(self.random_states, pathes)
        }
        
    def get_data(self, random_state_i, random_state_j):
        df1 = self.data[random_state_i]
        df2 = self.data[random_state_j]

        df1_clusters = df1["cluster"]
        df2_clusters = df2["cluster"]

        return df1, df1_clusters, df2_clusters
    
    def get_PC(self, df):
        embedding = PCA(random_state=42, n_components=2)

        X_pca = embedding.fit_transform(df.select_dtypes("float"))
        explained_variance_ratio = embedding.explained_variance_ratio_
        return X_pca, explained_variance_ratio
    
    def t_test(self, X_pca, df1_clusters, df2_clusters):
        n_clusters, c, is_equal_var = self.n_clusters, self.c, self.is_equal_var
        t_mtr = np.zeros((n_clusters, n_clusters))
        p_mtr = np.zeros((n_clusters, n_clusters))

        tmp = n_clusters
        for i in range(tmp):
            for j in range(tmp):
                t = ttest_ind(
                    X_pca[df1_clusters==i][c],
                    X_pca[df2_clusters==j][c],
                    equal_var=is_equal_var
                )

                t_mtr[i][j] = t.statistic
                p_mtr[i][j] = t.pvalue

        return t_mtr, p_mtr

    def manova(self, df, df1_clusters, df2_clusters):
        # クラスタ数を取得
        n_clusters = self.n_clusters
        p_mtr = np.zeros((n_clusters, n_clusters))

        # カラム名を修正
        df.columns = [f"Var_{i}" for i in range(1, 1 + len(df.columns))]
        cols = df.select_dtypes(float).columns
        fomula_str = " + ".join(cols) + " ~ cluster"

        # データフレームを作成
        df_duble = pd.concat([df, df])
        df_duble["cluster"] = pd.concat([df1_clusters.astype(str) + "_df1", df2_clusters.astype(str) + "_df2"])

        # クラスタごとのデータを辞書に格納
        cluster_dict = {
            f"{i}_df1": df_duble[df_duble["cluster"] == f"{i}_df1"]
            for i in range(n_clusters)
        }
        cluster_dict.update({
            f"{j}_df2": df_duble[df_duble["cluster"] == f"{j}_df2"]
            for j in range(n_clusters)
        })

        # 二重ループでMANOVAを実施
        for i in range(n_clusters):
            for j in range(n_clusters):
                # 既にフィルタ済みのデータを使う
                df_filtered = pd.concat([cluster_dict[f"{i}_df1"], cluster_dict[f"{j}_df2"]])

                # MANOVAを実施
                maov = MANOVA.from_formula(fomula_str, data=df_filtered)
                result = maov.mv_test()
                pillai_trace_p = result.results['cluster']['stat'].iloc[0, 4]
                p_mtr[i, j] = float(pillai_trace_p)

        return p_mtr

    def extract_connected_components(self, graph):
        visited = set()
        components = defaultdict(list)

        for node in graph:
            if node not in visited:
                queue = deque([node])
                visited.add(node)
                component_dict = {}

                while queue:
                    current = queue.popleft()
                    neighbors = graph[current]
                    component_dict[current] = neighbors
                    for neighbor in neighbors:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                components[len(component_dict)].append(component_dict)

        return components

    def create_subgraph(self, p_mtr, random_state_i, random_state_j):
        # p値の上位 n_clusters 個のエッジを記録
        # 上位 n % とか、上位 n 個とかのマジックナンバーを検証したい
        n_clusters = self.n_clusters
        edges = []
        ps = []

        p_mtr_reshaped = p_mtr.reshape(-1).copy()
        p_mtr_reshaped.sort()
        # threashold = p_mtr_reshaped[-int((n_clusters**2)*threashold_percentage)]
        threashold = p_mtr_reshaped[-n_clusters]
        # print(threashold)

        for i in range(n_clusters):
            for j in range(n_clusters):
                if p_mtr[i][j] >= threashold:
                    edges.append([f"{random_state_i}_{i}", f"{random_state_j}_{j}"])
                    ps.append(p_mtr[i][j])

        return edges, ps

    def completed_graph(self, edges):
        # グラフの隣接リスト作成（無向グラフとして扱う）
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)

        # BFSまたはDFSで連結成分を抽出

        # 実行
        return self.extract_connected_components(graph)

    def main_old(self):
        edges = []
        edges_p = []
        explained_variance_ratio_dict = {}
        for i, random_state_i in enumerate(self.random_states):
            for j, random_state_j in enumerate(self.random_states[i+1:]):
                df, df1_clusters, df2_clusters = self.get_data(random_state_i=random_state_i, random_state_j=random_state_j)
                if self.is_pca_done == False:
                    X_pca, explained_variance_ratio = self.get_PC(df=df)
                    self.is_pca_done = True
                    explained_variance_ratio_dict[(random_state_i, random_state_j)] = explained_variance_ratio
                _, p_mtr = self.t_test(X_pca, df1_clusters, df2_clusters)
                edge_sub, ps_sub = self.create_subgraph(p_mtr=p_mtr, random_state_i=random_state_i, random_state_j=random_state_j)
                edges.extend(edge_sub)
                edges_p.extend(ps_sub)

        graphs_dict = self.completed_graph(edges=edges)
        
        self.store_data(graphs_dict)
        
        return graphs_dict, edges, edges_p, explained_variance_ratio_dict

    def main(self):
        edges = []
        edges_p = []
        for i, random_state_i in enumerate(self.random_states):
            for j, random_state_j in enumerate(self.random_states[i+1:]):
                df, df1_clusters, df2_clusters = self.get_data(random_state_i=random_state_i, random_state_j=random_state_j)
                p_mtr = self.manova(df, df1_clusters, df2_clusters)
                edge_sub, ps_sub = self.create_subgraph(p_mtr=p_mtr, random_state_i=random_state_i, random_state_j=random_state_j)
                edges.extend(edge_sub)
                edges_p.extend(ps_sub)
                print(f"rst({i}, {j}) done")

        graphs_dict = self.completed_graph(edges=edges)
        
        self.store_data(graphs_dict)
        
        return graphs_dict, edges, edges_p

    def store_data(self, data):
        save_dir = "output_Mar04/tmp"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with open(f"{save_dir}/tmp.json", 'w') as f:
            json.dump(data, f, indent=2)        
        

            

if __name__ == "__main__":
    rst_experiment_instance = rst_experiment()
    adict, _, _ = rst_experiment_instance.main()

    for k, v in adict.items():
        print(f"{k}: {len(v)}")
