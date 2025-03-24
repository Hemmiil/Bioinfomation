from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from collections import defaultdict, deque

def main():
    # configs
    ward = "2024-05-s1" # サンプルラベル
    n_clusters = 64 # クラスタ数
    random_states = [(i, i+1) for i in range(4)] # 比較するランダムステートの組
    c = 0 # 検定対象の主成分番号
    is_equal_var = True # 平均値の差における、等分散条件の有無
    threashold_percentage = 0.01

    edges = []

    for random_state in random_states:
        # データ取得
        path1 = f"output_Feb25/{ward}/r03_X_RAW/{n_clusters}/{random_state[0]}/X_raw.csv"
        path2 = f"output_Feb25/{ward}/r03_X_RAW/{n_clusters}/{random_state[1]}/X_raw.csv"

        df1 = pd.read_csv(path1)
        df2 = pd.read_csv(path2)

        df1_clusters = df1["cluster"]
        df2_clusters = df2["cluster"]

        # 主成分の抽出
        embedding = PCA(random_state=42, n_components=2)

        X_pca = embedding.fit_transform(df1.select_dtypes("float"))

        # クラスター間で主成分同士を比較。平均値の差の検定を実施し、t値とp値を記録
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

        # p値の上位1%のクラスター同士を記録

        p_mtr_reshaped = p_mtr.reshape(-1).copy()
        p_mtr_reshaped.sort()
        threashold = p_mtr_reshaped[-int((n_clusters**2)*threashold_percentage)]
        print(threashold)

        for i in range(n_clusters):
            for j in range(n_clusters):
                if p_mtr[i][j] >= threashold:
                    edges.append([f"{random_state[0]}_{i}", f"{random_state[1]}_{j}"])



    from collections import defaultdict, deque

    # グラフの隣接リスト作成（無向グラフとして扱う）
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    # BFSまたはDFSで連結成分を抽出
    def extract_connected_components(graph):
        visited = set()
        components = defaultdict(list)

        for node in graph:
            if node not in visited:
                component = []
                queue = deque([node])
                visited.add(node)
                while queue:
                    current = queue.popleft()
                    component.append(current)
                    for neighbor in graph[current]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                
                components[len(component)].append(component)
                

        return components

    # 実行
    components = extract_connected_components(graph)
    return components

if __name__ == "__main__":
    print(main())
