
from package_Dec03 import p01_CellSorter_Data
from itertools import product


# データ整形
df = p01_CellSorter_Data.main()

print("------")
print(df)
print("------")
# GMM_loop実験の実行
from package_Dec03 import p02_GMM_loop

# configの全てのパターンを作成

config = {
    "n_movements": [3],
    "n_clusters": [4,8,16],
    "magnification": [0.1, 0.5, 1.0],
}
random_state_list = [0,1,2]

keys = config.keys()
values = product(*config.values())
config_sub_list = [{k: v for k, v in zip(keys, combination)} for combination in values]

print(config_sub_list)

for i, save_dir in enumerate([f"output_Jan20/data/{id}" for id in range(len(config_sub_list))]):
    config_sub = config_sub_list[i]
    n_movements, n_clusters, magnification = config_sub["n_movements"], config_sub["n_clusters"], config_sub["magnification"]
    result = p02_GMM_loop.experiments(
        df = df,
        N = n_movements, # 移動回数
        n_components=n_clusters, #クラスタ数
        save_dir=save_dir, #保存ディレクトリ
        alpha=magnification, # 移動倍率
        random_state_list=random_state_list, # ランダムステートのリスト
        is_dummy=False
    )
