import p02_CellSorter_Data
import p01_PlanDyO
import pandas as pd
from scipy.stats import pearsonr
from sklearn.mixture import GaussianMixture
import os
import json

configs = [
    {
        "rst": 0, "num_clusters": 64
    },
    {
        "rst": 1, "num_clusters": 64
    },
    {
        "rst": 2, "num_clusters": 64
    },
    {
        "rst": 3, "num_clusters": 64
    },
    {
        "rst": 4, "num_clusters": 64
    },
]

configs_tmp = [
    {
        "rst": 0, "num_clusters": 64
    }
]

# データ取得
def get_data():
    df = p02_CellSorter_Data.main()
    return df

# GMMクラスタリング
def gmm_clustering(df, config):
    gmm = GaussianMixture(
        random_state=config["rst"],
        n_components=config["num_clusters"]
    )
    X = df.select_dtypes(float)
    clusters = gmm.fit_predict(X)
    return clusters

# クラスタリング結果集計
def create_FCM_table(df, clusters):
    profiles = df[["ward"]].copy()
    profiles["cluster"] = clusters
    profiles["dummy"] = 1
    FCM_table = pd.pivot_table(
        profiles,
        columns="ward",
        index="cluster",
        values="dummy",
        aggfunc="sum",   
    ).fillna(0)
    FCM_table = (FCM_table / FCM_table.sum(axis=0))
    return FCM_table

# PlanDyO データ整形
def create_DNA_table():
    DNA_table = p01_PlanDyO.make_contig_csv(
        #path="/Users/henmi_note/Desktop/CellSorter/data/PlanDyO/Kraken.upper_group-sample.txt",
        path = "output_Apr04/PlanDyO_rawdata.txt",
        is_filtered=True
        )

    # PlanDyOデータ集計
    threashold = 1e-6
    is_drop = [ DNA_table.loc[indice].min() < threashold for indice in DNA_table.index ]

    DNA_table = DNA_table.drop(DNA_table.index[is_drop])
    return DNA_table

# CellSorter x PlanDyO データ結合
def create_concat_table(DNA_table, FCM_table):
    cat_table = pd.concat([DNA_table, FCM_table]).dropna(axis=1).T
    return cat_table

# 相関係数評価、データ集計
def r_test(FCM_table, DNA_table, cat_table):
    p_table, r_table = pd.DataFrame(
        columns=FCM_table.index,
        index=DNA_table.index
    ), pd.DataFrame(
        columns=FCM_table.index,
        index=DNA_table.index
    )

    for col_fcm in FCM_table.index:
        for col_dna in DNA_table.index:
            r, p = pearsonr(cat_table[col_fcm], cat_table[col_dna])
            p_table.loc[col_dna, col_fcm] = p
            r_table.loc[col_dna, col_fcm] = r

    p_table_stack = pd.DataFrame(p_table.stack().sort_values(), columns=["p-values"])

    return p_table, r_table, p_table_stack

def result_store(exp_id, config, p_table, r_table, p_table_stack, cat_table):
    save_dir = "output_Apr04"
    general_mkdir(save_dir)    

    sub_dirs = [
        "00__configs", "01__p_table", "02__r_table", "03__p_table_stack", "04__cat_table", 
    ]

    objects = [
        config, p_table, r_table, p_table_stack, cat_table
    ]

    for sub_dir, object in zip(sub_dirs, objects):
        general_mkdir(f"{save_dir}/{sub_dir}")
        if sub_dir == "00__configs":
            with open(f"{save_dir}/{sub_dir}/{exp_id}.json", 'w') as f:
                json.dump(object, f, indent=2)
        else:
            object.to_csv(f"{save_dir}/{sub_dir}/{exp_id}.csv")

def general_mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def main():
    for exp_id, config in enumerate(configs):
        df = get_data()
        clusters = gmm_clustering(df, config)
        FCM_table = create_FCM_table(df, clusters)
        DNA_table = create_DNA_table()
        cat_table = create_concat_table(DNA_table, FCM_table)
        p_table, r_table, p_table_stack = r_test(FCM_table, DNA_table, cat_table)
        result_store(
            exp_id, config, p_table, r_table, p_table_stack, cat_table
        )

def main_tmp():
    for exp_id, config in enumerate(configs_tmp):
        df = get_data()
        clusters = gmm_clustering(df, config)
        FCM_table = create_FCM_table(df, clusters)
        DNA_table = create_DNA_table()
        cat_table = create_concat_table(DNA_table, FCM_table)
        p_table, r_table, p_table_stack = r_test(FCM_table, DNA_table, cat_table)
        result_store(
            exp_id, config, p_table, r_table, p_table_stack, cat_table
        )
 
if __name__ == "__main__":
    # main()

    main_tmp()