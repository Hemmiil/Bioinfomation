from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd
import json
import os

def parallel_movement(X, w, n_components, alpha=0.9, random_state=0):
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    y = gmm.fit_predict(X)


    X_cp = X.copy()
    X_cp["cluster"] = y
    cols = X_cp.columns.drop("cluster")


    diff_cat_cat = pd.DataFrame()

    for i in w.sort_values().unique():
        diff_cat = pd.DataFrame()
        for j in X_cp["cluster"].sort_values().unique():
            indices = np.logical_and((w==i) , (X_cp["cluster"]==j))
            diff = X_cp[cols].loc[indices] - gmm.means_[j]
            diff_cat = pd.concat([diff_cat, diff])
        diff_cat_cat = pd.concat([
            diff_cat_cat, 
            pd.DataFrame(
                diff_cat.mean(),
                columns=[i]
            ).T
        ])

        # alpha = 0.9**alpha

        X_cp.loc[diff_cat.index, cols] = X_cp.loc[diff_cat.index, cols] - diff_cat.mean()*alpha
    cluster = X_cp["cluster"]
    return X_cp.drop("cluster", axis=1), diff_cat_cat, cluster

def setting_profile(ward, cluster):
    df_cls_ward = pd.DataFrame(
        {
            "cluster": cluster,
            "ward": ward,
            "dummy": [1 for i in range(len(cluster))]
        }
    )

    df_profile_mount = pd.pivot_table(
        df_cls_ward,
        index="cluster",
        columns="ward",
        aggfunc="sum"
    )

    df_profile = df_profile_mount["dummy"] / df_profile_mount["dummy"].sum(axis=0)
    return df_profile

def experiments(df, n_components=3, N = 2, alpha=0.5, save_dir="tmp", random_state_list=[0,1,2,3,4], is_dummy=False):
    # dfについて、GMM-Loopを実行する
    # 初期値設定
    df_cp = df.copy()

    X_init = df_cp.drop(["DATE","STATION_label","ward"], axis=1)
    w = df_cp["ward"]

    diff_dict = {}
    X_dict = {}
    cluster_dict = {}    

    X = X_init
    for random_state in random_state_list:
        diff_dict_sub = {}
        X_dict_sub = {}
        cluster_dict_sub = {}
        for n in range(N):
            X, diff, cluster = parallel_movement(X, w, alpha=alpha, n_components=n_components, random_state=random_state)
            diff_dict_sub[n] = diff
            X_dict_sub[n] = X
            cluster_dict_sub[n] = cluster

        diff_dict[random_state] = diff_dict_sub
        X_dict[random_state] = X_dict_sub
        cluster_dict[random_state] = cluster_dict_sub
    if is_dummy==False:
        # 結果の保存
        if not os.path.exists(save_dir):
            os.makedirs