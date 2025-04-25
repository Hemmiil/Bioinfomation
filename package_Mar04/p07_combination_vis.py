import pandas as pd
import re
from scipy.stats import pearsonr
import numpy as np
import random
from scipy.stats import gmean
import matplotlib.pyplot as plt
import os
from scipy.stats import t


config = {
    "rst": [0,1,2,3,4],
    "vis_num": 30,
    "n_bs": 1000
}

def get_data(rst):
    path = f"output_Apr04/04__cat_table/{rst}.csv"
    cat_table = pd.read_csv(path, index_col=0)
    pattern = r'\d+'

    FCM_condition = [re.match(pattern, content)!=None for content in cat_table.columns]
    PlanDyO_condition = [re.match(pattern, content)==None for content in cat_table.columns]
    FCM_cols = cat_table.columns[FCM_condition]
    PlanDyO_cols = cat_table.columns[PlanDyO_condition]

    return cat_table, FCM_cols, PlanDyO_cols


def bootstrap(FCM_cols, PlanDyO_cols, cat_table):
    r_mean_table_bs, r_median_table_bs, r_table_bs = pd.DataFrame(
        columns=FCM_cols,
        index=PlanDyO_cols
    ), pd.DataFrame(
        columns=FCM_cols,
        index=PlanDyO_cols
    ), pd.DataFrame(
        columns=FCM_cols,
        index=PlanDyO_cols
    )

    # ブートストラップ法。100回で実験

    # 相関係数を算出するリストが全て０だった場合をカットするようにして
    n_bs = config["n_bs"]
    for col_fcm in FCM_cols:
        for col_dna in PlanDyO_cols:

            r_direct = cat_table[[col_dna, col_fcm]].corr.iloc[0,1]
            fcm_values, dna_values = cat_table[col_fcm], cat_table[col_dna]
            k = len(fcm_values)
            rst_choices = [i for i in range(n_bs)]
            
            r_sub_list = []
            for i in range(n_bs):
                indices = np.random.choice(k, size=k, replace=True)
                dna_sample = dna_values[indices]
                fcm_sample = fcm_values[indices]
                
                # 相関係数が定義できるか確認
                if len(np.unique(dna_sample)) > 1 and len(np.unique(fcm_sample)) > 1:
                    r_sub, _ = pearsonr(fcm_sample, dna_sample)
                    r_sub_list.append(r_sub)

            # 相関係数の代表値を出力
            if len(r_sub_list)!=0:
                r_mean = np.mean(r_sub_list)
                r_median = np.median(r_sub_list)
                r_mean_table_bs.loc[col_dna, col_fcm] = r_mean
                r_median_table_bs.loc[col_dna, col_fcm] = r_median
                r_table_bs.loc[col_dna, col_fcm] = r_direct
    return r_mean_table_bs, r_median_table_bs, r_table_bs

def bootstrap2(FCM_cols, PlanDyO_cols, cat_table):
    r_mean_table_bs, r_median_table_bs, r_table_bs = pd.DataFrame(
        columns=FCM_cols,
        index=PlanDyO_cols
    ), pd.DataFrame(
        columns=FCM_cols,
        index=PlanDyO_cols
    ), pd.DataFrame(
        columns=FCM_cols,
        index=PlanDyO_cols
    )

    # ブートストラップ法。100回で実験

    # 相関係数を算出するリストが全て０だった場合をカットするようにして
    n_bs = config["n_bs"]
    for col_fcm in FCM_cols:
        for col_dna in PlanDyO_cols:

            r_direct = cat_table[[col_dna, col_fcm]].corr.iloc[0,1]
            fcm_values, dna_values = cat_table[col_fcm], cat_table[col_dna]
            k = len(fcm_values)
            rst_choices = [i for i in range(n_bs)]
            
            r_sub_list = []
            for i in range(n_bs):
                indices = np.random.choice(k, size=k, replace=True)
                dna_sample = dna_values[indices]
                fcm_sample = fcm_values[indices]
                
                # 相関係数が定義できるか確認
                if len(np.unique(dna_sample)) > 1 and len(np.unique(fcm_sample)) > 1:
                    r_sub, _ = pearsonr(fcm_sample, dna_sample)
                    r_sub_list.append(r_sub)

            # 相関係数の代表値を出力
            if len(r_sub_list)!=0:
                r_mean = np.mean(r_sub_list)
                r_median = np.median(r_sub_list)
                r_mean_table_bs.loc[col_dna, col_fcm] = r_mean
                r_median_table_bs.loc[col_dna, col_fcm] = r_median
                r_table_bs.loc[col_dna, col_fcm] = r_direct
    return r_mean_table_bs, r_median_table_bs, r_table_bs


def r_to_p(r, n):
    a = r * np.sqrt(n-2)
    b = (1-r**2)**(1/2)
    abs_t = abs(a/b)
    
    p = (-abs_t).apply(lambda t_sub: t.cdf(t_sub, df=n-2))
    return p


def vis(rst, p_table_bs, cat_table, method):
    N = config["vis_num"]
    p_table_stack = pd.DataFrame(p_table_bs.stack().sort_values(), columns=["p-values"])
    for i in range(N):
        dna_col, fcm_col = p_table_stack.index[i]
        p_value = p_table_stack.iloc[i].values
        if p_value < 1e-6:
            p_value_title = "> 1e-6"
        else:
            p_value_title = round(float(p_value), 6)
        plt.title(
            f"Predicted p-value: {p_value_title}",
            fontsize=20, weight="bold",
            )

        plt.scatter(
            cat_table[dna_col], cat_table[fcm_col]
        )
        
        for sample_label in cat_table.index:
            plt.text(
                x=cat_table.loc[sample_label, dna_col], 
                y=cat_table.loc[sample_label, fcm_col],
                s=sample_label
            )

        plt.xlim(0 - cat_table[dna_col].max()*0.05, cat_table[dna_col].max()*1.1)
        plt.ylim(0 - cat_table[fcm_col].max()*0.05, cat_table[fcm_col].max()*1.1)

        plt.xticks(fontsize=15, weight="bold", rotation=45)
        plt.yticks(fontsize=15, weight="bold",)

        plt.xlabel(dna_col, fontsize=20, weight="bold",)
        plt.ylabel(f"cluster: {fcm_col}", fontsize=20, weight="bold",)
        plt.tight_layout()

        robust_makedir(f"output_Apr04/05__images/rst_{rst}/{method}")

        p_table_stack.to_csv(f"output_Apr04/05__images/rst_{rst}/{method}/p_table.csv")
        
        plt.savefig(f"output_Apr04/05__images/rst_{rst}/{method}/rank_{i}.png")
        plt.close()

def robust_makedir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def main():
    for rst in config["rst"]:
        cat_table, FCM_cols, PlanDyO_cols = get_data(rst)
        n = int(len(cat_table))
        r_mean_table_bs, r_median_table_bs, r_table_bs = bootstrap(FCM_cols, PlanDyO_cols, cat_table)
        for method, object in zip(["mean", "mediun", "non_bs"], (r_mean_table_bs, r_median_table_bs, r_table_bs)):
            p_table_bs = object.apply(lambda r: r_to_p(r, n))
            vis(rst, p_table_bs, cat_table, method)
            print(f"rst: {rst} done")

if __name__ == "__main__":
    main()