import random
import pandas as pd
import numpy as np
from scipy.stats import norm
import os

class Exp():
    def __init__(self):
        self.rst = 0
        self.cat_table, self.plandyo_table, self.FCM_table, self.sample_label = self.get_data()
        self.random_controls, self.random_controls_corr, self.random_controls_corr_melt = None, None, None
        self.done__crt_dataset = False

        self.bootstrap_alpha, self.bootstrap_num = None, None
        self.under_thds = None
        self.randomseed = 42

    def get_data(self):
        rst = self.rst
        path = "output_Apr14/clusters.csv"
        cluster = pd.read_csv(path, index_col=0).reset_index(drop=True)

        cat_table = pd.read_csv(f"output_Apr14/04__cat_table/{rst}.csv", index_col=0)
        plandyo_table = cat_table.drop([str(i) for i in range(64)], axis=1)
        FCM_table = cat_table[[str(i) for i in range(64)]]

        sample_label = pd.read_csv("output_Apr14/07__FCM_rawdata.csv", usecols=["ward"])
        return cat_table, plandyo_table, FCM_table, sample_label

    def crt_random_comps1(self, FCM_table):
        # シードを固定
        random.seed(self.randomseed)
        np.random.seed(self.randomseed)

        random_control_1 = pd.DataFrame(
            columns=FCM_table.columns,
            index=FCM_table.index,
            data=None
        )

        for i, cluster_id in enumerate(FCM_table.columns):
            values = FCM_table[cluster_id].values.copy()
            random.shuffle(values)
            random_control_1[cluster_id] = values
        return random_control_1

    def crt_random_comps2(self, FCM_table):
        # シードを固定
        random.seed(42)
        np.random.seed(42)

        random_control_2 = pd.DataFrame(
            columns=FCM_table.columns,
            index=FCM_table.index,
            data=None
        )

        for i, label in enumerate(FCM_table.index):
            values = FCM_table.loc[label].values.copy()
            random.shuffle(values)
            random_control_2.loc[label] = values

        return random_control_2

    def crt_random_comps3(self, FCM_table):

        random.seed(42)
        np.random.seed(42)

        FCM_table_melt = pd.melt(
            frame=FCM_table,
            var_name="cluster",
            value_name="value",
            ignore_index=False   
        ).reset_index(names="sample")

        FCM_table_melt["cluster"] = FCM_table_melt["cluster"].astype(int)
        FCM_table_melt["value"] = FCM_table_melt["value"].astype(float)

        # シャッフル
        values = FCM_table_melt["value"].values.copy()
        random.shuffle(values)
        FCM_table_melt["value"] = values

        random_control_3 = pd.pivot_table(
            FCM_table_melt,
            index="sample",
            columns="cluster",
            values="value"
        )

        return random_control_3

    def bootstrap(self, X, alpha=0.01, bootstrap_num=100):
        # ブートストラップ法で、相関係数の信頼区間（有意水準 α）を算出する
        fcm_cols, plandyo_cols = self.FCS_table.columns, self.PlanDyO_tabel.columns
        df_under_thd = pd.DataFrame(
            columns=self.fcm_cols,
            index=self.plandyo_cols,
            data=None
        )
        df_upper_thd = pd.DataFrame(
            columns=self.fcm_cols,
            index=self.plandyo_cols,
            data=None
        )
        for fcm_col in fcm_cols:
            for plandyo_col in plandyo_cols:
                r_distribution = []
                for _ in range(bootstrap_num):
                    random_indices = random.choices(X.index, k=len(cat_table))
                    r_distribution.append(cat_table[[fcm_col, plandyo_col]].loc[random_indices].corr().iloc[0,1])

                under_thd, upper_thd = np.percentile(r_distribution, 100*alpha), np.percentile(r_distribution, 100*(1-alpha))
                df_under_thd.loc[plandyo_col, fcm_col] = under_thd
                df_upper_thd.loc[plandyo_col, fcm_col] = upper_thd

        return df_upper_thd, df_under_thd
    

    def crt_dataset(self):    # データ整形
        cat_table, plandyo_table, FCM_table, sample_label = self.get_data()
        random_controls = {
            key: v for key, v in zip(
                ["sample_shuffle", "cluster_id_shuffle", "both_shuffle", "real_data"],
                [self.crt_random_comps1(FCM_table), self.crt_random_comps2(FCM_table), self.crt_random_comps3(FCM_table), FCM_table]
            )
        }
        random_controls_corr = {
            key: pd.concat([plandyo_table, obj], axis=1).corr().loc[obj.columns, plandyo_table.columns] for key, obj in random_controls.items()
        }
        random_controls_corr_melt = [
            list(value.melt()["value"].values) for value in random_controls_corr.values()
        ]        

        self.random_controls, self.random_controls_corr, self.random_controls_corr_melt = random_controls, random_controls_corr, random_controls_corr_melt
        self.done__crt_dataset = True

    def crt_dataset2(self):
        cat_table, plandyo_table, FCM_table, sample_label = self.get_data()
        random_controls = {
            key: v for key, v in zip(
                ["control",  "actual"],
                [self.crt_random_comps1(FCM_table), FCM_table]
            )
        }
        random_controls_corr = {
            key: pd.concat([plandyo_table, obj], axis=1).corr().loc[obj.columns, plandyo_table.columns] for key, obj in random_controls.items()
        }
        random_controls_corr_melt = [
            list(value.melt()["value"].values) for value in random_controls_corr.values()
        ]        

        self.random_controls, self.random_controls_corr, self.random_controls_corr_melt = random_controls, random_controls_corr, random_controls_corr_melt
        self.done__crt_dataset = True


    def prop_test(self):
        random_controls_binned = pd.DataFrame()
        bins = np.linspace(-1, 1, 20)

        # データ整形
        if not self.done__crt_dataset:
            self.crt_dataset()

        # ビニング & カウント
        for i, key in enumerate(self.random_controls_corr.keys()):
            x = pd.Series(self.random_controls_corr_melt[i])
            random_controls_binned[key] = pd.cut(x, bins=bins, labels=np.arange(1, len(bins))).value_counts().sort_index()

        # main
        test_keys_pairs = [
            ["sample_shuffle", "real_data"],
            ["cluster_id_shuffle", "real_data"],
            ["both_shuffle", "real_data"]
        ]

        df_result = pd.DataFrame(
            columns=["r_range", "method_1", "method_2", "p_values"], 
            index=[]
        )

        indice = 0

        for dist_rank, range_ in zip([4,5,7,8], ["-0.25~0.00", "0.00~0.25", "0.5~0.75", "0.75~1.00"]):
            for test_keys in test_keys_pairs:
                xs, ns = [],[]
                for test_key in test_keys:
                    obj = random_controls_binned[test_key]

                    xs.append(obj[dist_rank])
                    ns.append(obj.sum())

                p_values = self.composition_test(xs, ns)
                df_result_sub = pd.DataFrame(
                    [[range_, test_keys[0], test_keys[1], p_values]], 
                    columns=["r_range", "method_1", "method_2", "p_values"], 
                    index=[indice]
                )
                df_result = pd.concat([df_result, df_result_sub])
                indice += 1

        df_result["is_significant"] = df_result["p_values"]<0.01

        return df_result


    def composition_test(self, xs, ns):
        # 入力チェック
        assert len(xs) == 2 and len(ns) == 2, "2群のデータが必要です"

        ps = [x/n for x, n in zip(xs, ns)]
        p_pool = sum(xs) / sum(ns)

        # プール法による分散推定
        se = np.sqrt(p_pool * (1 - p_pool) * (1/ns[0] + 1/ns[1]))
        z = (ps[0] - ps[1]) / se

        # 両側検定
        p_value = 2 * norm.cdf(-abs(z))
        return round(p_value, 7)

    def bootstrap_estimate(self, X, alpha=0.01, bootstrap_num=100):
        # ブートストラップ法で、相関係数の信頼区間（有意水準 α）を算出する
        fcm_cols, plandyo_cols = self.FCM_table.columns, self.plandyo_table.columns
        cat_table = self.cat_table

        df_under_thd = pd.DataFrame(
            columns=fcm_cols,
            index=plandyo_cols,
            data=None
        )
        df_upper_thd = pd.DataFrame(
            columns=fcm_cols,
            index=plandyo_cols,
            data=None
        )
        for fcm_col in fcm_cols:
            for plandyo_col in plandyo_cols:
                r_distribution = []
                for _ in range(bootstrap_num):
                    random_indices = random.choices(X.index, k=len(cat_table))
                    r_distribution.append(cat_table[[fcm_col, plandyo_col]].loc[random_indices].corr().iloc[0,1])

                under_thd, upper_thd = np.percentile(r_distribution, 100*alpha), np.percentile(r_distribution, 100*(1-alpha))
                df_under_thd.loc[plandyo_col, fcm_col] = under_thd
                df_upper_thd.loc[plandyo_col, fcm_col] = upper_thd

        return df_upper_thd, df_under_thd
    
    def bootstrap_main(self):

        alpha, bootstrap_num = self.bootstrap_alpha, self.bootstrap_num

        if not self.done__crt_dataset:
            self.crt_dataset()
        cat_tabels = {
            k: pd.concat([self.plandyo_table, self.random_controls[k]], axis=1) for k in ["sample_shuffle", "cluster_id_shuffle", "both_shuffle"]
        }

        cat_tabels["real data"] = self.cat_table   
        
        self.under_thds = {
            k: self.bootstrap_estimate(v, alpha=alpha, bootstrap_num=bootstrap_num)[-1] for k, v in cat_tabels.items()
        }
   


    def robust_mkdir(sekf, dirpath):
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    def store_bts(self):
        dirpath = "output_Apr18/01_r_bts_under"
        self.robust_mkdir(dirpath)
        for k, v in self.under_thds.items():
            path = f"{dirpath}/{k}.csv"
            v.to_csv(path)
        
def main():
    instance = Exp()

    instance.bootstrap_alpha, instance.bootstrap_num = 0.01, 1
    instance.bootstrap_main()  
    instance.store_bts()

if __name__ == "__main__":
    main()