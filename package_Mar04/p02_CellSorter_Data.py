import pandas as pd
import glob
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import flowkit as fk
import json

class FlowCytometryProcessor:
    def __init__(self, stations, base_path="../CellSorter_rawdata", return_TIME=False):
        self.stations = stations
        self.base_path = base_path
        self.df = pd.DataFrame()
        self.pathes_dict = self._get_paths()
        self.return_TIME = return_TIME

    def _get_paths(self):
        """ステーション表記の表記揺れに対応"""
        """wardラベルをPlanDyOに準拠"""

        with open('package_Mar04/dir_ward_dict.json') as f:
            dir_ward_dict = json.load(f)

        pathes_dict = dict()
        for date, ward  in dir_ward_dict.items():
            pathes_dict_sub = dict()
            for station in self.stations:
                pathes = glob.glob(f"{self.base_path}/{date}/user1_Experiment*{station}*.csv")
                #pathes.extend(glob.glob(f"{self.base_path}/{date}/user1_Experiment*S{station}*.csv"))
                if len(pathes) > 0:
                    pathes_dict_sub[station] = pathes[-1]
                # else:
                    # pathes_dict_sub[station] = None
            pathes_dict[ward] = pathes_dict_sub
        return pathes_dict

    def logicle_transform(self, df, cols):
        """Logicle変換の実行"""
        for col in cols:
            param_t = 1e7
            param_m = 7
            param_r = df[col].min()
            param_w = (param_m - np.log10(param_t / abs(param_r))) / 2

            logicle_xform = fk.transforms.LogicleTransform(
                'logicle',
                param_t=param_t,
                param_w=param_w,
                param_m=param_m,
                param_a=0
            )

            df[col] = logicle_xform.apply(df[col])
        return df

    def logicle_transform_rt(self, df, cols, params=None):
        """Logicle変換の実行"""
        if params==None:
            params = {}
            for col in cols:
                param_t = 1e7
                param_m = 7
                param_r = df[col].min()
                param_w = (param_m - np.log10(param_t / abs(param_r))) / 2

                logicle_xform = fk.transforms.LogicleTransform(
                    'logicle',
                    param_t=param_t,
                    param_w=param_w,
                    param_m=param_m,
                    param_a=0
                )

                params[col] = {
                    "param_t": param_t,
                    "param_w": param_w,
                    "param_m": param_m,
                    "param_r": param_r,                
                }

                df[col] = logicle_xform.apply(df[col])
        else:
            for col in cols:
                params_sub = params[col]
                param_t = params_sub["param_t"]
                param_m = params_sub["param_m"]
                param_r = params_sub["param_r"]
                param_w = params_sub["param_w"]

                logicle_xform = fk.transforms.LogicleTransform(
                    'logicle',
                    param_t=param_t,
                    param_w=param_w,
                    param_m=param_m,
                    param_a=0
                )

                df[col] = logicle_xform.apply(df[col])
        return df, params


    def scale_data(self, df):
        """MinMaxScalerでデータをスケール"""
        mm = MinMaxScaler()
        df_scaled = pd.DataFrame(mm.fit_transform(df), columns=df.columns)
        return df_scaled

    def gate_data(self, df):
        """データをクレンジング（ゲーティング、ノイズキャンセル）"""
        df_anomaly = pd.DataFrame()
        for col in df.columns:
            lower_bound = df[col].quantile(0.01)  # 下位1%
            upper_bound = df[col].quantile(0.99)  # 上位99%
            df_anomaly[col] = np.logical_and(df[col] > lower_bound, df[col] < upper_bound)

        is_anomaly = (df_anomaly.sum(axis=1) == len(df.columns)).values
        return df[is_anomaly]
    
    def fix_colname(self, df):
        columns_Compensated = df.columns[["-Compensated" in col for col in df.columns]]
        df = df.rename(
            {col : col.replace("-Compensated", "") for col in columns_Compensated}
            , axis=1
        )
        return df


    def process_file(self, date, station, path):
        """1つのファイルを処理"""
        print(path)
        df_sub = pd.read_csv(path).drop("Index", axis=1)
        original_length = len(df_sub)

        # 不要なカラムを削除
        df_sub = df_sub.drop(["TIME", "FSC-H", "FSC-W",], axis=1)
        df_sub = df_sub[np.logical_and(df_sub["FSC-A"] > 0, df_sub["BSC-A"] > 0)]


        # 正規化
        # df_sub = self.scale_data(df_sub)

        # ゲーティング
        df_sub = self.gate_data(df_sub)

        # 採取日ラベルの追加
        df_sub["DATE"] = date

        # 採取場所ラベルの追加
        df_sub["STATION_label"] = station

        # カラム名の修正（Compensated　という記述を削除）
        df_sub = self.fix_colname(df_sub)

        dropped_length = len(df_sub)
        print(f"Station {station}, File: {path} - Dropped: {dropped_length/original_length:.2f}")

        return df_sub

    def process_all(self):
        """すべてのステーションとファイルに対してデータを処理"""
        for date, adict in self.pathes_dict.items():
            for station, path in adict.items():
                df_sub = self.process_file(date, station, path)
                self.df = pd.concat([self.df, df_sub])

        # 変換
        print(self.pathes_dict)
        cols_log_transform = ['FSC-A', 'BSC-A']
        cols_logicle = ['FITC-A', 'PE-A', "PI-A",
                        'APC-A', 'PerCP-Cy5.5-A',
                        'PE-Cy7-A']

        self.df[cols_log_transform] = self.df[cols_log_transform].apply(np.log10)
        self.df = self.logicle_transform(self.df, cols_logicle)

        df_ward = self.df[["DATE", "STATION_label"]].astype(str)
        df_ward["DATE"] = self.df["DATE"]
        self.df["ward"] = df_ward["DATE"].str.cat(df_ward["STATION_label"], sep="-")


        return self.df

def main():
    instance_var = FlowCytometryProcessor(
        stations=["S1","s1","s4","s8","hm","mm","og"]
    )
    return instance_var.process_all()