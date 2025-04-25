from package_Dec03 import p01_CellSorter_Data
import glob
import json
import pandas as pd

class FlowCytometryProcessor(p01_CellSorter_Data.FlowCytometryProcessor):
    def _get_paths(self):
        """手動でパスを入力"""
        with open("package_Feb14/pathes_dict.json") as f:
            pathes_dict_raw = json.load(f)
        
        pathes_dict = {}

        for key in pathes_dict_raw.keys():
            pathes_dict[int(key)] = pathes_dict_raw[key]

        return pathes_dict
    
    def gate_data(self, df):
        """データをクレンジング（ゲーティング、ノイズキャンセル）"""
        return df


def main():
    instance_var = FlowCytometryProcessor(
        stations=[1,4,8,999]
    )
    return instance_var.process_all() 