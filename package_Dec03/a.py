# df_gmmの作成
import pandas as pd
import json
def gmm(n_components):
    df = pd.read_csv("../output_Nov16/data_148.csv", index_col=0)

    # 実験区画ラベルの付与
    df_ward = df[["DATE", "STATION_label"]].astype(str)
    df_ward["DATE"] = df["DATE"].apply(lambda x: x[:-3])
    df["ward"] = df_ward["DATE"].str.cat(df_ward["STATION_label"], sep="-s")

    with open("result_Nov20/result_list_dict.json", "r") as f:
        result_list_dict = json.load(f)

    df["gmm_result"] = result_list_dict[str(n_components)][0]

    df_gmm = (pd.pivot_table(
        df,
        index="gmm_result",
        columns="ward",
        values="TIME",
        aggfunc="count"
    ).fillna(0)/len(df))

    return df_gmm