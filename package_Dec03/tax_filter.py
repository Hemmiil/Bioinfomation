import pandas as pd
import numpy as np
import os
import json
from scipy.stats import norm, lognorm

def make_contig_csv():
    pathes = [
        "../data/10.contig-domain4.txt",
        "../data/11.contig-phylum.txt",
        "../data/11.contig-order.txt",
        "../data/11.contig-family.txt",
        "../data/11.contig-genus.txt",
        "../data/11.contig-species.txt",
    ]
    keys = [
        "domain",
        "phylum",
        "order",
        "family",
        "genus",
        "species"
    ]

    df_dict = {}
    for key, path in zip(keys, pathes):
        with open(path, "r") as f:
            data = f.read()

        a = data.split("\n")
        b = [v.split("\t") for v in a][:-1]

        if key == "domain":
            cols = ["Contig",key,0,1,2,3,4]
        else:
            cols = ["Contig","Score","Taxid",key]
        df = pd.DataFrame(b, columns=cols, index=np.arange(len(b)))[["Contig", key]]
        df_dict[key] = df

    df_merge = df_dict[keys[0]]

    for key in keys[1:]:
        df_merge = pd.merge(df_merge, df_dict[key], how="outer")
    df_merge.to_csv(
        "../data/contig-tax.csv"
    )


def contig_composition(with_noise=False):
    path = "../data/03.contig-sample.after_total_output_norm.txt"
    with open(path, "r") as f:
        data = f.read()

    a = data.split("\n")
    b = [v.split("\t") for v in a][:-1]

    cols = ["index"] + b[0]

    df = pd.DataFrame(b[1:], columns=cols, index=np.arange(len(b)-1))

    df = df.set_index("index").astype(float)

    new_cols = [
    '2024-05-s1',
    '2024-05-s4',
    '2024-06-s4',
    '2024-06-s8',
    #'2024-07-s1',
    '2024-07-s4',
    '2024-08-s1',
    '2024-08-s4',]

    df_composition = (df/df.sum()[new_cols]).dropna(axis=1)

    if with_noise:
        df_noise = pd.DataFrame(
            lognorm.rvs(s=2, loc=0, scale=1e-5, size=df_composition.shape, random_state=0),
            index=df_composition.index,
            columns=df_composition.columns
        ).abs()
        df_composition = df_composition + df_noise
    
        df_composition.to_csv("../data/contig-composition_with_noise.csv")
        return df_composition
    else:
        df_composition.to_csv("../data/contig-composition.csv")
        return df_composition
        

def tax_filter(tax_key, threashold=100, with_noise=False):

    if with_noise:
        tax_path = "../data/contig-composition_with_noise.csv"
    else:
        tax_path = "../data/contig-composition.csv"

    df_merge = pd.read_csv("../data/contig-tax.csv", index_col=0)
    df_composition = pd.read_csv(tax_path, index_col=0)

    a = df_merge[["Contig", tax_key]].set_index("Contig")
    c = pd.concat([df_composition, a], axis=1).dropna()
    df_composition_merged = c.groupby(tax_key).sum()

    # 幾何平均存在量の上位threashold%を採択
    from scipy.stats import gmean
    df_gmean = df_composition_merged.apply(gmean, axis=1)
    valid_indices = df_gmean.sort_values().iloc[int(len(df_gmean)*threashold/100):]
    df_composition_merged = df_composition_merged.loc[valid_indices.index]

    return df_composition_merged

def gmm(n_components, random_state_id=0):
    df = pd.read_csv("../output_Nov16/data_148.csv", index_col=0)

    # 実験区画ラベルの付与
    df_ward = df[["DATE", "STATION_label"]].astype(str)
    df_ward["DATE"] = df["DATE"].apply(lambda x: x[:-3])
    df["ward"] = df_ward["DATE"].str.cat(df_ward["STATION_label"], sep="-s")

    with open("result_Nov20/result_list_dict.json", "r") as f:
        result_list_dict = json.load(f)

    df["gmm_result"] = result_list_dict[str(n_components)][random_state_id]

    df_gmm = (pd.pivot_table(
        df,
        index="gmm_result",
        columns="ward",
        values="TIME",
        aggfunc="count"
    ).fillna(0)/len(df))

    return df_gmm

# 相関係数の検定
# 最終成果物は、p-valueが0.05より小さい組み合わせ群
import pandas as pd
from scipy import stats
import numpy as np

def analysis_corr(df_tax, df_gmm):
  # パラメータ
  df_concat = pd.concat([
      df_tax, df_gmm
  ]).dropna(axis=1)

  length_plandyo = len(df_tax)
  length_gmm = len(df_gmm)

  p_m, r_m = np.zeros((length_plandyo,length_gmm)), np.zeros((length_plandyo, length_gmm))

  for i in range(length_plandyo):
      for j in range(-length_gmm, 0):
          r_m[i][j], p_m[i][j] = stats.pearsonr(
              df_concat.iloc[i], df_concat.iloc[j]
          )
          
  p_m, r_m = pd.DataFrame(p_m, index=df_tax.index), pd.DataFrame(r_m, index=df_tax.index)

  p_m_tmp = p_m.reset_index(names="Groups")
  r_m_tmp = r_m.reset_index(names="Groups")

  df_p_long = p_m_tmp.melt(
    id_vars="Groups", 
    value_vars=np.arange(length_gmm),
    var_name="gmm_clusters",
    value_name="p-values" 
    )

  df_r_long = r_m_tmp.melt(
    id_vars="Groups", 
    value_vars=np.arange(length_gmm),
    var_name="gmm_clusters",
    value_name="r-statics" 
    )

  df_result = pd.merge(df_p_long, df_r_long, how="inner").sort_values("p-values")

  # df_result["is_significant"] = df_result["p-values"]<(p_value_alpha/len(df_result))
  return df_result, p_m, r_m

def analysis_corr_all(df_tax, df_gmm):
  # パラメータ
  df_concat = pd.concat([
      df_tax, df_gmm
  ]).dropna(axis=1)

  
  length_concat = len(df_concat)

  p_m, r_m = np.zeros((length_concat,length_concat)), np.zeros((length_concat, length_concat))

  for i in range(len(df_concat)):
      for j in range(i+1, len(df_concat)):
          r_m[i][j], p_m[i][j] = stats.pearsonr(
              df_concat.iloc[i], df_concat.iloc[j]
          )
          
  p_m, r_m = pd.DataFrame(p_m, index=df_concat.index), pd.DataFrame(r_m, index=df_concat.index)
  
  p_m_tmp = p_m.reset_index(names="Groups")
  r_m_tmp = r_m.reset_index(names="Groups")

  df_p_long = p_m_tmp.melt(
    id_vars="Groups", 
    value_vars=df_concat.index.values,
    var_name="gmm_clusters",
    value_name="p-values" 
    )

  df_r_long = r_m_tmp.melt(
    id_vars="Groups", 
    value_vars=df_concat.index.values,
    var_name="gmm_clusters",
    value_name="r-statics" 
    )

  df_result = pd.merge(df_p_long, df_r_long, how="inner").sort_values("p-values")

  # df_result["is_significant"] = df_result["p-values"]<(p_value_alpha/len(df_result))
  return df_result, p_m, r_m