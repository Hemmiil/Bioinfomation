import pandas as pd
import numpy as np
from scipy.stats import pearsonr

def analysis_corr(df_tax, df_gmm):
  # パラメータ
  df_concat = pd.concat([
      df_tax, df_gmm#["dummy"]
  ], axis=0).dropna(axis=1)

  length_plandyo = len(df_tax)
  length_gmm = len(df_gmm)

  p_m, r_m = np.zeros((length_plandyo,length_gmm)), np.zeros((length_plandyo, length_gmm))

  for i in range(length_plandyo):
      for j in range(-length_gmm, 0):
          r_m[i][j], p_m[i][j] = pearsonr(
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
