- ラベルだけ書式を変更したい時
  ```
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  
  cluster_ranks = np.arange(1, 4)
  random_states = [0, 1, 2, 3, 4,]
  ward = "2024-06-s4"
  df_filtered_dict = {}
  df_dict = {}
  
  for random_state in random_states:
      path = f"output_Feb25/{ward}/r03_X_RAW/64/{random_state}/X_raw.csv"
      df = pd.read_csv(path)
  
      for cluster_rank in cluster_ranks:
          cluster = df["cluster"].value_counts().index[cluster_rank - 1]
          df_filtered = df[df["cluster"]==cluster]
          df_filtered_dict[cluster_rank] = df_filtered
  
      plt.scatter(
          df["FSC-A"],
          df["APC-A"],
          alpha=0.01,
          color="gray"
      )
      
      for cluster_rank in cluster_ranks:
  
          plt.scatter(
              df_filtered_dict[cluster_rank]["FSC-A"],
              df_filtered_dict[cluster_rank]["APC-A"],
              alpha=0.1,
          )
          plt.plot(
              [],
              [],
              marker='o',
              linestyle='',
              label=f"rank:{cluster_rank}",
              alpha=1.0
          )
      plt.legend(
          framealpha=1.0
      )
      plt.title(f"Random_state: {random_state}", fontsize=20, weight="bold")
      
      plt.xlabel("FSC-A", fontsize=20, weight="bold")
      plt.ylabel("APC-A", fontsize=20, weight="bold")
  
      plt.show()
  ```
  - 実際に表示する書式（line22~）
