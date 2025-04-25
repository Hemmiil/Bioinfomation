# データ作成
from package_Dec03 import tax_filter
import pandas as pd
import numpy as np
import os
import math
from scipy.stats import t
import matplotlib.pyplot as plt
import seaborn as sns
import json
import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gmean
# データフレーム df_noise_gmean と df_raw_gmean が既に定義されていると仮定

def noise_compare(df_noise, df_raw, suptitle="Sample"):
    if "index" in df_noise.columns:
        df_noise_gmean = df_noise.set_index("index").apply(gmean, axis=1)
        df_raw_gmean = df_raw.set_index("index").apply(gmean, axis=1)
    else:
        df_noise_gmean = df_noise.apply(gmean, axis=1)
        df_raw_gmean = df_raw.apply(gmean, axis=1)

    fig = plt.figure(figsize=(8, 4))
    axis = fig.subplots(1, 2)

    # 軸のタイトルやラベルを設定
    axis[0].hist(
        df_noise_gmean, 
        bins=[10**(i) for i in np.linspace(-10, 0, 50)],
        label="with noise",
    )
    axis[0].set_title('Noise Added Data')
    axis[0].set_xscale('log')
    axis[0].set_xlabel('Composition Rate')
    axis[0].set_ylabel('Frequency')
    axis[0].legend()

    axis[1].hist(
        df_raw_gmean[df_raw_gmean != 0], 
        bins=[10**(i) for i in np.linspace(-10, 0, 50)],
        label="raw",
    )
    axis[1].set_title('Raw Data')
    axis[1].set_xscale('log')
    axis[1].set_xlabel('Composition Rate')
    axis[1].set_ylabel('Frequency')
    axis[1].legend()

    fig.suptitle(suptitle, fontweight="bold")
    plt.show()
