def create_data(data_range_min, data_range_max, data_length, num_sum ):
    import random
    import numpy as np

    lst = list(range(data_range_min, data_range_max))
    
    data = np.array([0 for i in range(data_length)])
    for _ in range(num_sum):
        data = data + np.array(random.choices(lst, k=data_length))
    x_obs = data / num_sum
    return x_obs

def norm_test(x_obs):
    ### 歪度の検定
    import scipy.stats as stats

    skewness, skew_p_value = stats.skewtest(x_obs)

    ### 尖度の検定
    kurtosis, kurt_p_value = stats.kurtosistest(x_obs)
    return {
        "skewness": skewness,
        "skew_p_value": skew_p_value,
        "kurtosis": kurtosis,
        "kurt_p_value": kurt_p_value
    }

def norm_statistics(x_obs):
    from math import sqrt
    n = len(x_obs)
    if n > 3:
        mu = sum(x_obs) / n
        s = sqrt(sum([(v - mu) ** 2 for v in x_obs]) / (n - 1))

        skewness = n/((n-1)*(n-2)) * sum([((v-mu)/s)**3 for v in x_obs])
        kurtosis = (n*(n+1))/((n-1)*(n-2)*(n-3)) * sum([((v-mu)/s)**4 for v in x_obs]) - (3*(n-1)**2) / ((n-2)*(n-3))

        return {
            "skewness": skewness,
            "kurtosis": kurtosis,
            "cluster_size": n,
            "zero_division": False,
        }
    else:
        return {
            "skewness": 0,
            "kurtosis": 0,
            "cluster_size": n,
            "zero_division": True
        }
    
import numpy as np
from math import pi, sqrt, log

import numpy as np
from math import sqrt, log, pi

def KL_D(x_obs):
    from scipy.stats import norm
    n = len(x_obs)

    if n <= 1:
        return None, None, None, n  # データ数が1以下の場合は計算不能

    mu = np.sum(x_obs) / n
    sigma = np.sqrt(np.sum([(x - mu) ** 2 for x in x_obs]) / (n - 1))  # 不偏標準偏差

    bin_width = 0.01

    m, M = min(x_obs), max(x_obs)
    def f(x):
        v_range = np.arange(m, M+bin_width, bin_width)
        for i in range(len(v_range)-1):
            if v_range[i] <= x and x < v_range[i+1]+bin_width:
                return v_range[i]

    adict = [
        f(v) for v in x_obs
    ]

    norm_instance = norm(loc=mu, scale=sigma)
    
    def P_likelihood(bin_start, bin_end):
        y_cdf_start = norm_instance.cdf(bin_start)
        y_cdf_end = norm_instance.cdf(bin_end)
        return y_cdf_end - y_cdf_start

    P = np.array([P_likelihood(adict[i], adict[i]+bin_width) for i in range(len(x_obs))]) # 累積分布関数の計算
    Q = np.full(n, 1 / n)  # 離散一様分布を想定


    # KL ダイバージェンス計算
    D_KL = np.sum([q * log(q / p) for p, q in zip(P, Q)])

    return D_KL, mu, sigma, n

### 可視化

def QQplot(x_obs, figure_limitation, is_save=False, filename=""):
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt


    n = len(x_obs)
    mu = np.sum(x_obs)/n
    sigma = sqrt(np.sum([(x-mu)**2 for x in x_obs])/(n-1))

    model = stats.norm(loc=mu, scale=sigma)
    x_exp = model.rvs(size=n)


    plt.plot(
        [figure_limitation[0], figure_limitation[1]],
        [figure_limitation[0], figure_limitation[1]],
        color="black", linestyle="dashed",
        zorder=1
    )

    plt.scatter(
        np.sort(x_exp), np.sort(x_obs),
        zorder=2
    )

    plt.xlabel("Expected data", weight="bold", fontsize=20)
    plt.ylabel("Observed data", weight="bold", fontsize=20)

    plt.xlim(figure_limitation[0], figure_limitation[1])
    plt.ylim(figure_limitation[0], figure_limitation[1])

    if is_save:
        plt.savefig(filename)
    
    plt.show()

def main(x):
    result_dict = norm_statistics(x)
    D_KL_v, _, _, _ = KL_D(x)

    result_dict["KL_D"] = D_KL_v

    # QQplot(x, figure_limitation=figure_limitation, is_save=True, filename=filename)

    return result_dict

