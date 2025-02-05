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

import numpy as np
from math import pi, sqrt, log

def KL_D(x_obs):
    n = len(x_obs)

    mu = np.sum(x_obs)/n
    sigma = sqrt(np.sum([(x-mu)**2 for x in x_obs])/(n-1))

    def P_likelihood(x, mu, sigma):
        return np.exp(- (x - mu) / (2*sigma) ) / sqrt( 2*pi*sigma )

    P = [P_likelihood(x, mu, sigma) for x in x_obs]
    Q = [1/n for _ in range(n)]

    D_KL = np.sum([p*log(p/q) for p, q in zip(P, Q)])

    return D_KL, mu, sigma, n

### 可視化

def QQplot(x_obs, figure_limitation, is_save=False, filename=""):
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt



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

def main(x, filename, figure_limitation):
    result_dict = norm_test(x)
    D_KL, _, _, _ = D_KL(x)

    result_dict["KL_D"] = D_KL

    QQplot(x, figure_limitation=figure_limitation, is_save=True, filename=filename)

    return result_dict

