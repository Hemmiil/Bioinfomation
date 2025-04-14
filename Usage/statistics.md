## 尖度と歪度
```
def norm_test(self, x_obs):
    ## p値が小さい＝正規性がある

    ### 歪度の検定
    skewness, skew_p_value = stats.skewtest(x_obs)

    ### 尖度の検定
    kurtosis, kurt_p_value = stats.kurtosistest(x_obs)
    return {
        "skewness": skewness,
        "skew_p_value": skew_p_value,
        "kurtosis": kurtosis,
        "kurt_p_value": kurt_p_value
    }
```
