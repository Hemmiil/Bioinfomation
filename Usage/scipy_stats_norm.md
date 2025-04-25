# 統計モデルの分布に関するモジュール、関数まとめ

正規分布のモジュール名は `norm` であり，以下が主な関数です。

## 確率密度関数： `norm.pdf(x, loc=0, scale=1)`

- `pdf` は Probability Density Function の頭文字
- `loc` = 平均
- `scale` = 標準偏差
- `x` = から の間の値

**返り値**： `x` の値が発生する確率（％）

`loc` と `scale` を省略すると、標準正規分布の確率密度関数となります。

---

## 累積分布関数： `norm.cdf(x, loc=0, scale=1)`

- `cdf` は Cumulative Distribution Function の頭文字
- `loc` = 平均
- `scale` = 標準偏差
- `x` = から の間の値

**返り値**： `x` 以下の値が発生する確率（％）

`loc` と `scale` を省略すると、標準正規分布の累積分布関数となります。

---

## パーセント・ポイント関数： `norm.ppf(a, loc=0, scale=1)`

- `ppf` は Percent Point Function の頭文字
- `loc` = 平均
- `scale` = 標準偏差
- `a` = 0 ~ 1 の間の値

**返り値**： 累積分布関数の値が `a` である場合の `x` の値（累積分布関数の逆関数）

`loc` と `scale` を省略すると、標準正規分布のパーセント・ポイント関数となります。

---

## ランダム変数生成関数： `norm.rvs(loc=0, scale=1, size=1)`

- `rvs` は Random VariableS の大文字の部分
- `loc` = 平均
- `scale` = 標準偏差
- `size` = 生成されるランダム変数の数

**返り値**： 正規分布に従って発生したランダム変数

`loc` と `scale` を省略すると、標準正規分布のランダム変数生成関数となります。
