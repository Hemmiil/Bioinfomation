# 実験手順

## CellSorterデータ処理

1. データの整形

`p01_CellSorter_Data.py`

2. GMM-Loop実験

`p02_GMM_loop.py`
出力：

```
result["result_type"][0][1]
```

引数１：結果の種類（X, diff, cluster)
引数２：ランダムシード
引数３：移動回数

3. 結果の可視化

## PlanDyOデータ処理

1. データの整形

`p03_PlanDyO.py`
2. ノイジング前後の変化の可視化

`p04_PlanDyO_vis.py`

## 類似性評価
1. 相関係数、p値の算出
`similarity.py`

2. 可視化
`p05_Similarity.py`
ヒートマップ：`heatmap_manual`

QQプロット：`qqplot_manual`
