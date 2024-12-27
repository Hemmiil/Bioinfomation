# 実験手順

## CellSorterデータ処理

### 1. データの整形

`p01_CellSorter_Data.py`

使用例
```
import p01_CellSorter_Data
p01 = p01_CellSorter_Data.FlowCytometryProcessor(
    stations=[1,4,8], 
    base_path="/Users/henmi_note/Desktop/CellSorter/CellSorter_rawdata", 
    return_TIME=False
)

data = p01.process_all()
```
`p01_CellSorter_Data.FlowCytometryProcessor.process_all`
入力
|引数名|データタイプ|説明|
|---|---|---|
|stations|list[int]|データに加えるステーションを指定する|
|base_path|str|セルソーターデータが格納されているディレクトリ。直下に月別でcsvファイルが格納されたディレクトリを格納する|
|return_TIME|bool|測定時刻のカラムを出力。Trueの場合、`TIME`カラムが追加される|

出力
|引数名|データタイプ|説明|
|---|---|---|
|df_cellsorter|pandas.DataFrame|logicle変換済みを含む前処理済みのセルソーターデータ|
- カラムについて
  - `DATE`
    - データの測定日
  - `STATION_label`
    - ステーションID
  - `ward`
    - データの測定日とステーションIDを結合したラベル。PlanDyOデータとの表記揺れを解消した表記方法
    - 以降、**実験区画**と表記する
    - 例：
      - `2024-05-s1`




### 2. GMM-Loop実験

`p02_GMM_loop.py`

`p02_GMM_loop.experiments`
入力

|引数名|データタイプ|説明|
|---|---|---|
|df|pandas.DataFrame|セルソーター解析データ|
|N|int|移動実験における移動回数|
|n_components|int|移動実験におけるクラスター数|
|save_dir|str|結果を保存するディレクトリ名|
|alpha|int|移動実験における移動距離の倍率|
|random_state_list|list[int]|移動実験におけるランダムシード。リストの長さだけランダムシードを変更して実行される|
|is_dummy|bool|もしTrueの場合、実験結果が保存されない（セットアップ用）|

出力
|引数名|データタイプ|説明|
|---|---|---|
|result|dict|実験結果。後述|

- `result` について
```
result["result_type"][random_state][n_move]
```

- `result_type`：結果の種類（X, diff, cluster, profile)
- `random_state`：ランダムシード
- `n_move`：移動回数

### 3. 結果の可視化

(編集中)

## PlanDyOデータ処理

### 1. データの整形

`p04_PlanDyO.py`

`p03_PlanDyO.make_contig_csv`
- 慣例的分類方法における、生物群の時空間プロファイルをcsv形式で出力する
- オリジナルデータはこちら https://plandyo.jp/2024-12-26/download/Kraken.upper_group-sample.txt

出力
|引数名|データタイプ|説明|
|---|---|---|
|df_upper|pandas.DataFrame|テーブル形式のプロファイル。後述|

- 構成
  - Index: 生物群
  - Columns: 実験区画

`p03_PlanDyO.add_noise`
- ノイズを付与して、微細なシステムエラーの影響を抑制する

|引数名|データタイプ|説明|
|---|---|---|
|df_upper|pandas.DataFrame|`make_contig_csv`で出力したファイルにノイズを付与する|
  


### 2. ノイジング前後の変化の可視化

`p05_PlanDyO_vis.py`
`noise_compare`
- ノイズ付与前後のデータを比較する
- 異なる測定日、同じ生物群の存在割合から幾何平均を算出し、ヒストグラムとして出力する

|引数名|データタイプ|説明|
|---|---|---|
|df_noise|pandas.DataFrame|ノイズ付加後のPlanDyOデータ|
|df_raw|pandas.DataFrame|ノイズ付加前のPlanDyOデータ|
|suptitle|str|グラフのタイトル|

- 現在は単に画像をshowするだけで保存はしない。要編集


## 類似性評価

### 1. 相関係数、p値の算出
`similarity.py`

`analysis_corr`

|引数名|データタイプ|説明|
|---|---|---|
|df_tax|pandas.DataFrame|PlanDyOデータ|
|df_gmm|pandas.DataFrame|CellSorterデータ|

出力

|変数名|データタイプ|説明|
|---|---|---|
|df_result|pandas.DataFrame|類似性評価テーブル。後述|
|p_m|pandas.DataFrame|相関係数のp値のテーブル|
|r_m|pandas.DataFrame|相関係数のテーブル|

- df_result
```
	Groups	gmm_clusters	p-values	r-statics
25	Archaea	1	0.000199	0.974391
48	Archaea	2	0.000825	0.954558
2	Archaea	0	0.008229	0.884188
29	Bacteria:CFB group	1	0.027216	0.809967
24	Bacteria:alpha-proteobacteria	1	0.029576	0.803234
...	...	...	...	...
23	Metazoa:Bony fishes	1	0.945459	0.032144
21	Eukaryota:Green plants	0	0.952953	0.027724
11	Bacteria:Cyanobacteriota	0	0.954054	-0.027074
10	Eukaryota:Cryptomonads	0	0.966087	-0.019980
37	Bacteria:gamma-proteobacteria	1	0.991273	0.005141
```




### 2. 可視化
`p05_Similarity.py`
ヒートマップ：`heatmap_manual`

QQプロット：`qqplot_manual`