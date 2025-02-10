やること

- フィルタリングしたデータを用意する
- クラスタリングする
- クラスタリング結果を反映させたクラスター別データを作成する
- クラスター別データの正規性を検証する

# p01_Normtest.py

- 正規性検証ライブラリ

## create_data
- 一様分布に従う数列 A をn個作成し、それらの各項目を足し合わせた数列 B を作成する
- 中心極限定理より、n が大きくなると数列 B は正規分布に近づく
- 環境テスト用のデータ生成

|変数名|データ型|説明|
|--|--|--|
|data_range_min|int|数列Aの一様分布の下限|
|data_range_max|int|数列Aの一様分布の上限|
|data_length|int|数列Aの長さ|
|num_sum|int|n (数列Aの個数)|

- 出力

|変数名|データ型|説明|
|--|--|--|
|x_obs|list[int]|数列B|


## norm_test
- 歪度、尖度のk統計量と検定によるp値を出力する
- ライブラリを使用している。scipy.stats.skewtest と kurtosistest

|変数名|データ型|説明|
|--|--|--|
|x_obs|list[float]|統計量計算、検定の対象となるデータ|

- 出力：辞書型

|キー名|データ型|説明|
|--|--|--|
|skewness|float|歪度のk統計量。正だと正の方向に裾が長い|
|skew_p_value|float|歪度のp値。値が０に近いと正規分布|
|kurtosis|float|尖度のk統計量。正だと中心に偏っている|
|kurtosis_p_value|float|尖度のp値。値が０に近いと正規分布|



## norm_statistics

- 尖度、歪度をスクラッチで計算する
- 定義の都合より、n<3の時にはZeroDivisionErrorが発生するため、統計量が０で出力される

|変数名|データ型|説明|
|--|--|--|
|data_range_min|int|数列Aの一様分布の下限|

- 出力：辞書型

|キー名|データ型|説明|
|--|--|--|
|skewness|float|歪度のk統計量。正だと正の方向に裾が長い|
|kurtosis|float|尖度のk統計量。正だと中心に偏っている|
|cluster_size|int|データの要素の個数|

# p02_Exp

- 正規性実験を管理するためのライブラリ
- 12月のデータに関する実験に特化している。そのため、別のサンプルについて実験するときは要調整

## Experiment

### get_data

- 12月データだけフィルタリングしてデータセットをクラス内で初期化する
- 入出力なし

### experiment

- 実験のメイン関数
  - GMMライブラリの初期化
  - データの主成分化
  - GMMクラスタリング実行

- 入力：config：辞書型

|変数名|データ型|説明|
|--|--|--|
|PCA_random_state|list[int]|PCAライブラリのランダムシード。i番目の要素はi番目の実験に適用される|
|GMM_random_state|list[int]|GMMライブラリのランダムシード。i番目の要素はi番目の実験に適用される|
|GMM_num_clusters|list[int]|GMMライブラリのクラスター数パラメータ。i番目の要素はi番目の実験に適用される|
|date|str|実験実施日の日付。自動で変更されたりしないので、手動で変更する必要あり|
|n_samples|int|実験の回数|

- 出力

|変数名|データ型|説明|
|--|--|--|
|X_cluster|pd.DataFrame|主成分テーブルに `cluster` カラムが追加されたテーブル|
|X_raw_cluster|int|生データのテーブルに `cluster` カラムが追加されたテーブル|

### output_X

- テーブルを保存する

### output_statistics

- 統計量（歪度、尖度）をjsonファイルで保存する

  


