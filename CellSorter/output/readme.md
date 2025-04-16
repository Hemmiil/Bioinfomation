# output の内容

## output_Dec27
- [リンク](https://github.com/Hemmiil/Bioinfomation/blob/main/CellSorter/package_Dec27/result_rule/image_result_rule.md)
- GMMの誤差推定

## output_Jan20
- 確認中

## output_Feb17
- images
  - sample_label
    - clusters_n
      - 指定サンプルにおける、全体の分布に、指定クラスタの分布を重ねたFSC-APCプロット
    - umap_nonbeads
      - random_seed
        - UMAPにおける、各のクラスターの位置
        - package_Feb17をチェック
## output_Feb25
- sample_label
  - images
    - statistis_summary
      - (指標の代表値の算出方法(mean, var))
        - クラスタ個数の推移に伴うモデル指標の推移。指標はクラスタの尖度、歪度、クラスタサイズ、第一主成分の寄与率
        - 画像は重みつけなし、クラスタ数による加重平均、クラスタ数の平方根による加重平均
  - r01_statistics
    - クラスタ数
      - ランダムシード
        - クラスタの第一主成分の統計量（json）
  - r02_X_PCA
    - クラスタ数
      - ランダムシード
        - 第一、二主成分のcsvファイル
  - r_03_X_RAW
    - クラスタ数
      - ランダムシード
        - クラスタID情報が追加されたテーブル

          
