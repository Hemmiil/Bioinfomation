# memo

## 目的

- 日報、記録
- 今まではnotionに保存していたが、先生、メンバーにも見てもらいやすいようこちらに記録する

### 2025/01/06
- p29.ipynb 可視化プログラムの練習用

### 2025/01/08
- p30.ipynb
  - システムエラーの可視化スクリプト
    - 個別の可視化（done)
   
### 2025/01/09
- p30.ipynb
  - クラスタリング後のデータが正規分布に従っているかの検証
- output_Dec27/r05, output_Dec27/r06を追加
- output_Dec27の画像をmatsuに保存後、ローカル環境のデータを削除
- 明日やること：可視化プログラムをライブラリ化


### 2025/01/20

- p07に、システムエラーの可視化ライブラリを記録

- p31.ipynb
- 移動実験前後でシステムエラー評価はどのように変化するか？
  - 目的：システムエラーの概要を知り、システムエラー（以降SEと省略）の影響を抑えることができる、再現性の高い方法を検討する
  - 実験概要：以下の項目を変更し、結果を比較する
    - クラスター数（４、８、１６）（クラスタ数の増減がSEに影響を与えるか調べる）
    - 移動倍率（0.1, 0.5, 1）（移動倍率がSEに影響を与えるか調べる）
    - ランダムシード（異なるシードで３回）（再現性の検証のため）
    - 移動回数（１〜３回程度）（移動回数がSEに影響を与えるか調べる）
  - 評価方法
    - violin plot
    - オリジナル手法（長方形を並べたグラフ）

### 2025/01/21

- なぜサーバーとローカルで、生データの長さが異なっていたか？
  - 以前ローカルで修正した、 **csvファイルの重複がサーバー側で修正されていなかったため**
  - 修正し、長さが一致したことを確認した
- p07_SystemError_vis.py の微調整
- p02_GMM_loop.py のリザルト項目の追加と構成変更
