# 00__configs
- 実験条件

# 01__p_table
- インデックスに生物グループ名、カラムにクラスターID
- PlanDyOデータとセルソーターデータの相関係数のp値を記録

# 02__r_table
- インデックスに生物グループ名、カラムにクラスターID
- PlanDyOデータとセルソーターデータの相関係数を記録

# 03__p_table_stack
- 01__p_table のテーブルを積み上げ型にし、p値で昇順にソートしたデータ。p値が小さい生物グループ、クラスターの組み合わせがわかる

# 04__cat_table
- PlanDyO, CellSorterの集計後のテーブル
- インデックス名がサンプルラベル（サンプリング年月-場所大分類ID-場所小分類ID）
- カラム名が生物グループ名とクラスターID

# 05__images
- (root)-(ランダムシード)-(ブートストラップによる相関係数リストからの代表値算出方法)-(画像), (推定p値テーブル)
