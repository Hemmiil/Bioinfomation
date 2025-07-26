# 一般化線形モデルの結果出力フォーマット
- output
  - 01__dataset
    - X_test.csv
    - X_train.csv
    - y_test.csv
    - y_train.csv
  - 02__y_preds
    - y_pred.json
  - 03__taxonomies
    - taxonomies.json
  - 04__model_metrics
    - 01__params.json
    - 02__p_values.json
    - 03__AIC.json
    - 04__RMSE.json
    - 05__sample_error.json
  - 05__graphs
    - 01__Act_Pred_Plot
    - 02__RMSE_Bar
   
- json構成 (最低限、これが整備されていたら、GLM と NNを比較できる)
  - 02__y_preds/y_pred.json
    ```
    {
      "train": {
        "linear": {
          "taxonomy_i": {
            "sample_j": value,
            ...
            }
          }
        }
    }
    ```
  - 03__taxonomies/taxonomies.json
    ```
    [
      "taxonomy_i",
      ...
    ]
  - 04__model_metrics/04__RMSE.json
    ```
    {
      "mode_type": {
        "taxonomy": RMSE_value,
        ...
        },
      }
    ```
  - 04__model_metrics/05__sample_error.json
    ```
    {
      "mode_type": {
        "taxonomy": {
          "sample": absolute_error_value,
          ...
        },
      },
    }
    ```
