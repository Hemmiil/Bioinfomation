- ラベルだけ書式を変更したい時
  ```
  plt.legend(
      loc='upper left',
      fontsize=12,
      title='凡例のタイトル',
      title_fontsize='large',
      frameon=True,
      ncol=2,
      bbox_to_anchor=(1, 1)  # プロットの外側に凡例を表示
  )
  ```
  - 実際に表示する書式（line22~）
 
## バイオリンプロットの色を変える
```python
ax.violinplot(
    rawdata[col],
    positions=[-1],
    showmeans=False,
    showmedians=True,
    violinprops={"facecolor": "tab:orange"}
)

```
