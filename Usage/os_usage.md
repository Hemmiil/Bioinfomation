# `os` ライブラリでよく使うフレーズ

## ディレクトリ作成
```
import os

def robust_makedir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

```
