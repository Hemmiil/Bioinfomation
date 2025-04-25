# ディレクトリ間でのpythonライブラリのインポート

## ノートブックファイルの場合
```python
import sys
import os

# dir1 のパスを sys.path に追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), '..', 'packages')))

# lib1.py から関数をインポート
from package_Apr23 import p00_tmp
p00_tmp.main()

```

## .pyファイルの場合
```
import sys
import os

# dir1 のパスを sys.path に追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'packages')))

# lib1.py から関数をインポート
from package_Apr23 import p00_tmp
p00_tmp.main()
```
