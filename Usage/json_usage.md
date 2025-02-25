# JSONファイルの扱い

## JSONファイルの読み取り

```
with open('data/src/test.json') as f:
    d = json.load(f)

print(d)
# {'A': {'X': 1, 'Y': 1.0, 'Z': 'abc'}, 'B': [True, False, None, nan, inf]}

print(type(d))
```

## JSONファイルの保存

```
d = {
    'A': {'X': 1, 'Y': 1.0, 'Z': 'abc'},
    'B': [True, False, None, float('nan'), float('inf')]
}

with open('data/temp/test.json', 'w') as f:
    json.dump(d, f, indent=2)
```
