# datetime座標の問題分析結果

## 問題の概要

`/media/dl-box/SSD-SCTU3A/graphcast_data/data/processed/graphcast_inputs/source-era5_date-2022-01-01T12_res-1.0_levels-13_steps-01.nc`を使用した際に、以下のエラーが発生：

```
ValueError: 'datetime' must be in `data` coordinates.
```

## 根本原因

### 1. ファイル構造の違い

**Cacheファイル（動作する）:**
- `datetime` が **`coords`** に存在
- `datetime` coords の形状: `(batch, time)` = `(1, 3)`
- `datetime` coords の次元: `('batch', 'time')`
- データ型: `datetime64[ns]`

**Processedファイル（エラー）:**
- `datetime` が **`data_vars`** に存在
- `datetime` data_var の形状: `(batch, time)` = `(1, 3)`
- `datetime` data_var の次元: `('batch', 'time')`
- データ型: `int32` (時間オフセット: [0, 6, 12])

### 2. 変換処理の問題

`excute_graphcast.py`の250-281行目の変換処理で、以下の問題が発生：

1. **265-266行目**: `datetime_offset[0]` で`batch`次元を削除
   ```python
   if datetime_offset.ndim == 2:
       datetime_offset = datetime_offset[0]  # batch次元を削除
   ```

2. **276行目**: `batch`次元なしで`datetime` coordsを作成
   ```python
   datetime_coord = xarray.DataArray(datetime_vals, dims=("time",))  # batch次元がない！
   ```

3. **結果**: `datetime` coords が `(time,)` の1次元になる
   - Cacheファイル: `(batch, time)` = `(1, 3)`
   - Processedファイル（変換後）: `(time,)` = `(3,)`

### 3. エラーの発生箇所

`add_derived_vars`関数（`graphcast/data_utils.py`の135-178行目）が呼ばれた際：

1. `datetime` coords は存在するが、次元が `(time,)` の1次元
2. データセットには`batch`次元が存在する（`{'batch': 1, 'time': 3, ...}`）
3. `featurize_progress`関数が`batch`次元を期待しているが、`datetime` coords が1次元のため次元不一致でエラー
   ```
   ValueError: Number of feature dimensions (2) must be equal to the number of data dimensions: 1.
   ```

## 詳細な比較

| 項目 | Cacheファイル | Processedファイル（変換前） | Processedファイル（変換後） |
|------|--------------|---------------------------|---------------------------|
| `datetime`の場所 | `coords` | `data_vars` | `coords` |
| `datetime`の形状 | `(1, 3)` | `(1, 3)` | `(3,)` |
| `datetime`の次元 | `('batch', 'time')` | `('batch', 'time')` | `('time',)` |
| `datetime`の型 | `datetime64[ns]` | `int32` | `datetime64[ns]` |
| データセットの`batch`次元 | あり | あり | あり |
| `add_derived_vars`の結果 | ✅ 成功 | - | ❌ 失敗 |

## 問題の本質

**Processedファイルには`batch`次元があるのに、`datetime` coords を追加する際に`batch`次元を考慮していないため、次元の不一致が発生している。**

Cacheファイルでは`datetime` coords が`(batch, time)`の2次元で保存されているが、Processedファイルの変換処理では`(time,)`の1次元で作成されている。

## 解決方法（参考）

`excute_graphcast.py`の276行目を修正して、`batch`次元を含める必要がある：

```python
# 現在のコード（問題あり）
datetime_coord = xarray.DataArray(datetime_vals, dims=("time",))

# 修正案
if "batch" in example_batch.dims:
    # batch次元がある場合、2次元で作成
    datetime_vals_2d = [datetime_vals]  # shape: (1, 3)
    datetime_coord = xarray.DataArray(datetime_vals_2d, dims=("batch", "time"))
else:
    # batch次元がない場合
    datetime_coord = xarray.DataArray(datetime_vals, dims=("time",))
```

ただし、ユーザーの指示によりコードは変更していません。

