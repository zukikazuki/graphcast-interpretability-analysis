# 要件メモ（暫定）

## 目的
- 2022年の各6時間時刻について、層ごとの decoder 互換性スコアを Hutchinson 推定によって算出する。
- 出力は1行=1時刻、17列（$\ell=0..16$）のCSV。

## モデル・設定
- **モデル**: GraphCast-small
- **対象期間**: `config.yaml` の period（start/end）から取得（6時間刻み）
- **Processor 層**: $\ell = 0..16$（計17層）
  - $\ell=0$: grid2mesh\_gnn（encoder）直後の mesh latent
  - $\ell=1..16$: mesh\_gnn の各 message step 後の mesh latent（残差接続後）
- **Hutchinson サンプル数**: $K = 8$
- **乱数**: Rademacher

## 各時刻 $t$ の処理

1. encoder（grid2mesh\_gnn）を実行し、grid latent と mesh latent を得る（grid/edge は固定）。
2. 層 $\ell$ ごとに:
   - $x_\ell$（mesh latent）を用意。
   - decoder を実行し、$\hat{y}_\ell = \mathrm{Dec}(x_\ell)$ を得る（grid latent と mesh2grid edges は固定）。
   - $k = 1 \ldots K$ について:
     - 出力と同じ shape の Rademacher ベクトル $v^{(k)}$ を生成
     - VJP: $g_\ell^{(k)} = (J_{\mathrm{dec}})^\top v^{(k)}$
     - $\|g_\ell^{(k)}\|^2$ を蓄積
   - 
     $$
     S_\ell(t) = \frac{1}{K} \sum_{k=1}^{K} \|g_\ell^{(k)}\|^2
     $$

## 出力
- CSV: `time, S_l0, S_l1, ..., S_l16`
- 1行=1時刻（6時間刻み）

## 実行条件
- 1ステップ推論のみ（rolloutは使わない）
- 推論パスを改変して組み込み