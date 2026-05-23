# Graphcast 予測関数の入力データ概要

Graphcast の予測ステップで利用される主な入力データセットの概要と、各データセットに含まれる変数についてまとめます。

## データセットの概要

| データセット名        | 説明                                   | 形状 (Dimensions)                                         |
| :-------------------- | :------------------------------------- | :-------------------------------------------------------- |
| `eval_inputs`         | モデルへの入力データ (過去2ステップ分の状態) | `(batch: 1, time: 2, lat: 181, lon: 360, level: 13)` |
| `eval_targets`        | 予測対象の真値データ (次の1ステップ分)   | `(batch: 1, time: 1, lat: 181, lon: 360, level: 13)` |
| `eval_forcings`       | 予測対象時刻の外部強制力データ           | `(batch: 1, time: 1, lat: 181, lon: 360)`                |

*   `batch`: バッチサイズ
*   `time`: 時間ステップ
*   `lat`: 緯度
*   `lon`: 経度
*   `level`: 気圧面レベル

## `eval_inputs` の変数

モデルの入力となる過去2ステップ分の気象状態と、静的な地理情報、時間情報が含まれます。

| 変数名                      | 説明                                       | 形状 (Dimensions)                     | 備考                               |
| :-------------------------- | :----------------------------------------- | :------------------------------------ | :--------------------------------- |
| `2m_temperature`            | 地上2m気温 (K)                             | `(batch, time, lat, lon)`             |                                    |
| `mean_sea_level_pressure`   | 平均海面気圧 (Pa)                          | `(batch, time, lat, lon)`             |                                    |
| `10m_v_component_of_wind`   | 地上10mの風のV成分 (南北風, m/s)           | `(batch, time, lat, lon)`             | 正が北向き                         |
| `10m_u_component_of_wind`   | 地上10mの風のU成分 (東西風, m/s)           | `(batch, time, lat, lon)`             | 正が東向き                         |
| `total_precipitation_6hr` | 過去6時間の総降水量 (m)                    | `(batch, time, lat, lon)`             |                                    |
| `temperature`               | 気温 (K)                                   | `(batch, time, level, lat, lon)`      | 各気圧面レベル                     |
| `geopotential`              | ジオポテンシャル (m^2/s^2)                   | `(batch, time, level, lat, lon)`      | 各気圧面レベル                     |
| `u_component_of_wind`       | 風のU成分 (東西風, m/s)                    | `(batch, time, level, lat, lon)`      | 各気圧面レベル, 正が東向き         |
| `v_component_of_wind`       | 風のV成分 (南北風, m/s)                    | `(batch, time, level, lat, lon)`      | 各気圧面レベル, 正が北向き         |
| `vertical_velocity`         | 鉛直速度 (Pa/s)                            | `(batch, time, level, lat, lon)`      | 各気圧面レベル, 負が上昇流       |
| `specific_humidity`         | 比湿 (kg/kg)                               | `(batch, time, level, lat, lon)`      | 各気圧面レベル                     |
| `toa_incident_solar_radiation` | 大気上端での入射太陽放射量 (W/m^2)       | `(batch, time, lat, lon)`             | 強制力データ (Forcing)             |
| `year_progress_sin`         | 年の進行度 (サイン成分)                    | `(batch, time)`                       | 時間的特徴量                       |
| `year_progress_cos`         | 年の進行度 (コサイン成分)                  | `(batch, time)`                       | 時間的特徴量                       |
| `day_progress_sin`          | 日の進行度 (サイン成分)                    | `(batch, time, lon)`                  | 時間的特徴量 (経度依存)            |
| `day_progress_cos`          | 日の進行度 (コサイン成分)                  | `(batch, time, lon)`                  | 時間的特徴量 (経度依存)            |
| `geopotential_at_surface`   | 地表面ジオポテンシャル (m^2/s^2)             | `(lat, lon)`                          | 静的な地形データ (Static)          |
| `land_sea_mask`             | 陸海マスク (1: 陸, 0: 海)                | `(lat, lon)`                          | 静的な地理的特徴量 (Static)        |

## `eval_targets` の変数

モデルが予測すべき、次の時間ステップにおける気象状態の真値です。

| 変数名                      | 説明                               | 形状 (Dimensions)                | 備考                               |
| :-------------------------- | :--------------------------------- | :------------------------------- | :--------------------------------- |
| `2m_temperature`            | 地上2m気温 (K)                     | `(batch, time, lat, lon)`        | 予測対象                           |
| `mean_sea_level_pressure`   | 平均海面気圧 (Pa)                  | `(batch, time, lat, lon)`        | 予測対象                           |
| `10m_v_component_of_wind`   | 地上10mの風のV成分 (南北風, m/s)   | `(batch, time, lat, lon)`        | 予測対象, 正が北向き               |
| `10m_u_component_of_wind`   | 地上10mの風のU成分 (東西風, m/s)   | `(batch, time, lat, lon)`        | 予測対象, 正が東向き               |
| `total_precipitation_6hr` | 予測時刻までの6時間総降水量 (m)    | `(batch, time, lat, lon)`        | 予測対象                           |
| `temperature`               | 気温 (K)                           | `(batch, time, level, lat, lon)` | 予測対象, 各気圧面レベル           |
| `geopotential`              | ジオポテンシャル (m^2/s^2)           | `(batch, time, level, lat, lon)` | 予測対象, 各気圧面レベル           |
| `u_component_of_wind`       | 風のU成分 (東西風, m/s)            | `(batch, time, level, lat, lon)` | 予測対象, 各気圧面レベル, 正が東向き |
| `v_component_of_wind`       | 風のV成分 (南北風, m/s)            | `(batch, time, level, lat, lon)` | 予測対象, 各気圧面レベル, 正が北向き |
| `vertical_velocity`         | 鉛直速度 (Pa/s)                    | `(batch, time, level, lat, lon)` | 予測対象, 各気圧面レベル, 負が上昇流 |
| `specific_humidity`         | 比湿 (kg/kg)                       | `(batch, time, level, lat, lon)` | 予測対象, 各気圧面レベル           |

## `eval_forcings` の変数

予測時刻における外部強制力（太陽放射）と時間情報が含まれます。これらの変数は `eval_inputs` にも含まれていますが、予測ステップでは予測時刻に対応する値が使われます。

| 変数名                      | 説明                                 | 形状 (Dimensions)           | 備考                      |
| :-------------------------- | :----------------------------------- | :-------------------------- | :------------------------ |
| `toa_incident_solar_radiation` | 大気上端での入射太陽放射量 (W/m^2) | `(batch, time, lat, lon)`   | 強制力データ              |
| `year_progress_sin`         | 年の進行度 (サイン成分)              | `(batch, time)`             | 時間的特徴量              |
| `year_progress_cos`         | 年の進行度 (コサイン成分)            | `(batch, time)`             | 時間的特徴量              |
| `day_progress_sin`          | 日の進行度 (サイン成分)              | `(batch, time, lon)`        | 時間的特徴量 (経度依存)   |
| `day_progress_cos`          | 日の進行度 (コサイン成分)            | `(batch, time, lon)`        | 時間的特徴量 (経度依存)   |
