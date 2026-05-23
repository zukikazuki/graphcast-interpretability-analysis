#!/usr/bin/env python
"""
CDS API 認証テストスクリプト

1. cdsapi.Client() で認証確認
2. ERA5 の 2m気温（2023-01-01 00:00）のサンプルをダウンロード
   └ 保存先: ./data/raw/era5_sample.nc
"""

import pathlib
import cdsapi
import yaml

# プロジェクト root を config.yaml から取得
CONFIG = pathlib.Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
with CONFIG.open(encoding="utf-8") as f:
    ROOT_DIR = pathlib.Path(yaml.safe_load(f).get("root", ".")).expanduser()


def main() -> None:
    # ① クライアント生成（ここで認証が行われる）
    client = cdsapi.Client()

    # ② ダウンロード設定
    output_path = ROOT_DIR / "data/raw/era5_sample_1deg.nc"
    output_path.parent.mkdir(parents=True, exist_ok=True)  # ディレクトリ作成

    # ③ サンプルデータ取得
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": ["2m_temperature"],
            "year": "2023",
            "month": "01",
            "day": "01",
            "time": ["00:00"],
            "grid": "1/1",  # 1 度格子
            "format": "netcdf",
        },
        str(output_path),
    )

    print(f"ダウンロード完了: {output_path.resolve()}")


if __name__ == "__main__":
    main()