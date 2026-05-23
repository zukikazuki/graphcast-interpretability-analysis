#!/usr/bin/env python
"""パイプライン Step 1 用ダウンロードスクリプト（6 時間刻み）。

内部では `graphcast_pipeline.io.era5_downloader` を呼び出すだけのラッパー。
以下のように実行することで YAML 設定に従い ERA5 を取得できます::

    python scripts/download_era5.py --config graphcast_pipeline/configs/config.yaml

期間・解像度などは YAML 側で一元管理されています。
"""
from __future__ import annotations

import argparse
import logging

from graphcast_pipeline.io.era5_downloader import Cfg, Era5Downloader


def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Download ERA5 data (6-hour step)")
    parser.add_argument("--config", default="graphcast_pipeline/configs/config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

    cfg = Cfg.from_yaml(args.config)
    downloader = Era5Downloader(cfg)
    downloader.download_range()


if __name__ == "__main__":
    main() 