#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

# Ensure repo root is on sys.path when running as a script
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from graphcast_pipeline.transform.era5_to_graphcast import (
    TransformConfig,
    Era5ToGraphcastTransformer,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shape ERA5 raw data into GraphCast inputs (cache-like)")
    p.add_argument("--config", default="graphcast_pipeline/configs/config.yaml")
    p.add_argument("--start", required=False, help="Start date YYYY-MM-DD (overrides config period.start)")
    p.add_argument("--end", required=False, help="End date YYYY-MM-DD (overrides config period.end)")
    p.add_argument("--out-dir", required=False, help="Output directory (default: config.storage.processed_dir or {root}/data/processed/graphcast_inputs)")
    p.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    return p.parse_args()


def load_yaml(path: Path) -> dict:
    import yaml  # type: ignore
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def daterange(start_date: str, end_date: str) -> List[str]:
    import pandas as pd  # type: ignore
    dates = pd.date_range(start_date, end_date, freq="1D")
    return [d.strftime("%Y-%m-%d") for d in dates]


def anchors_6h(start_date: str, end_date: str) -> List[str]:
    import pandas as pd  # type: ignore
    rng = pd.date_range(start_date, end_date, freq="6H", inclusive="both", tz="UTC")
    return [d.strftime("%Y-%m-%dT%H") for d in rng]


def timestamps_for_anchor(anchor_str: str) -> list[str]:
    # Given anchor like YYYY-MM-DDTHH (UTC), return [t-12h, t-6h, t] as YYYYMMDDHH
    import pandas as pd  # type: ignore
    t = pd.to_datetime(anchor_str + ":00:00Z")
    ts = [t - pd.Timedelta(hours=12), t - pd.Timedelta(hours=6), t]
    return [x.strftime("%Y%m%d%H") for x in ts]


def main() -> int:
    args = parse_args()

    cfg_yaml = load_yaml(Path(args.config))
    root = Path(cfg_yaml.get("root"))
    raw_dir = Path(cfg_yaml.get("storage", {}).get("raw_dir", "{root}/data/raw/era5").format(root=str(root)))
    processed_dir = Path(args.out_dir) if args.out_dir else Path(cfg_yaml.get("storage", {}).get("processed_dir", "{root}/data/processed/graphcast_inputs").format(root=str(root)))
    resolution = float(cfg_yaml.get("resolution", 1.0))
    levels = int(cfg_yaml.get("levels", 13))

    period = cfg_yaml.get("period", {})
    start = args.start or period.get("start")
    end = args.end or period.get("end")
    if not start or not end:
        raise ValueError("start/end date is required (via config or CLI)")

    tcfg = TransformConfig(
        root=root,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        resolution=resolution,
        levels=levels,
    )
    transformer = Era5ToGraphcastTransformer(tcfg)

    processed_dir.mkdir(parents=True, exist_ok=True)

    for anchor in anchors_6h(start, end):
        out_path = transformer.output_path_for_anchor(anchor)
        if out_path.exists() and not args.force:
            print(f"skip exists: {out_path}")
            continue
        ts = timestamps_for_anchor(anchor)
        ds = transformer.build_dataset(ts)
        print(f"writing: {out_path}")
        ds.to_netcdf(out_path)
        try:
            ds.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


