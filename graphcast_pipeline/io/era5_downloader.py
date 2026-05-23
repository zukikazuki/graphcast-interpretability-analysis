"""ERA5 ダウンロード用ユーティリティ（パイプライン Step 1）。

機能概要
---------
• **Cfg dataclass** — YAML 設定を読み込み、期間・解像度・出力パスなどを保持。
• **Era5Downloader** — 6 時間刻みで単時刻 NetCDF を CDS から取得。

使い方（スクリプト側）::

    from graphcast_pipeline.io.era5_downloader import Cfg, Era5Downloader

    cfg = Cfg.from_yaml("graphcast_pipeline/configs/config.yaml")
    Era5Downloader(cfg).download_range()

ディレクトリ名の `io` は *Input / Output* の略で、
“外部 I/O（ダウンロードや保存）を担うユーティリティ” をまとめています。
"""
from __future__ import annotations

import concurrent.futures as _futures
import dataclasses
import datetime as _dt
import logging
import pathlib
import sys
from typing import List, Tuple
import json

try:
    import yaml  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyYAML is required: pip install PyYAML") from exc

try:
    import cdsapi  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("cdsapi is required: pip install cdsapi") from exc

# Optional: retry. Tenacity is lighter than implementing our own loop.
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type  # type: ignore
except ImportError:  # pragma: no cover

    def retry(*dargs, **dkw):  # type: ignore  # noqa: D401
        """Fallback no-op decorator if tenacity is missing."""

        def _decorator(fn):
            return fn

        return _decorator

    def stop_after_attempt(*_a, **_kw):  # type: ignore
        return None

    def wait_exponential(*_a, **_kw):  # type: ignore
        return None

    def retry_if_exception_type(*_a, **_kw):  # type: ignore
        return None

try:
    from tqdm.auto import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):  # type: ignore
        """Fallback tqdm replacement if tqdm is not installed (no-op)."""
        return iterable if iterable is not None else []


# -----------------------------------------------------------------------------
# Configuration dataclass
# -----------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class Cfg:
    period_start: _dt.datetime
    period_end: _dt.datetime
    lookback_hours: int
    raw_dir: pathlib.Path
    resolution: float  # ° grid step – e.g. 1.0 or 0.25
    compute_num_workers: int = 4
    # variables
    single_levels: List[str] = dataclasses.field(default_factory=list)
    pressure_levels_variables: List[str] = dataclasses.field(default_factory=list)
    pressure_levels: List[int] = dataclasses.field(
        default_factory=lambda: [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    )
    log_dir: pathlib.Path = pathlib.Path("logs")

    # ------------------------------------------------------------------
    # YAML loader
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> "Cfg":
        path = pathlib.Path(path)
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        # Navigate keys with defaults
        period = raw.get("period", {})
        ps = _dt.datetime.fromisoformat(period.get("start"))
        pe = _dt.datetime.fromisoformat(period.get("end"))

        resolution = float(raw.get("resolution", 1.0))
        lookback = int(raw.get("lookback_hours", raw.get("lookback", 12)))

        root_var = raw.get("root") 
        storage = raw.get("storage", {})
        raw_dir_str = storage.get("raw_dir", "data/raw").format(root=root_var)
        raw_dir = pathlib.Path(raw_dir_str).expanduser()
        log_dir_str = storage.get("log_dir", "logs").format(root=root_var)
        log_dir = pathlib.Path(log_dir_str).expanduser()

        compute = raw.get("compute", {})
        workers = int(compute.get("num_workers", 4))

        variables = raw.get("variables", {})
        single_levels = variables.get("single_levels")
        pressure_vars = variables.get("pressure_levels")

        if not single_levels or not pressure_vars:
            raise ValueError(
                "[config.yaml] 'variables.single_levels' と 'variables.pressure_levels' を必ず定義してください。"
            )

        return cls(
            period_start=ps,
            period_end=pe,
            lookback_hours=lookback,
            raw_dir=raw_dir,
            resolution=resolution,
            compute_num_workers=workers,
            single_levels=single_levels,
            pressure_levels_variables=pressure_vars,
            log_dir=log_dir,
        )


# -----------------------------------------------------------------------------
# Downloader class
# -----------------------------------------------------------------------------


class Era5Downloader:
    """Download ERA5 files (single & pressure levels) 6-hourly."""

    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.client = cdsapi.Client()  # Authentication at init
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cfg.raw_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download_range(self) -> None:
        """Download all 6-hour slots covering [period_start-lookback, period_end]."""

        start_ts = self.cfg.period_start - _dt.timedelta(hours=self.cfg.lookback_hours)
        end_ts = self.cfg.period_end
        hours = int(((end_ts - start_ts).total_seconds()) // 3600)
        timestamps = [start_ts + _dt.timedelta(hours=6 * i) for i in range(hours // 6 + 1)]

        start_time = _dt.datetime.utcnow()
        self.logger.info("Total %d timestamps to download", len(timestamps))

        # ------------------------------------------------------------------
        # まず降水（tp）は 1 時間積算なので、6h 積算を作るために
        # [start_ts - 5h, end_ts] の毎時データを取得しておく
        # （後段で 6 本を合計して total_precipitation_6hr を生成）
        # ------------------------------------------------------------------
        hourly_start = start_ts - _dt.timedelta(hours=5)
        hourly_hours = int(((end_ts - hourly_start).total_seconds()) // 3600)
        hourly_timestamps = [hourly_start + _dt.timedelta(hours=i) for i in range(hourly_hours + 1)]
        self.logger.info("Downloading hourly tp for %d hours (from %s to %s)", len(hourly_timestamps), hourly_start, end_ts)

        with _futures.ThreadPoolExecutor(max_workers=self.cfg.compute_num_workers) as pool:
            # Kick off hourly tp downloads
            futures = [pool.submit(self._download_tp_hour, ts) for ts in hourly_timestamps]
            # Kick off 6-hourly slots for other variables (and tp at 6h will be skipped as file exists)
            futures += [pool.submit(self._download_hour, ts) for ts in timestamps]
            for _ in tqdm(
                _futures.as_completed(futures),
                total=len(futures),
                desc="ERA5 6h slots",
                unit="slot",
                leave=False,
            ):
                pass

        # -----------------------------------------------
        # Write summary log
        # -----------------------------------------------
        end_time = _dt.datetime.utcnow()
        files = list(self.cfg.raw_dir.rglob("*.nc"))
        total_bytes = sum(f.stat().st_size for f in files)
        duration = (end_time - start_time).total_seconds()
        summary = {
            "started_at": start_time.isoformat() + "Z",
            "ended_at": end_time.isoformat() + "Z",
            "duration_sec": duration,
            "resolution": self.cfg.resolution,
            "lookback_hours": self.cfg.lookback_hours,
            "period_start": self.cfg.period_start.isoformat(),
            "period_end": self.cfg.period_end.isoformat(),
            "num_workers": self.cfg.compute_num_workers,
            "slots": len(timestamps),
            "files": len(files),
            "bytes": total_bytes,
            "avg_mib_per_sec": round(total_bytes / 1024 / 1024 / duration, 2) if duration else None,
        }
        # ダウンロード用のログは {log_dir}/download/ 配下にまとめる
        download_log_dir = self.cfg.log_dir / "download"
        download_log_dir.mkdir(parents=True, exist_ok=True)
        log_path = download_log_dir / f"download_log_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with log_path.open("w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)
        self.logger.info("Download summary saved to %s", log_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _output_path(self, ts: _dt.datetime, *subdirs: str) -> pathlib.Path:
        ts_str = ts.strftime("%Y%m%d%H")
        base = self.cfg.raw_dir
        for sd in subdirs:
            base = base / sd
        return base / f"{ts_str}.nc"

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _download_cds(self, dataset: str, request: dict, target: str) -> None:
        """Wrapper around cdsapi to allow retries."""
        self.client.retrieve(dataset, request, target)

    def _download_hour(self, ts: _dt.datetime) -> None:
        # Split single-level variables into instant vs accum (e.g. total_precipitation)
        single_vars = list(self.cfg.single_levels)
        accum_names = {"total_precipitation"}
        instant_vars = [v for v in single_vars if v not in accum_names]
        accum_vars = [v for v in single_vars if v in accum_names]

        # Single-level instant
        if instant_vars:
            path = self._output_path(ts, "single", "instant")
            if path.exists():
                self.logger.debug("%s exists, skipping", path)
            else:
                dataset = "reanalysis-era5-single-levels"
                request = {
                    "product_type": "reanalysis",
                    "variable": instant_vars,
                    "year": ts.strftime("%Y"),
                    "month": ts.strftime("%m"),
                    "day": ts.strftime("%d"),
                    "time": ts.strftime("%H:%M"),
                    "grid": f"{self.cfg.resolution}/{self.cfg.resolution}",
                    "format": "netcdf",
                }
                path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    self.logger.info("Downloading %s -> %s", dataset, path)
                    self._download_cds(dataset, request, str(path))
                except Exception as exc:  # pragma: no cover
                    self.logger.error("Failed %s %s: %s", dataset, ts, exc)
                    raise

        # Single-level accum (e.g. total_precipitation)
        if accum_vars:
            path = self._output_path(ts, "single", "accum")
            if path.exists():
                self.logger.debug("%s exists, skipping", path)
            else:
                dataset = "reanalysis-era5-single-levels"
                request = {
                    "product_type": "reanalysis",
                    "variable": accum_vars,
                    "year": ts.strftime("%Y"),
                    "month": ts.strftime("%m"),
                    "day": ts.strftime("%d"),
                    "time": ts.strftime("%H:%M"),
                    "grid": f"{self.cfg.resolution}/{self.cfg.resolution}",
                    "format": "netcdf",
                }
                path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    self.logger.info("Downloading %s -> %s", dataset, path)
                    self._download_cds(dataset, request, str(path))
                except Exception as exc:  # pragma: no cover
                    self.logger.error("Failed %s %s: %s", dataset, ts, exc)
                    raise

        # Pressure-levels (instant)
        path = self._output_path(ts, "pressure")
        if path.exists():
            self.logger.debug("%s exists, skipping", path)
        else:
            dataset, request = self._build_request(ts, "pressure")
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                self.logger.info("Downloading %s -> %s", dataset, path)
                self._download_cds(dataset, request, str(path))
            except Exception as exc:  # pragma: no cover
                self.logger.error("Failed %s %s: %s", dataset, ts, exc)
                raise

    # ------------------------------------------------------------------
    # Request builder
    # ------------------------------------------------------------------

    def _build_request(self, ts: _dt.datetime, level: str) -> Tuple[str, dict]:
        if level == "single":
            dataset = "reanalysis-era5-single-levels"
            variables = self.cfg.single_levels
            request = {
                "product_type": "reanalysis",
                "variable": variables,
                "year": ts.strftime("%Y"),
                "month": ts.strftime("%m"),
                "day": ts.strftime("%d"),
                "time": ts.strftime("%H:%M"),
                "grid": f"{self.cfg.resolution}/{self.cfg.resolution}",
                "format": "netcdf",
            }
        else:
            dataset = "reanalysis-era5-pressure-levels"
            variables = self.cfg.pressure_levels_variables
            request = {
                "product_type": "reanalysis",
                "variable": variables,
                "pressure_level": [str(p) for p in self.cfg.pressure_levels],
                "year": ts.strftime("%Y"),
                "month": ts.strftime("%m"),
                "day": ts.strftime("%d"),
                "time": ts.strftime("%H:%M"),
                "grid": f"{self.cfg.resolution}/{self.cfg.resolution}",
                "format": "netcdf",
            }

        return dataset, request

    # ------------------------------------------------------------------
    # Hourly tp downloader
    # ------------------------------------------------------------------
    def _download_tp_hour(self, ts: _dt.datetime) -> None:
        """Download single-level total_precipitation for a specific hour.
        保存先は single/accum/{YYYYMMDDHH}.nc （1h積算; 後段で6本を合計）
        """
        path = self._output_path(ts, "single", "accum")
        if path.exists():
            self.logger.debug("%s exists (tp hourly), skipping", path)
            return
        dataset = "reanalysis-era5-single-levels"
        request = {
            "product_type": "reanalysis",
            "variable": ["total_precipitation"],
            "year": ts.strftime("%Y"),
            "month": ts.strftime("%m"),
            "day": ts.strftime("%d"),
            "time": ts.strftime("%H:%M"),
            "grid": f"{self.cfg.resolution}/{self.cfg.resolution}",
            "format": "netcdf",
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.logger.info("Downloading hourly tp %s -> %s", ts, path)
            self._download_cds(dataset, request, str(path))
        except Exception as exc:  # pragma: no cover
            self.logger.error("Failed hourly tp %s: %s", ts, exc)
            raise


# -----------------------------------------------------------------------------
# Script helper
# -----------------------------------------------------------------------------


def _main() -> None:  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Download ERA5 data (6-hourly)")
    parser.add_argument(
        "--config", default="graphcast_pipeline/configs/config.yaml", help="Path to YAML config"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

    cfg = Cfg.from_yaml(args.config)
    downloader = Era5Downloader(cfg)
    downloader.download_range()


if __name__ == "__main__":  # pragma: no cover
    _main() 