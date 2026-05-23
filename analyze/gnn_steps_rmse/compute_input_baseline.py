from __future__ import annotations

import math
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from tqdm import tqdm


WORKSPACE_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = WORKSPACE_ROOT / "graphcast_pipeline/configs/config.yaml"
INPUTS_DIR = Path("/media/dl-box/SSD-SCTU3A/graphcast_data/data/processed/graphcast_inputs")
CSV_PATH = WORKSPACE_ROOT / "analyze/gnn_steps_rmse/data/gnn_steps_rmse_table.csv"

TIME_INDEX_TO_STEP = {0: -12, 1: -6}


@dataclass
class BaselineStats:
    """RMSEと入力統計量を累積するクラス"""
    sum_sq_err: float = 0.0
    pred_sum: float = 0.0
    pred_sum_sq: float = 0.0
    count: int = 0

    def update(self, mse_val: float, pred_mean: float) -> None:
        self.sum_sq_err += mse_val
        self.pred_sum += pred_mean
        self.pred_sum_sq += pred_mean ** 2
        self.count += 1

    def rmse(self) -> float:
        if self.count == 0:
            return math.nan
        return math.sqrt(self.sum_sq_err / self.count)

    def pred_mean(self) -> float:
        if self.count == 0:
            return math.nan
        return self.pred_sum / self.count

    def pred_std(self) -> float:
        if self.count == 0:
            return math.nan
        mean = self.pred_mean()
        var = (self.pred_sum_sq / self.count) - (mean ** 2)
        return math.sqrt(max(0, var))


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_datetime(dt_str: str) -> datetime:
    return datetime.strptime(dt_str, "%Y-%m-%dT%H")


def parse_dt(path: Path) -> str:
    m = re.search(r"date-([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2})", path.name)
    if not m:
        raise ValueError(f"Unexpected filename format: {path.name}")
    return m.group(1)


def list_input_files(
    inputs_dir: Path,
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None,
) -> List[Path]:
    files = sorted(inputs_dir.glob("source-era5_date-*_res-1.0_levels-13_steps-01.nc"))
    if start_dt is None and end_dt is None:
        return files

    filtered = []
    for fpath in files:
        dt_str = parse_dt(fpath)
        file_dt = parse_datetime(dt_str)
        if start_dt is not None and file_dt < start_dt:
            continue
        if end_dt is not None and file_dt > end_dt:
            continue
        filtered.append(fpath)
    return filtered


def select_time(da: xr.DataArray, time_index: int) -> xr.DataArray:
    if "time" not in da.dims:
        return da
    if da.sizes["time"] <= max(time_index, 2):
        raise ValueError(f"time size {da.sizes['time']} too small for index {time_index}")
    return da.isel(time=time_index)


def compute_baseline_vectorized(
    pred_da: xr.DataArray, truth_da: xr.DataArray, level_filter: Optional[set] = None
) -> Dict[Optional[int], Tuple[float, float]]:
    """
    Returns: Dict[level, (mse_val, pred_mean)]
    levelがない変数の場合は level=None として扱う
    """
    results = {}
    spatial_dims = [d for d in pred_da.dims if d in ("lat", "lon")]

    if "level" in pred_da.dims:
        levels = pred_da["level"].values
        if level_filter is not None:
            levels = [lvl for lvl in levels if int(lvl) in level_filter]
            if not levels:
                return results
            pred_da = pred_da.sel(level=levels)
            truth_da = truth_da.sel(level=levels)

        pred_mean = pred_da.mean(dim=spatial_dims)
        pred_mean_vals = pred_mean.mean(dim=[d for d in pred_mean.dims if d != "level"]).values

        diff = pred_da - truth_da
        mse = (diff ** 2).mean(dim=spatial_dims)
        mse_vals = mse.mean(dim=[d for d in mse.dims if d != "level"]).values

        for i, lvl in enumerate(levels):
            lvl_int = int(lvl)
            results[lvl_int] = (float(mse_vals[i]), float(pred_mean_vals[i]))
    else:
        pred_mean = pred_da.mean(dim=spatial_dims)
        pred_mean_val = pred_mean.mean().item()

        diff = pred_da - truth_da
        mse = (diff ** 2).mean(dim=spatial_dims)
        mse_val = mse.mean().item()

        results[None] = (float(mse_val), float(pred_mean_val))

    return results


def process_single_file(
    fpath: Path,
    level_filter_set: Optional[set],
) -> Dict[Tuple[str, int], Tuple[float, float]]:
    """
    Returns: Dict[(variable, step), (mse_val, pred_mean)]
    """
    results: Dict[Tuple[str, int], Tuple[float, float]] = {}

    try:
        with xr.open_dataset(fpath, decode_times=False) as ds:
            for var in ds.data_vars:
                da = ds[var]
                for time_index, step in TIME_INDEX_TO_STEP.items():
                    pred_da = select_time(da, time_index)
                    truth_da = select_time(da, -1)
                    var_results = compute_baseline_vectorized(
                        pred_da, truth_da, level_filter=level_filter_set
                    )
                    for lvl, (mse_val, pred_mean) in var_results.items():
                        if lvl is not None:
                            key = (f"{var}-level{lvl}", step)
                        else:
                            key = (var, step)
                        results[key] = (mse_val, pred_mean)
    except Exception as e:
        print(f"[error] Failed to process {fpath}: {e}")

    return results


def accumulate_stats(
    input_files: Iterable[Path],
    level_filter: Optional[List[int]] = None,
    num_workers: int = 25,
) -> Dict[Tuple[str, int], BaselineStats]:
    stats: Dict[Tuple[str, int], BaselineStats] = defaultdict(BaselineStats)
    level_filter_set = set(level_filter) if level_filter is not None else None

    input_files_list = list(input_files)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {
            executor.submit(process_single_file, fpath, level_filter_set): fpath
            for fpath in input_files_list
        }
        for future in tqdm(
            as_completed(future_to_file),
            total=len(input_files_list),
            desc="Processing files",
            unit="file",
        ):
            fpath = future_to_file[future]
            try:
                file_results = future.result()
                for key, (mse_val, pred_mean) in file_results.items():
                    stats[key].update(mse_val, pred_mean)
            except Exception as e:
                print(f"[error] Failed to process {fpath}: {e}")

    return stats


def stats_to_dataframe(stats: Dict[Tuple[str, int], BaselineStats]) -> pd.DataFrame:
    rows = []
    for (var, step), st in stats.items():
        rows.append({
            "variable": var,
            "step": step,
            "rmse": st.rmse(),
            "mae": np.nan,
            "cvrmse": np.nan,
            "pred_mean": st.pred_mean(),
            "pred_std": st.pred_std(),
            "pred_cv": np.nan,
            "truth_mean": np.nan,
            "truth_std": np.nan,
            "truth_cv": np.nan,
            "count": np.nan,
        })
    df = pd.DataFrame(rows)
    return df.sort_values(["variable", "step"])


def main() -> None:
    config = load_config(CONFIG_PATH)
    period = config.get("period", {})
    start_str = period.get("start")
    end_str = period.get("end")

    start_dt = None
    end_dt = None
    if start_str:
        start_dt = datetime.strptime(start_str, "%Y-%m-%dT%H:%M:%S")
    if end_str:
        end_dt = datetime.strptime(end_str, "%Y-%m-%dT%H:%M:%S")

    level_filter = None

    print("Config loaded:")
    if start_dt:
        print(f"  Start date: {start_dt}")
    if end_dt:
        print(f"  End date: {end_dt}")
    if level_filter:
        print(f"  Level filter: {level_filter}")
    else:
        print("  Level filter: All levels")

    input_files = list_input_files(INPUTS_DIR, start_dt=start_dt, end_dt=end_dt)
    if not input_files:
        print(f"No input files found under {INPUTS_DIR}")
        return

    print(f"\nFound {len(input_files)} input files (after filtering)")
    print(f"Using {25} parallel workers")
    stats = accumulate_stats(input_files, level_filter=level_filter, num_workers=25)
    df_baseline = stats_to_dataframe(stats)

    if CSV_PATH.exists():
        df_existing = pd.read_csv(CSV_PATH)
        df_out = pd.concat([df_existing, df_baseline], ignore_index=True)
        df_out = df_out.sort_values(["variable", "step"])
    else:
        df_out = df_baseline

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(CSV_PATH, index=False)
    print(f"Saved baseline rows to {CSV_PATH}")


if __name__ == "__main__":
    main()
