from __future__ import annotations

import math
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import yaml
from tqdm import tqdm


WORKSPACE_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = WORKSPACE_ROOT / "graphcast_pipeline/configs/config.yaml"
FORECASTS_DIR = Path("/media/dl-box/SSD-SCTU3A/graphcast_data/outputs/forecasts")
TRUTH_DIR = Path("/media/dl-box/SSD-SCTU3A/graphcast_data/data/processed/graphcast_inputs")
FIG_DIR = WORKSPACE_ROOT / "analyze/gnn_steps_rmse/figs"
CSV_PATH = WORKSPACE_ROOT / "analyze/gnn_steps_rmse/data/gnn_steps_rmse_table.csv"


@dataclass
class Stats:
    """統計量を累積するクラス"""
    sum_sq_err: float = 0.0  # MSE用
    sum_abs_err: float = 0.0  # MAE用
    pred_sum: float = 0.0
    pred_sum_sq: float = 0.0
    truth_sum: float = 0.0
    truth_sum_sq: float = 0.0
    count: int = 0

    def update(self, mse_val: float, mae_val: float, pred_mean: float, truth_mean: float) -> None:
        self.sum_sq_err += mse_val
        self.sum_abs_err += mae_val
        self.pred_sum += pred_mean
        self.pred_sum_sq += pred_mean ** 2
        self.truth_sum += truth_mean
        self.truth_sum_sq += truth_mean ** 2
        self.count += 1

    def rmse(self) -> float:
        if self.count == 0:
            return math.nan
        return math.sqrt(self.sum_sq_err / self.count)

    def mae(self) -> float:
        if self.count == 0:
            return math.nan
        return self.sum_abs_err / self.count

    def pred_mean(self) -> float:
        if self.count == 0:
            return math.nan
        return self.pred_sum / self.count

    def truth_mean(self) -> float:
        if self.count == 0:
            return math.nan
        return self.truth_sum / self.count

    def pred_std(self) -> float:
        if self.count == 0:
            return math.nan
        mean = self.pred_mean()
        var = (self.pred_sum_sq / self.count) - (mean ** 2)
        return math.sqrt(max(0, var))

    def truth_std(self) -> float:
        if self.count == 0:
            return math.nan
        mean = self.truth_mean()
        var = (self.truth_sum_sq / self.count) - (mean ** 2)
        return math.sqrt(max(0, var))

    def pred_cv(self) -> float:
        """変動係数 = std / mean"""
        mean = self.pred_mean()
        if abs(mean) < 1e-10:
            return math.nan
        return self.pred_std() / mean

    def truth_cv(self) -> float:
        """変動係数 = std / mean"""
        mean = self.truth_mean()
        if abs(mean) < 1e-10:
            return math.nan
        return self.truth_std() / mean

    def cvrmse(self) -> float:
        """Coefficient of Variation of RMSE = RMSE / truth_mean"""
        rmse_val = self.rmse()
        truth_mean_val = self.truth_mean()
        if abs(truth_mean_val) < 1e-10:
            return math.nan
        return rmse_val / truth_mean_val


def load_config(config_path: Path) -> dict:
    """Load config.yaml"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_datetime(dt_str: str) -> datetime:
    """Parse datetime string (YYYY-MM-DDTHH) to datetime object"""
    return datetime.strptime(dt_str, "%Y-%m-%dT%H")


def list_forecast_files(
    forecasts_dir: Path,
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None,
) -> List[Path]:
    """
    Collect forecast files and return sorted by datetime and gnn step.
    Optionally filter by date range.
    """
    files = sorted(forecasts_dir.glob("source-era5_date-*_res-1.0_levels-13_steps-01_gnn*_predictions.nc"))
    
    if start_dt is not None or end_dt is not None:
        filtered = []
        for fpath in files:
            dt_str, _ = parse_dt_and_step(fpath)
            file_dt = parse_datetime(dt_str)
            if start_dt is not None and file_dt < start_dt:
                continue
            if end_dt is not None and file_dt > end_dt:
                continue
            filtered.append(fpath)
        return filtered
    
    return files


def parse_dt_and_step(path: Path) -> Tuple[str, int]:
    """
    Extract datetime string (YYYY-MM-DDTHH) and gnn step int from filename.
    Example: source-era5_date-2022-01-01T12_res-1.0_levels-13_steps-01_gnn16_predictions.nc
    """
    m = re.search(r"date-([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}).*gnn([0-9]{2})_", path.name)
    if not m:
        raise ValueError(f"Unexpected filename format: {path.name}")
    dt_str = m.group(1)
    step = int(m.group(2))
    return dt_str, step


def truth_path(dt_str: str, truth_dir: Path) -> Path:
    return truth_dir / f"source-era5_date-{dt_str}_res-1.0_levels-13_steps-01.nc"


def common_variables(ds_pred: xr.Dataset, ds_true: xr.Dataset) -> List[str]:
    return sorted(set(ds_pred.data_vars) & set(ds_true.data_vars))


def align_truth(pred_da: xr.DataArray, truth_da: xr.DataArray) -> xr.DataArray:
    """Align truth to prediction dims (time first), using time=-1 from truth."""
    if "time" in truth_da.dims:
        truth_da = truth_da.isel(time=-1)
    # add missing dims to truth to match prediction
    for dim in pred_da.dims:
        if dim not in truth_da.dims:
            truth_da = truth_da.expand_dims({dim: pred_da.coords[dim]})
    # reorder
    truth_da = truth_da.transpose(*pred_da.dims)
    return truth_da


def compute_stats_for_var(
    pred_da: xr.DataArray, truth_da: xr.DataArray, level_value=None
) -> Tuple[float, float, float, float]:
    """
    Returns: (mse_val, mae_val, pred_mean, truth_mean)
    lat/lonで平均し、残りの次元も平均してスカラー値に
    """
    if level_value is not None:
        pred_da = pred_da.sel(level=level_value)
        truth_da = truth_da.sel(level=level_value)
    
    # lat/lonで平均
    spatial_dims = [d for d in pred_da.dims if d in ("lat", "lon")]
    pred_mean = pred_da.mean(dim=spatial_dims)
    truth_mean = truth_da.mean(dim=spatial_dims)
    
    # 残りの次元（time, batch）も平均
    pred_mean_val = pred_mean.mean().item()
    truth_mean_val = truth_mean.mean().item()
    
    # 誤差計算
    diff = pred_da - truth_da
    mse = (diff ** 2).mean(dim=spatial_dims)
    mae = abs(diff).mean(dim=spatial_dims)
    mse_val = mse.mean().item()
    mae_val = mae.mean().item()
    
    return float(mse_val), float(mae_val), float(pred_mean_val), float(truth_mean_val)


def compute_stats_vectorized(
    pred_da: xr.DataArray, truth_da: xr.DataArray, level_filter: Optional[set] = None
) -> Dict[Optional[int], Tuple[float, float, float, float]]:
    """
    ベクトル化された統計計算。全レベルを一度に処理。
    Returns: Dict[level, (mse_val, mae_val, pred_mean, truth_mean)]
    levelがない変数の場合は level=None として扱う
    """
    results = {}
    spatial_dims = [d for d in pred_da.dims if d in ("lat", "lon")]
    
    if "level" in pred_da.dims:
        # levelがある変数: 全レベルを一度に処理
        levels = pred_da["level"].values
        
        # levelフィルタ適用
        if level_filter is not None:
            levels = [lvl for lvl in levels if int(lvl) in level_filter]
            if not levels:
                return results
            pred_da = pred_da.sel(level=levels)
            truth_da = truth_da.sel(level=levels)
        
        # lat/lonで平均（全レベルを一度に）
        pred_mean = pred_da.mean(dim=spatial_dims)
        truth_mean = truth_da.mean(dim=spatial_dims)
        
        # 残りの次元（time, batch）も平均
        pred_mean_vals = pred_mean.mean(dim=[d for d in pred_mean.dims if d != "level"]).values
        truth_mean_vals = truth_mean.mean(dim=[d for d in truth_mean.dims if d != "level"]).values
        
        # 誤差計算（全レベルを一度に）
        diff = pred_da - truth_da
        mse = (diff ** 2).mean(dim=spatial_dims)
        mae = abs(diff).mean(dim=spatial_dims)
        mse_vals = mse.mean(dim=[d for d in mse.dims if d != "level"]).values
        mae_vals = mae.mean(dim=[d for d in mae.dims if d != "level"]).values
        
        # 各レベルごとに結果を格納
        for i, lvl in enumerate(levels):
            lvl_int = int(lvl)
            results[lvl_int] = (
                float(mse_vals[i]),
                float(mae_vals[i]),
                float(pred_mean_vals[i]),
                float(truth_mean_vals[i]),
            )
    else:
        # levelがない変数
        pred_mean = pred_da.mean(dim=spatial_dims)
        truth_mean = truth_da.mean(dim=spatial_dims)
        
        pred_mean_val = pred_mean.mean().item()
        truth_mean_val = truth_mean.mean().item()
        
        diff = pred_da - truth_da
        mse = (diff ** 2).mean(dim=spatial_dims)
        mae = abs(diff).mean(dim=spatial_dims)
        mse_val = mse.mean().item()
        mae_val = mae.mean().item()
        
        results[None] = (
            float(mse_val),
            float(mae_val),
            float(pred_mean_val),
            float(truth_mean_val),
        )
    
    return results


def process_single_file(
    fpath: Path,
    truth_dir: Path,
    level_filter_set: Optional[set],
) -> Dict[Tuple[str, int], Tuple[float, float, float, float]]:
    """
    1つの予測ファイルを処理するワーカー関数（並列処理用）
    Returns: Dict[(variable, step), (mse_val, mae_val, pred_mean, truth_mean)]
    """
    results = {}
    
    dt_str, step = parse_dt_and_step(fpath)
    tpath = truth_path(dt_str, truth_dir)
    if not tpath.exists():
        return results
    
    try:
        with xr.open_dataset(tpath, decode_times=False) as ds_true, \
             xr.open_dataset(fpath, decode_times=False) as ds_pred:
            vars_common = common_variables(ds_pred, ds_true)
            for var in vars_common:
                pred_da = ds_pred[var]
                truth_da = ds_true[var]
                truth_aligned = align_truth(pred_da, truth_da)
                
                # ベクトル化された統計計算
                var_results = compute_stats_vectorized(
                    pred_da, truth_aligned, level_filter=level_filter_set
                )
                
                # 結果を格納
                for lvl, (mse_val, mae_val, pred_mean, truth_mean) in var_results.items():
                    if lvl is not None:
                        key = (f"{var}-level{lvl}", step)
                    else:
                        key = (var, step)
                    results[key] = (mse_val, mae_val, pred_mean, truth_mean)
    except Exception as e:
        print(f"[error] Failed to process {fpath}: {e}")
    
    return results


def accumulate_stats(
    forecast_files: Iterable[Path],
    truth_dir: Path,
    level_filter: Optional[List[int]] = None,
    num_workers: int = 25,
) -> Dict[Tuple[str, int], Stats]:
    """
    Accumulate statistics for all forecast files using parallel processing.
    
    Args:
        forecast_files: List of forecast file paths
        truth_dir: Directory containing truth files
        level_filter: Optional list of level values (hPa) to include. If None, all levels are used.
        num_workers: Number of parallel workers (default: 10)
    """
    stats: Dict[Tuple[str, int], Stats] = defaultdict(Stats)
    
    level_filter_set = set(level_filter) if level_filter is not None else None

    # Convert to list for progress bar
    forecast_files_list = list(forecast_files)
    
    # 並列処理でファイルを処理
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 全タスクをsubmit
        future_to_file = {
            executor.submit(process_single_file, fpath, truth_dir, level_filter_set): fpath
            for fpath in forecast_files_list
        }
        
        # 進捗バー付きで結果を取得
        for future in tqdm(as_completed(future_to_file), total=len(forecast_files_list), desc="Processing files", unit="file"):
            fpath = future_to_file[future]
            try:
                file_results = future.result()
                # 結果をstatsに追加
                for key, (mse_val, mae_val, pred_mean, truth_mean) in file_results.items():
                    stats[key].update(mse_val, mae_val, pred_mean, truth_mean)
            except Exception as e:
                print(f"[error] Failed to process {fpath}: {e}")
    
    return stats


def stats_to_dataframe(stats: Dict[Tuple[str, int], Stats]) -> pd.DataFrame:
    rows = []
    for (var, step), st in stats.items():
        rows.append({
            "variable": var,
            "step": step,
            "rmse": st.rmse(),
            "mae": st.mae(),
            "cvrmse": st.cvrmse(),
            "pred_mean": st.pred_mean(),
            "pred_std": st.pred_std(),
            "pred_cv": st.pred_cv(),
            "truth_mean": st.truth_mean(),
            "truth_std": st.truth_std(),
            "truth_cv": st.truth_cv(),
            "count": st.count,
        })
    df = pd.DataFrame(rows)
    return df.sort_values(["variable", "step"])


def plot_rmse(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # levelがある変数とない変数を分ける
    level_vars = {}  # {base_var: {level: group_df}}
    non_level_vars = {}  # {var: group_df}
    
    for var, g in df.groupby("variable"):
        if "-level" in var:
            # levelがある変数: temperature-level50 -> (temperature, 50)
            base_var, level_str = var.rsplit("-level", 1)
            level = int(level_str)
            if base_var not in level_vars:
                level_vars[base_var] = {}
            level_vars[base_var][level] = g
        else:
            # levelがない変数
            non_level_vars[var] = g
    
    # levelがある変数: 複数のサブプロットを1つのPNGに配置（最後にoverlay版を追加）
    for base_var, level_groups in level_vars.items():
        levels = sorted(level_groups.keys())
        n_levels = len(levels)
        
        # overlay版を含めた総サブプロット数
        total_subplots = n_levels + 1
        
        # グリッドサイズを計算（例: 14サブプロットなら4列×4行など）
        n_cols = min(4, total_subplots)  # 最大4列
        n_rows = (total_subplots + n_cols - 1) // n_cols  # 切り上げ
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        fig.suptitle(base_var, fontsize=16)
        
        # axesを1次元配列に変換（1行または1列の場合）
        if n_rows == 1:
            axes = axes if n_cols > 1 else [axes]
        elif n_cols == 1:
            axes = axes if n_rows > 1 else [axes]
        else:
            axes = axes.flatten()
        
        # 各levelの個別サブプロット
        for idx, level in enumerate(levels):
            ax = axes[idx]
            g = level_groups[level]
            ax.plot(g["step"], g["rmse"], marker="o")
            ax.set_title(f"level {level}")
            ax.set_xlabel("GNN message steps")
            ax.set_ylabel("RMSE")
            ax.grid(True, alpha=0.3)
            ax.set_xticks(sorted(g["step"].unique()))
        
        # 最後のサブプロット: 全てのlevelを重ねて表示
        ax_overlay = axes[n_levels]
        # 色は1つの色で、levelが小さいほど濃い色にする
        # colormapの値を変えて濃淡を調整（0.9=濃い、0.3=薄い）
        for idx, level in enumerate(levels):
            g = level_groups[level]
            # levelが小さいほど濃い色（colormapの値を調整）
            color_value = 0.9 - (idx / max(1, n_levels - 1)) * 0.6  # 0.9から0.3まで
            color = plt.cm.Blues(color_value)
            ax_overlay.plot(g["step"], g["rmse"], marker="o", label=f"level {level}", 
                          color=color)
        ax_overlay.set_title("All levels (overlay)")
        ax_overlay.set_xlabel("GNN message steps")
        ax_overlay.set_ylabel("RMSE")
        ax_overlay.legend(title="Level (hPa)", loc="best", ncol=2, fontsize=8)
        ax_overlay.grid(True, alpha=0.3)
        ax_overlay.set_xticks(sorted(df["step"].unique()))
        
        # 余ったサブプロットを非表示
        for idx in range(total_subplots, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(out_dir / f"{base_var}.png", dpi=200)
        plt.close()
    
    # levelがない変数: 個別のグラフ
    for var, g in non_level_vars.items():
        plt.figure(figsize=(6, 4))
        plt.plot(g["step"], g["rmse"], marker="o")
        plt.title(var)
        plt.xlabel("GNN message steps")
        plt.ylabel("RMSE")
        plt.grid(True, alpha=0.3)
        plt.xticks(sorted(g["step"].unique()))
        plt.tight_layout()
        plt.savefig(out_dir / f"{var}.png", dpi=200)
        plt.close()


def main() -> None:
    # Load config
    config = load_config(CONFIG_PATH)
    period = config.get("period", {})
    start_str = period.get("start")
    end_str = period.get("end")
    
    # Parse date range
    start_dt = None
    end_dt = None
    if start_str:
        start_dt = datetime.strptime(start_str, "%Y-%m-%dT%H:%M:%S")
    if end_str:
        end_dt = datetime.strptime(end_str, "%Y-%m-%dT%H:%M:%S")
    
    # Level filter: configには具体的なlevel値のリストがないので、全レベルを使用
    # 必要に応じて後で追加可能
    level_filter = None  # None = 全レベルを使用
    
    print(f"Config loaded:")
    if start_dt:
        print(f"  Start date: {start_dt}")
    if end_dt:
        print(f"  End date: {end_dt}")
    if level_filter:
        print(f"  Level filter: {level_filter}")
    else:
        print(f"  Level filter: All levels")
    
    forecast_files = list_forecast_files(FORECASTS_DIR, start_dt=start_dt, end_dt=end_dt)
    if not forecast_files:
        print(f"No forecast files found under {FORECASTS_DIR}")
        return

    print(f"\nFound {len(forecast_files)} forecast files (after filtering)")
    print(f"Using {25} parallel workers")
    stats = accumulate_stats(forecast_files, TRUTH_DIR, level_filter=level_filter, num_workers=25)
    df = stats_to_dataframe(stats)

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved statistics table to {CSV_PATH}")
    
    # countの統計情報を表示
    print(f"\nCount statistics:")
    print(f"  Min: {df['count'].min()}")
    print(f"  Max: {df['count'].max()}")
    print(f"  Mean: {df['count'].mean():.1f}")
    print(f"  Total variables×steps: {len(df)}")

    plot_rmse(df, FIG_DIR)
    print(f"\nSaved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()

