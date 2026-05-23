from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


WORKSPACE_ROOT = Path(__file__).parent.parent.parent
CSV_PATH = WORKSPACE_ROOT / "analyze/gnn_steps_rmse/data/gnn_steps_rmse_table.csv"
FIG_DIR = WORKSPACE_ROOT / "analyze/gnn_steps_rmse/stats_fig"
FIG_DIR_INDIVIDUAL = WORKSPACE_ROOT / "analyze/gnn_steps_rmse/stats_fig_individual"

# プロット対象の統計量
STAT_NAMES = [
    "rmse",
    "pred_mean",
    "pred_std",
]
CV_EXCLUDE_BASE_VARS = {
    "u_component_of_wind",
    "v_component_of_wind",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "vertical_velocity",
}
EXCLUDE_VARS = {
    "land_sea_mask",
    "geopotential_at_surface",
}


def load_csv(csv_path: Path) -> pd.DataFrame:
    """CSVファイルを読み込む"""
    return pd.read_csv(csv_path)


def parse_variable(var_name: str) -> Tuple[str, Optional[int]]:
    """
    変数名を解析してbase変数名とlevelを返す
    Returns: (base_var, level) or (var_name, None)
    """
    if "-level" in var_name:
        base_var, level_str = var_name.rsplit("-level", 1)
        level = int(level_str)
        return base_var, level
    return var_name, None


def separate_variables(df: pd.DataFrame) -> Tuple[Dict[str, Dict[int, pd.DataFrame]], Dict[str, pd.DataFrame]]:
    """
    変数をlevelありとlevelなしに分ける
    Returns: (level_vars, non_level_vars)
    level_vars: {base_var: {level: group_df}}
    non_level_vars: {var: group_df}
    """
    level_vars: Dict[str, Dict[int, pd.DataFrame]] = {}
    non_level_vars: Dict[str, pd.DataFrame] = {}
    
    for var, g in df.groupby("variable"):
        base_var, level = parse_variable(var)
        if level is not None:
            if base_var not in level_vars:
                level_vars[base_var] = {}
            level_vars[base_var][level] = g
        else:
            non_level_vars[var] = g
    
    return level_vars, non_level_vars


def format_step_label(step: int) -> str:
    if step < 0:
        return f"t{step}"
    return str(step)

def step_positions(steps: List[int]) -> Dict[int, int]:
    return {step: idx for idx, step in enumerate(steps)}


def plot_level_vars(
    df: pd.DataFrame,
    stat_name: str,
    base_var: str,
    level_groups: Dict[int, pd.DataFrame],
    ax1: plt.Axes,
    ax2: plt.Axes,
) -> None:
    """
    levelあり変数のプロット（2つのサブプロット）
    - サブプロット1: 統計量を横軸、levelを縦軸、GNN stepごとに系列（step16が最も濃い）
    - サブプロット2: GNN stepを横軸、統計量を縦軸、levelごとに系列（levelが低いほど濃い）
    """
    levels = sorted(level_groups.keys())
    steps = sorted(df["step"].unique())
    step_pos = step_positions(steps)
    
    ax1.set_title(f"{base_var}", fontsize=10)
    
    n_steps = len(steps)
    n_levels = len(levels)
    
    # サブプロット1: 統計量を横軸、levelを縦軸、GNN stepごとに系列（step16が最も濃い）
    for step_idx, step in enumerate(steps):
        # このstepのデータを抽出
        step_data = []
        step_levels = []
        for level in levels:
            g = level_groups[level]
            step_df = g[g["step"] == step]
            if not step_df.empty:
                stat_value = step_df[stat_name].iloc[0]
                step_data.append(stat_value)
                step_levels.append(level)
        
        if step_data:
            # stepが大きいほど濃い色（step16が最も濃い）
            color_value = 0.3 + (step_idx / max(1, n_steps - 1)) * 0.6  # 0.3から0.9まで
            color = plt.cm.Blues(color_value)
            ax1.plot(step_data, step_levels, marker="o", label=format_step_label(step), color=color, linewidth=1.5)
    
    ax1.set_xlabel(stat_name)
    ax1.set_ylabel("Level (hPa)")
    ax1.legend(title="steps", loc="best", ncol=3, fontsize=7)

    ax1.grid(True, alpha=0.3)
    ax1.set_yticks(levels)
    ax1.invert_yaxis()  # y軸を反転（下が1000、上が50）
    
    # サブプロット2: GNN stepを横軸、levelごとに系列（levelが高いほど濃い）
    ax2.set_title(f"{base_var}", fontsize=10)
    for level_idx, level in enumerate(levels):
        g = level_groups[level]
        level_data = []
        level_steps = []
        for step in steps:
            step_df = g[g["step"] == step]
            if not step_df.empty:
                stat_value = step_df[stat_name].iloc[0]
                level_data.append(stat_value)
                level_steps.append(step_pos[step])
        
        if level_data:
            # levelが高いほど濃い色（level 1000が最も濃い）
            color_value = 0.3 + (level_idx / max(1, n_levels - 1)) * 0.6  # 0.3から0.9まで
            color = plt.cm.Blues(color_value)
            ax2.plot(level_steps, level_data, marker="o", label=f"level {level}", color=color, linewidth=1.5)
    
    ax2.set_xlabel("steps")
    ax2.set_ylabel(stat_name)
    ax2.legend(title="Level (hPa)", loc="best", ncol=3, fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(list(step_pos.values()))
    ax2.set_xticklabels([format_step_label(step) for step in steps])


def plot_non_level_vars(
    df: pd.DataFrame,
    stat_name: str,
    var: str,
    var_df: pd.DataFrame,
    ax: plt.Axes,
) -> None:
    """
    levelなし変数のプロット（GNN stepを横軸、統計量を縦軸）
    """
    steps = sorted(var_df["step"].unique())
    step_pos = step_positions(steps)
    x = [step_pos[step] for step in var_df["step"]]
    ax.plot(x, var_df[stat_name], marker="o", linewidth=1.5)
    ax.set_title(f"{var}", fontsize=10)
    ax.set_xlabel("steps")
    ax.set_ylabel(stat_name)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(list(step_pos.values()))
    ax.set_xticklabels([format_step_label(step) for step in steps])


def plot_stat_by_level(df: pd.DataFrame, stat_name: str, out_dir: Path) -> None:
    """
    指定された統計量について、全ての変数を1つのPNGファイルにプロット
    """
    df = df[~df["variable"].isin(EXCLUDE_VARS)]
    if stat_name in {"pred_cv", "cvrmse"}:
        df = df.copy()
        df["base_var"] = df["variable"].apply(lambda v: parse_variable(v)[0])
        df = df[~df["base_var"].isin(CV_EXCLUDE_BASE_VARS)]
        df = df.drop(columns=["base_var"])

    level_vars, non_level_vars = separate_variables(df)
    
    # 必要なサブプロット数を計算
    # levelあり変数: 各2つのサブプロット
    # levelなし変数: 各1つのサブプロット
    n_level_vars = len(level_vars)
    n_non_level_vars = len(non_level_vars)
    n_subplots = n_level_vars * 2 + n_non_level_vars
    
    if n_subplots == 0:
        print(f"  No variables found for {stat_name}")
        return
    
    # レイアウトを決定（列数を4列に固定）
    n_cols = 4
    n_rows = (n_subplots + n_cols - 1) // n_cols
    
    # figureを作成
    fig = plt.figure(figsize=(22, 5 * n_rows))
    fig.suptitle(f"{stat_name.upper()} - All Variables", fontsize=16, y=0.995)
    
    subplot_idx = 0
    
    # levelあり変数のプロット
    for base_var, level_groups in sorted(level_vars.items()):
        ax1 = plt.subplot(n_rows, n_cols, subplot_idx + 1)
        ax2 = plt.subplot(n_rows, n_cols, subplot_idx + 2)
        plot_level_vars(df, stat_name, base_var, level_groups, ax1, ax2)
        subplot_idx += 2
    
    # levelなし変数のプロット
    for var, var_df in sorted(non_level_vars.items()):
        ax = plt.subplot(n_rows, n_cols, subplot_idx + 1)
        plot_non_level_vars(df, stat_name, var, var_df, ax)
        subplot_idx += 1
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # 1つのPNGファイルとして保存
    out_file = out_dir / f"{stat_name}_all_variables.png"
    try:
        plt.savefig(out_file, dpi=200, bbox_inches='tight')
        print(f"  Saved: {out_file}")
    except PermissionError:
        print(f"  [WARNING] Permission denied: {out_file}")
        print(f"  Please run with sudo or change directory permissions")
        raise
    finally:
        plt.close()


def plot_stat_by_level_individual(df: pd.DataFrame, stat_name: str, out_dir: Path) -> None:
    """
    指定された統計量について、変数ごとにPNGファイルを出力
    """
    df = df[~df["variable"].isin(EXCLUDE_VARS)]
    if stat_name in {"pred_cv", "cvrmse"}:
        df = df.copy()
        df["base_var"] = df["variable"].apply(lambda v: parse_variable(v)[0])
        df = df[~df["base_var"].isin(CV_EXCLUDE_BASE_VARS)]
        df = df.drop(columns=["base_var"])

    level_vars, non_level_vars = separate_variables(df)

    for base_var, level_groups in sorted(level_vars.items()):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        plot_level_vars(df, stat_name, base_var, level_groups, ax1, ax2)
        plt.tight_layout()
        out_file = out_dir / f"{stat_name}_{base_var}.png"
        try:
            plt.savefig(out_file, dpi=200, bbox_inches="tight")
            print(f"  Saved: {out_file}")
        except PermissionError:
            print(f"  [WARNING] Permission denied: {out_file}")
            print("  Please run with sudo or change directory permissions")
            raise
        finally:
            plt.close()

    for var, var_df in sorted(non_level_vars.items()):
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        plot_non_level_vars(df, stat_name, var, var_df, ax)
        plt.tight_layout()
        out_file = out_dir / f"{stat_name}_{var}.png"
        try:
            plt.savefig(out_file, dpi=200, bbox_inches="tight")
            print(f"  Saved: {out_file}")
        except PermissionError:
            print(f"  [WARNING] Permission denied: {out_file}")
            print("  Please run with sudo or change directory permissions")
            raise
        finally:
            plt.close()


def main() -> None:
    """メイン関数"""
    print(f"Loading CSV from {CSV_PATH}")
    df = load_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows")
    
    # 出力ディレクトリの確認
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {FIG_DIR}")
    FIG_DIR_INDIVIDUAL.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {FIG_DIR_INDIVIDUAL}")
    
    # 各統計量についてプロット
    for stat_name in STAT_NAMES:
        print(f"\nPlotting {stat_name}...")
        plot_stat_by_level(df, stat_name, FIG_DIR)
        plot_stat_by_level_individual(df, stat_name, FIG_DIR_INDIVIDUAL)
    
    print(f"\nAll plots saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
