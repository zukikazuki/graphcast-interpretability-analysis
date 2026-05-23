from __future__ import annotations

from pathlib import Path

import pandas as pd

WORKSPACE_ROOT = Path(__file__).parent.parent.parent
INPUT_CSV = WORKSPACE_ROOT / "analyze/gnn_steps_rmse/data/gnn_steps_rmse_table.csv"
OUTPUT_CSV = WORKSPACE_ROOT / "analyze/gnn_steps_rmse_aggregate/gnn_steps_rmse_relative.csv"

METRICS = {
    "rmse": "rmse_score",
    "pred_mean": "mean_score",
    "pred_std": "std_score",
}


def load_scores(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["step"] = pd.to_numeric(df["step"], errors="coerce").astype("Int64")
    for metric in METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=["step", "variable"]).copy()
    df["step"] = df["step"].astype(int)
    return df


def add_relative_scores(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df["step"] == 16][["variable", *METRICS.keys()]]
    base = base.rename(columns={m: f"{m}_base" for m in METRICS})
    merged = df.merge(base, on="variable", how="left")
    for metric, score_name in METRICS.items():
        merged[score_name] = merged[metric] / merged[f"{metric}_base"]
    return merged


def main() -> None:
    df = load_scores(INPUT_CSV)
    df = add_relative_scores(df)
    out_cols = ["variable", "step", "rmse_score", "mean_score", "std_score"]
    df[out_cols].to_csv(OUTPUT_CSV, index=False)


if __name__ == "__main__":
    main()
