from __future__ import annotations

import csv
from collections import defaultdict
from math import ceil, nan
from pathlib import Path

import matplotlib.pyplot as plt

WORKSPACE_ROOT = Path(__file__).parent.parent.parent
INPUT_CSV = WORKSPACE_ROOT / "analyze/gnn_steps_rmse_aggregate/gnn_steps_rmse_relative.csv"
OUTPUT_DIR = WORKSPACE_ROOT / "analyze/gnn_steps_rmse_aggregate/figs"
OUTPUT_DIR_INDIVIDUAL = WORKSPACE_ROOT / "analyze/gnn_steps_rmse_aggregate/figs_individual"
OUTPUT_DIR_COMBINED = WORKSPACE_ROOT / "analyze/gnn_steps_rmse_aggregate/figs_combined"
OUTPUT_DIR_SELECTED_STACKED = (
    WORKSPACE_ROOT / "analyze/gnn_steps_rmse_aggregate/figs_selected_stacked"
)

METRICS = [
    "rmse_score",
    "mean_score",
    "std_score",
]
EXCLUDE_VARS = {
    "geopotential_at_surface",
    "land_sea_mask",
}
EXCLUDE_MEAN_BASE_VARS = {
    # "10m_u_component_of_wind",
    # "10m_v_component_of_wind",
    # "u_component_of_wind",
    # "v_component_of_wind",
    # "vertical_velocity",
}
SELECTED_VAR_ALIASES = {
    "10m_u": "10m_u_component_of_wind",
    "10m_v": "10m_v_component_of_wind",
    "wmtemp": "2m_temperature",
    "geopo": "geopotential",
    "pressure": "mean_sea_level_pressure",
    "humid": "specific_humidity",
}


def _to_float(value: str) -> float:
    if value is None:
        return nan
    value = value.strip()
    if value == "":
        return nan
    return float(value)


def _base_var_name(var_name: str) -> str:
    if "-level" in var_name:
        return var_name.split("-level", 1)[0]
    return var_name


def _safe_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def load_data(path: Path) -> tuple[list[str], list[int], dict[str, dict[int, dict[str, float]]]]:
    variables = set()
    steps = set()
    sums: dict[str, dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    counts: dict[str, dict[int, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            var = _base_var_name(row["variable"])
            step = int(float(row["step"]))
            if var in EXCLUDE_VARS:
                continue
            variables.add(var)
            steps.add(step)
            for metric in METRICS:
                value = _to_float(row[metric])
                if value != value:
                    continue
                prev = sums[var][step].get(metric, 0.0)
                sums[var][step][metric] = prev + value
                counts[var][step][metric] = counts[var][step].get(metric, 0) + 1

    data: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    for var in variables:
        for step in steps:
            if step not in sums[var]:
                continue
            data[var][step] = {}
            for metric, total in sums[var][step].items():
                count = counts[var][step][metric]
                data[var][step][metric] = total / count if count > 0 else nan

    return sorted(variables), sorted(steps), data


def plot_metric(
    variables: list[str],
    steps: list[int],
    data: dict[str, dict[int, dict[str, float]]],
    metric: str,
    out_path: Path,
) -> None:
    plot_vars = variables
    if metric == "mean_score":
        plot_vars = [v for v in variables if v not in EXCLUDE_MEAN_BASE_VARS]
    n_vars = len(plot_vars)
    ncols = 3
    nrows = ceil(n_vars / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.0, nrows * 3.2), sharex=True)
    axes_list = axes.flatten() if n_vars > 1 else [axes]

    for ax in axes_list[n_vars:]:
        ax.axis("off")

    for idx, var in enumerate(plot_vars):
        ax = axes_list[idx]
        y = [data[var].get(step, {}).get(metric, nan) for step in steps]
        x = list(range(len(steps)))
        if len(x) >= 3:
            line, = ax.plot(x[2:], y[2:], linewidth=1.0)
            color = line.get_color()
        else:
            line, = ax.plot(x, y, linewidth=1.0)
            color = line.get_color()
        # Make the first two segments dotted (-12 and -6)
        if len(x) >= 2:
            ax.plot(x[:2], y[:2], linestyle=(0, (1, 1)), linewidth=1.0, color=color)
        if len(x) >= 3:
            ax.plot(x[1:3], y[1:3], linestyle=(0, (1, 1)), linewidth=1.0, color=color)
        for xi, step, yi in zip(x, steps, y):
            if yi != yi:
                continue
            alpha = 0.15 if step in (-12, -6) else 1.0
            ax.scatter([xi], [yi], s=18, alpha=alpha, color=color)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.set_title(var, fontsize=9)
        ax.grid(True, alpha=0.2)
        valid = [v for v in y if v == v]
        if valid:
            y_max = max(valid)
            pad = max(0.0, y_max * 0.05)
            ax.set_ylim(0.0, y_max + pad)

        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in steps], fontsize=8)
        ax.tick_params(axis="x", labelbottom=True)

    fig.suptitle(metric, fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def plot_metric_individual(
    variables: list[str],
    steps: list[int],
    data: dict[str, dict[int, dict[str, float]]],
    metric: str,
    out_dir: Path,
) -> None:
    plot_vars = variables
    if metric == "mean_score":
        plot_vars = [v for v in variables if v not in EXCLUDE_MEAN_BASE_VARS]

    out_dir.mkdir(parents=True, exist_ok=True)
    for var in plot_vars:
        y = [data[var].get(step, {}).get(metric, nan) for step in steps]
        x = list(range(len(steps)))

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        if len(x) >= 3:
            line, = ax.plot(x[2:], y[2:], linewidth=1.2)
            color = line.get_color()
        else:
            line, = ax.plot(x, y, linewidth=1.2)
            color = line.get_color()
        if len(x) >= 2:
            ax.plot(x[:2], y[:2], linestyle=(0, (1, 1)), linewidth=1.2, color=color)
        if len(x) >= 3:
            ax.plot(x[1:3], y[1:3], linestyle=(0, (1, 1)), linewidth=1.2, color=color)

        for xi, step, yi in zip(x, steps, y):
            if yi != yi:
                continue
            alpha = 0.15 if step in (-12, -6) else 1.0
            ax.scatter([xi], [yi], s=22, alpha=alpha, color=color)

        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(True, alpha=0.2)
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in steps], fontsize=9)
        ax.tick_params(axis="x", labelbottom=True)

        valid = [v for v in y if v == v]
        if valid:
            y_max = max(valid)
            pad = max(0.0, y_max * 0.05)
            ax.set_ylim(0.0, y_max + pad)

        fig.tight_layout()
        out_path = out_dir / f"{_safe_name(var)}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def plot_metric_combined(
    variables: list[str],
    steps: list[int],
    data: dict[str, dict[int, dict[str, float]]],
    metric: str,
    out_path: Path,
) -> None:
    plot_vars = variables
    if metric == "mean_score":
        plot_vars = [v for v in variables if v not in EXCLUDE_MEAN_BASE_VARS]

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    x = list(range(len(steps)))

    for var in plot_vars:
        y = [data[var].get(step, {}).get(metric, nan) for step in steps]
        ax.plot(x, y, marker="o", linewidth=1.0, label=var)

    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.set_title(metric, fontsize=14)
    ax.grid(True, alpha=0.2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in steps], fontsize=9)
    ax.tick_params(axis="x", labelbottom=True)

    # Legend outside to avoid covering lines
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _resolve_selected_vars(variables: list[str]) -> list[str]:
    resolved: list[str] = []
    for alias, var in SELECTED_VAR_ALIASES.items():
        if var in variables:
            resolved.append(var)
            continue
        if alias == "wmtemp" and "temperature" in variables:
            resolved.append("temperature")
            continue
        raise ValueError(f"Selected variable not found: {alias} -> {var}")
    return resolved


def plot_metric_selected_stacked(
    variables: list[str],
    steps: list[int],
    data: dict[str, dict[int, dict[str, float]]],
    metric: str,
    out_path: Path,
) -> None:
    selected_vars = _resolve_selected_vars(variables)
    nrows = len(selected_vars)
    fig, axes = plt.subplots(
        nrows, 1, figsize=(6.0, max(2.2 * nrows, 8.5)), sharex=True
    )
    axes_list = axes.flatten().tolist() if hasattr(axes, "flatten") else [axes]

    for idx, var in enumerate(selected_vars):
        ax = axes_list[idx]
        y = [data[var].get(step, {}).get(metric, nan) for step in steps]
        x = list(range(len(steps)))
        if len(x) >= 3:
            line, = ax.plot(x[2:], y[2:], linewidth=1.2)
            color = line.get_color()
        else:
            line, = ax.plot(x, y, linewidth=1.2)
            color = line.get_color()
        if len(x) >= 2:
            ax.plot(x[:2], y[:2], linestyle=(0, (1, 1)), linewidth=1.2, color=color)
        if len(x) >= 3:
            ax.plot(x[1:3], y[1:3], linestyle=(0, (1, 1)), linewidth=1.2, color=color)

        for xi, step, yi in zip(x, steps, y):
            if yi != yi:
                continue
            alpha = 0.15 if step in (-12, -6) else 1.0
            ax.scatter([xi], [yi], s=22, alpha=alpha, color=color)

        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(True, alpha=0.2)
        ax.tick_params(axis="x", labelbottom=True)

        valid = [v for v in y if v == v]
        if valid:
            y_max = max(valid)
            pad = max(0.0, y_max * 0.05)
            ax.set_ylim(0.0, y_max + pad)

    axes_list[-1].set_xticks(list(range(len(steps))))
    axes_list[-1].set_xticklabels([str(s) for s in steps], fontsize=9)

    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR_INDIVIDUAL.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR_COMBINED.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR_SELECTED_STACKED.mkdir(parents=True, exist_ok=True)
    variables, steps, data = load_data(INPUT_CSV)

    for metric in METRICS:
        out_path = OUTPUT_DIR / f"relative_{metric}.png"
        plot_metric(variables, steps, data, metric, out_path)
        plot_metric_individual(variables, steps, data, metric, OUTPUT_DIR_INDIVIDUAL / metric)
        plot_metric_combined(
            variables,
            steps,
            data,
            metric,
            OUTPUT_DIR_COMBINED / f"relative_{metric}_combined.png",
        )
        plot_metric_selected_stacked(
            variables,
            steps,
            data,
            metric,
            OUTPUT_DIR_SELECTED_STACKED / f"relative_{metric}_selected_stacked.png",
        )


if __name__ == "__main__":
    main()
