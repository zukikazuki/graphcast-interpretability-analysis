# Usage Notes

This repository preserves the experiment code used for the original local GraphCast-small analysis. It is not a turnkey reproduction package: model weights, normalization statistics, ERA5 data, and local cache files are not included.

Before running, edit `graphcast_pipeline/configs/config.yaml` so that `root`, `storage`, `model.checkpoint_path`, and `model.stats_dir` point to valid local paths.

## Data Preparation

Download ERA5 data using the configured period and variable list:

```bash
python -m graphcast_pipeline.scripts.download_era5 --config graphcast_pipeline/configs/config.yaml
```

Convert the downloaded ERA5 files into GraphCast input datasets:

```bash
python graphcast_pipeline/scripts/shape_era5_to_graphcast.py \
  --config graphcast_pipeline/configs/config.yaml \
  --force
```

## Inference

Run GraphCast-small inference for the configured period:

```bash
python -m graphcast_pipeline.run.run
```

The same entry point is used for the standard forecast mode and VJP/Hutchinson mode. Select the mode in `graphcast_pipeline/configs/config.yaml`:

```yaml
model:
  mode: standard
```

or:

```yaml
model:
  mode: vjp
```

## Analysis

Layerwise RMSE/readout aggregation:

```bash
python -m analyze.gnn_steps_rmse
python analyze/gnn_steps_rmse_aggregate/aggregate_scores.py
python analyze/gnn_steps_rmse_aggregate/plot_relative_by_variable.py
```

Some analysis scripts still contain local absolute paths from the original experiment machine. Treat those as configuration points rather than portable defaults.

