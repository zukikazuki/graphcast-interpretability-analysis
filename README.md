# GraphCast Internal Representation Analysis

This repository is a cleaned public research artifact on internal representation analysis in GraphCast-small.

I used layerwise decoding and output-side readouts to examine how decoded forecast fields, RMSE, and distributional statistics change across GraphCast's encoder and processor layers. I also compared these layerwise readouts with VJP/Hutchinson-based decoder sensitivity estimates. The goal was to study how intermediate mesh representations change across message-passing depth, and whether representation-level changes align with decoder sensitivity.

This repository is intended as a research artifact, not a fully self-contained reproduction package. It includes selected analysis code, figures, notes, and the GraphCast source code used for the experiments. It does not include pretrained model weights, ERA5 data, local cache files, or the full original experimental environment.

For a short summary of the main observations and interpretation, see [`RESULTS.md`](RESULTS.md).

## Research Focus

GraphCast maps weather fields through a grid-to-mesh encoder, repeated mesh message passing, and a mesh-to-grid decoder. This project probes that internal computation by:

- decoding intermediate mesh representations across message-passing steps
- measuring RMSE, MAE, prediction mean, prediction standard deviation, and related readout statistics by variable and pressure level
- aggregating layerwise relative scores against the final GraphCast-small processor step
- estimating decoder sensitivity with VJP/Hutchinson scores for processor layers

Instead of treating GraphCast only as an input-output forecaster, this project examines how its intermediate representations evolve across processor depth.

## Repository Layout

- `analyze/gnn_steps_rmse/`: layerwise forecast/readout statistics and plots.
- `analyze/gnn_steps_rmse_aggregate/`: aggregate relative RMSE, mean, and standard-deviation scores.
- `analyze/vjp/`: notebook for VJP/Hutchinson sensitivity analysis.
- `graphcast_pipeline/`: ERA5 download, preprocessing, and GraphCast-small inference pipeline used in the experiments.
- `docs/research_summary.md`: Japanese research summary and framing notes.
- `docs/decoder_compatibility_hutchinson.md`: VJP/Hutchinson decoder-compatibility design note.
- `docs/variable.md`: GraphCast input, target, and forcing variable notes.
- `graphcast/`: GraphCast/GenCast model source code derived from Google DeepMind's public implementation.

## Example Outputs

Representative aggregate figures are included in:

- `analyze/gnn_steps_rmse_aggregate/figs_selected_stacked/relative_rmse_score_selected_stacked.png`
- `analyze/gnn_steps_rmse_aggregate/figs_selected_stacked/relative_mean_score_selected_stacked.png`
- `analyze/gnn_steps_rmse_aggregate/figs_selected_stacked/relative_std_score_selected_stacked.png`

The corresponding aggregate table is:

- `analyze/gnn_steps_rmse_aggregate/gnn_steps_rmse_relative.csv`

Detailed per-variable plots are under `analyze/gnn_steps_rmse/figs/` and `analyze/gnn_steps_rmse/stats_fig_individual/`.

## Running Notes

The code reflects the original local experiment environment. To run the pipeline, update paths in `graphcast_pipeline/configs/config.yaml` for your machine and provide the required external artifacts:

- GraphCast-small pretrained checkpoint
- normalization statistics
- ERA5-derived input datasets
- CDS API credentials if downloading ERA5 data

See `docs/usage.md` for the current pipeline entry points.

## Attribution

The original GraphCast and GenCast implementation is by Google DeepMind and is licensed under Apache 2.0. The files under `graphcast/` and the original demo structure come from that implementation.

My contributions in this repository are the public cleanup, GraphCast-small analysis pipeline, layerwise decoding/RMSE experiments, VJP/Hutchinson sensitivity analysis, generated figures/tables, and research notes.

## Not Included

This repository intentionally excludes:

- pretrained model weights
- ERA5 data
- generated local cache files
- the original private git history
- unrelated application, thesis build, and administrative materials
