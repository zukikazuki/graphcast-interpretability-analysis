# Results Summary

This project studies how intermediate mesh representations in GraphCast-small change across processor depth, and when those representations become useful to the decoder.

The core experiment is a layerwise decoding analysis. For each processor layer, I connected the intermediate mesh latent to the same mesh-to-grid decoder and decoded it back into physical forecast fields. I then compared the decoded fields across depth using prediction mean, prediction standard deviation, RMSE, and decoder sensitivity estimates.

## Experimental Setup

- Model: GraphCast-small
- Data: ERA5, calendar year 2022
- Evaluation times: 6-hour intervals, 1460 timestamps
- Layers: encoder output and 16 processor steps, indexed as layers 0 through 16
- Variables: 5 single-level variables and 6 pressure-level variables across 13 pressure levels
- Readouts: prediction mean, prediction standard deviation, RMSE, and related aggregate scores
- Sensitivity check: VJP/Hutchinson estimates of decoder sensitivity using Rademacher vectors

## Main Observations

### 1. Mean fields are mostly stable across layers

Decoded prediction means did not change dramatically across processor depth for many variables. This suggests that the model does not strongly disrupt the large-scale mean structure of the forecast field while updating its internal representation.

### 2. Prediction standard deviation drops after the encoder and recovers later

A clearer pattern appeared in prediction standard deviation. At layer 0, immediately after the grid-to-mesh encoder, decoded fields often showed a large drop in standard deviation. Across later processor layers, this standard deviation tended to recover.

This suggests that the encoder does not simply preserve the input field as-is. Instead, it may map the input into a compressed or smoothed latent representation, while later processor steps reintroduce or reallocate variation that is useful for prediction.

### 3. RMSE improves differently across variables

RMSE did not follow a single universal pattern. Some variables already improved substantially at the encoder output, while others improved more gradually through processor depth.

This suggests that message-passing depth does not play the same role for every forecast quantity. Some variables may be largely shaped by the initial encoding, while others depend more on the repeated processor updates.

### 4. Decoder sensitivity changes little across layers

To check whether layerwise decoding is a reasonable comparison, I estimated decoder sensitivity for each layer using VJP/Hutchinson scores. These scores did not show large changes across processor depth.

This supports the interpretation that processor representations remain in a roughly decoder-compatible coordinate system. In other words, decoding intermediate layers with the same decoder does not appear to be an obviously broken operation, at least at the scale measured by this sensitivity estimate.

## Interpretation

The results support a simple conceptual picture of GraphCast-small:

1. The encoder maps the input weather fields into a mesh latent space.
2. The processor updates that latent state through repeated message passing.
3. The decoder maps the updated latent state back to physical forecast variables.

Under this view, the processor is not just gradually making the decoder "work better" at each layer. Instead, it appears to update a latent state that already remains readable by the decoder, while changing the forecast-relevant information represented in that state.

The drop and recovery of standard deviation suggests that the learned latent dynamics may not preserve all physical detail directly. The model may first suppress or compress local variation, then reconstruct or reallocate the variation that is useful for the next-step forecast. This is consistent with the idea that the learned dynamics are optimized for forecast loss, not for perfectly preserving the full physical state.

## Limitations

This repository is a cleaned public research artifact, not a fully self-contained reproduction package. It does not include pretrained weights, ERA5 data, local caches, or the full original experimental environment.

The analysis also leaves several research questions open:

- how much of the RMSE improvement comes from distributional changes versus spatially meaningful forecast improvements
- whether the standard-deviation pattern holds uniformly across all variables and pressure levels
- how the model behaves on extremes, sharp gradients, and spatial spectra
- whether VJP sensitivity can be localized spatially to identify which regions or variables dominate decoder sensitivity

These are natural directions for extending the analysis.
