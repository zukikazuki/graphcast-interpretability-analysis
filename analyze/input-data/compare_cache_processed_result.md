# Cache vs Processed Dataset Comparison

**Generated:** 2025-11-28 14:30:37

## Files Compared

- **Cache:** `/home/dl-box/Fujikawa/research/machine_learning/graphcast/graphcast_cache/source-era5_date-2022-01-01_res-1.0_levels-13_steps-01.nc`
- **Processed:** `/media/dl-box/SSD-SCTU3A/graphcast_data/data/processed/graphcast_inputs/source-era5_date-2022-01-01T12_res-1.0_levels-13_steps-01.nc`

## Summary

- **Structure:** ❌ Mismatch
- **Values:** ❌ Mismatch
- **Overall:** ❌ **FAIL**

## Errors

- ❌ Only in cache: {'toa_incident_solar_radiation'}

## Variable Comparisons

| Variable | Dims Match | Shape Match | Dtype Match | Values Match | Max Diff | Mean Diff | Rel Mean Diff (%) | Cache Mean | Processed Mean | NaN Info | Cache Coords | Processed Coords |
|----------|------------|-------------|-------------|--------------|----------|-----------|-------------------|------------|----------------|----------|--------------|-----------------|
| `10m_u_component_of_wind` | ✅ | ✅ | ✅ | ❌ | 0.000453948975 | 0.000226734890 | 0.084263820775 | -0.269077390432 | -0.269075274467 | - | lat, lon, time | lat, lon, time |
| `10m_v_component_of_wind` | ✅ | ✅ | ✅ | ❌ | 0.000382423401 | 0.000191880288 | 0.060833857873 | -0.315416932106 | -0.315416604280 | - | lat, lon, time | lat, lon, time |
| `2m_temperature` | ✅ | ✅ | ✅ | ✅ | - | - | - | 276.924011230469 | 276.923980712891 | - | lat, lon, time | lat, lon, time |
| `datetime` | ✅ | ✅ | ✅ | ✅ | - | - | - | 6.000000000000 | 6.000000000000 | - | time | time |
| `geopotential` | ✅ | ✅ | ✅ | ❌ | 0.140625000000 | 0.047366973013 | 0.000061028670 | 77614.296875000000 | 77614.296875000000 | - | lat, level, lon, time | lat, level, lon, time |
| `geopotential_at_surface` | ✅ | ✅ | ✅ | ❌ | 0.449707031250 | 0.225935399532 | 0.006002051398 | 3764.302978515625 | 3764.306396484375 | - | lat, lon | lat, lon |
| `land_sea_mask` | ✅ | ✅ | ✅ | ❌ | 0.000007629395 | 0.000000815648 | 0.000242277686 | 0.336658269167 | 0.336658269167 | - | lat, lon | lat, lon |
| `mean_sea_level_pressure` | ✅ | ✅ | ✅ | ✅ | - | - | - | 101078.750000000000 | 101078.765625000000 | - | lat, lon, time | lat, lon, time |
| `specific_humidity` | ✅ | ✅ | ✅ | ❌ | 0.000000182539 | 0.000000031370 | 0.001897957405 | 0.001652831561 | 0.001652831328 | - | lat, level, lon, time | lat, level, lon, time |
| `temperature` | ✅ | ✅ | ✅ | ✅ | - | - | - | 242.506622314453 | 242.506622314453 | - | lat, level, lon, time | lat, level, lon, time |
| `total_precipitation_6hr` | ✅ | ✅ | ✅ | ❌ | 0.000001486391 | 0.000000200987 | 0.035588654716 | 0.000564750691 | 0.000564670830 | - | lat, lon, time | lat, lon, time |
| `u_component_of_wind` | ✅ | ✅ | ✅ | ❌ | 0.001075744629 | 0.000394434814 | 0.005188132066 | 7.602636337280 | 7.602635860443 | - | lat, level, lon, time | lat, level, lon, time |
| `v_component_of_wind` | ✅ | ✅ | ✅ | ❌ | 0.001213073730 | 0.000407329033 | 1.241607483963 | 0.032806586474 | 0.032807845622 | - | lat, level, lon, time | lat, level, lon, time |
| `vertical_velocity` | ✅ | ✅ | ✅ | ❌ | 0.000168561935 | 0.000050533417 | 1.183949640974 | 0.004268206656 | 0.004268160556 | - | lat, level, lon, time | lat, level, lon, time |

## Coordinate Comparisons

| Coordinate | In Cache | In Processed | Dims Match | Shape Match | Dtype Match | Values Match | Max Diff | Mean Diff | Cache Dims | Processed Dims | Cache Shape | Processed Shape | Cache Dtype | Processed Dtype | Cache Size | Processed Size |
|------------|----------|---------------|------------|-------------|-------------|--------------|----------|-----------|------------|----------------|-------------|-----------------|------------|-----------------|------------|----------------|
| `lat` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | - | - | ('lat',) | ('lat',) | (181,) | (181,) | float32 | float32 | 181 | 181 |

### `lat` Coordinate Details

- **Cache Values:** `[-90.0, ..., 90.0] (length=181)`
- **Processed Values:** `[-90.0, ..., 90.0] (length=181)`

| `level` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | - | - | ('level',) | ('level',) | (13,) | (13,) | int32 | int32 | 13 | 13 |

### `level` Coordinate Details

- **Cache Values:** `[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]`
- **Processed Values:** `[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]`

| `lon` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | - | - | ('lon',) | ('lon',) | (360,) | (360,) | float32 | float32 | 360 | 360 |

### `lon` Coordinate Details

- **Cache Values:** `[0.0, ..., 359.0] (length=360)`
- **Processed Values:** `[0.0, ..., 359.0] (length=360)`

| `time` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | - | - | ('time',) | ('time',) | (3,) | (3,) | int32 | int32 | 3 | 3 |

### `time` Coordinate Details

- **Cache Values:** `[0, 6, 12]`
- **Processed Values:** `[0, 6, 12]`

grid2mesh_gnn/~_networks_builder/encoder_nodes_grid_nodes_mlp/act_layer_0
grid2mesh_gnn/~_networks_builder/encoder_nodes_grid_nodes_mlp/act_layer_1
grid2mesh_gnn/~_networks_builder/encoder_nodes_mesh_nodes_mlp/act_layer_0
grid2mesh_gnn/~_networks_builder/encoder_nodes_mesh_nodes_mlp/act_layer_1
grid2mesh_gnn/~_networks_builder/encoder_edges_grid2mesh_mlp/act_layer_0
grid2mesh_gnn/~_networks_builder/encoder_edges_grid2mesh_mlp/act_layer_1
grid2mesh_gnn/~_networks_builder/processor_edges_0_grid2mesh_mlp/act_layer_0
grid2mesh_gnn/~_networks_builder/processor_edges_0_grid2mesh_mlp/act_layer_1
grid2mesh_gnn/~_networks_builder/processor_nodes_0_grid_nodes_mlp/act_layer_0
grid2mesh_gnn/~_networks_builder/processor_nodes_0_grid_nodes_mlp/act_layer_1
grid2mesh_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/act_layer_0
grid2mesh_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/act_layer_1

mesh_gnn/~_networks_builder/encoder_edges_mesh_mlp/act_layer_0
mesh_gnn/~_networks_builder/encoder_edges_mesh_mlp/act_layer_1
mesh_gnn/~_networks_builder/processor_edges_0_mesh_mlp/act_layer_0
mesh_gnn/~_networks_builder/processor_edges_0_mesh_mlp/act_layer_1
mesh_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/act_layer_0
mesh_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/act_layer_1
mesh_gnn/~_networks_builder/processor_edges_1_mesh_mlp/act_layer_0
mesh_gnn/~_networks_builder/processor_edges_1_mesh_mlp/act_layer_1
mesh_gnn/~_networks_builder/processor_nodes_1_mesh_nodes_mlp/act_layer_0
mesh_gnn/~_networks_builder/processor_nodes_1_mesh_nodes_mlp/act_layer_1
mesh_gnn/~_networks_builder/processor_edges_2_mesh_mlp/act_layer_0
mesh_gnn/~_networks_builder/processor_edges_2_mesh_mlp/act_layer_1
mesh_gnn/~_networks_builder/processor_nodes_2_mesh_nodes_mlp/act_layer_0
mesh_gnn/~_networks_builder/processor_nodes_2_mesh_nodes_mlp/act_layer_1
mesh_gnn/~_networks_builder/processor_edges_3_mesh_mlp/act_layer_0
mesh_gnn/~_networks_builder/processor_edges_3_mesh_mlp/act_layer_1
mesh_gnn/~_networks_builder/processor_nodes_3_mesh_nodes_mlp/act_layer_0
mesh_gnn/~_networks_builder/processor_nodes_3_mesh_nodes_mlp/act_layer_1
mesh_gnn/~_networks_builder/processor_edges_4_mesh_mlp/act_layer_0
mesh_gnn/~_networks_builder/processor_edges_4_mesh_mlp/act_layer_1
mesh_gnn/~_networks_builder/processor_nodes_4_mesh_nodes_mlp/act_layer_0
mesh_gnn/~_networks_builder/processor_nodes_4_mesh_nodes_mlp/act_layer_1
mesh_gnn/~_networks_builder/processor_edges_5_mesh_mlp/act_layer_0
mesh_gnn/~_networks_builder/processor_edges_5_mesh_mlp/act_layer_1
mesh_gnn/~_networks_builder/processor_nodes_5_mesh_nodes_mlp/act_layer_0
mesh_gnn/~_networks_builder/processor_nodes_5_mesh_nodes_mlp/act_layer_1
mesh_gnn/~_networks_builder/processor_edges_6_mesh_mlp/act_layer_0
mesh_gnn/~_networks_builder/processor_edges_6_mesh_mlp/act_layer_1
mesh_gnn/~_networks_builder/processor_nodes_6_mesh_nodes_mlp/act_layer_0
mesh_gnn/~_networks_builder/processor_nodes_6_mesh_nodes_mlp/act_layer_1
mesh_gnn/~_networks_builder/processor_edges_7_mesh_mlp/act_layer_0
mesh_gnn/~_networks_builder/processor_edges_7_mesh_mlp/act_layer_1
mesh_gnn/~_networks_builder/processor_nodes_7_mesh_nodes_mlp/act_layer_0
mesh_gnn/~_networks_builder/processor_nodes_7_mesh_nodes_mlp/act_layer_1
mesh_gnn/~_networks_builder/processor_edges_8_mesh_mlp/act_layer_0
mesh_gnn/~_networks_builder/processor_edges_8_mesh_mlp/act_layer_1
mesh_gnn/~_networks_builder/processor_nodes_8_mesh_nodes_mlp/act_layer_0
mesh_gnn/~_networks_builder/processor_nodes_8_mesh_nodes_mlp/act_layer_1
mesh_gnn/~_networks_builder/processor_edges_9_mesh_mlp/act_layer_0
mesh_gnn/~_networks_builder/processor_edges_9_mesh_mlp/act_layer_1
mesh_gnn/~_networks_builder/processor_nodes_9_mesh_nodes_mlp/act_layer_0
mesh_gnn/~_networks_builder/processor_nodes_9_mesh_nodes_mlp/act_layer_1
mesh_gnn/~_networks_builder/processor_edges_10_mesh_mlp/act_layer_0
mesh_gnn/~_networks_builder/processor_edges_10_mesh_mlp/act_layer_1
mesh_gnn/~_networks_builder/processor_nodes_10_mesh_nodes_mlp/act_layer_0
mesh_gnn/~_networks_builder/processor_nodes_10_mesh_nodes_mlp/act_layer_1
mesh_gnn/~_networks_builder/processor_edges_11_mesh_mlp/act_layer_0
mesh_gnn/~_networks_builder/processor_edges_11_mesh_mlp/act_layer_1
mesh_gnn/~_networks_builder/processor_nodes_11_mesh_nodes_mlp/act_layer_0
mesh_gnn/~_networks_builder/processor_nodes_11_mesh_nodes_mlp/act_layer_1
mesh_gnn/~_networks_builder/processor_edges_12_mesh_mlp/act_layer_0
mesh_gnn/~_networks_builder/processor_edges_12_mesh_mlp/act_layer_1
mesh_gnn/~_networks_builder/processor_nodes_12_mesh_nodes_mlp/act_layer_0
mesh_gnn/~_networks_builder/processor_nodes_12_mesh_nodes_mlp/act_layer_1
mesh_gnn/~_networks_builder/processor_edges_13_mesh_mlp/act_layer_0
mesh_gnn/~_networks_builder/processor_edges_13_mesh_mlp/act_layer_1
mesh_gnn/~_networks_builder/processor_nodes_13_mesh_nodes_mlp/act_layer_0
mesh_gnn/~_networks_builder/processor_nodes_13_mesh_nodes_mlp/act_layer_1
mesh_gnn/~_networks_builder/processor_edges_14_mesh_mlp/act_layer_0
mesh_gnn/~_networks_builder/processor_edges_14_mesh_mlp/act_layer_1
mesh_gnn/~_networks_builder/processor_nodes_14_mesh_nodes_mlp/act_layer_0
mesh_gnn/~_networks_builder/processor_nodes_14_mesh_nodes_mlp/act_layer_1
mesh_gnn/~_networks_builder/processor_edges_15_mesh_mlp/act_layer_0
mesh_gnn/~_networks_builder/processor_edges_15_mesh_mlp/act_layer_1
mesh_gnn/~_networks_builder/processor_nodes_15_mesh_nodes_mlp/act_layer_0
mesh_gnn/~_networks_builder/processor_nodes_15_mesh_nodes_mlp/act_layer_1

mesh2grid_gnn/~_networks_builder/encoder_edges_mesh2grid_mlp/act_layer_0
mesh2grid_gnn/~_networks_builder/encoder_edges_mesh2grid_mlp/act_layer_1
mesh2grid_gnn/~_networks_builder/processor_edges_0_mesh2grid_mlp/act_layer_0
mesh2grid_gnn/~_networks_builder/processor_edges_0_mesh2grid_mlp/act_layer_1
mesh2grid_gnn/~_networks_builder/processor_nodes_0_grid_nodes_mlp/act_layer_0
mesh2grid_gnn/~_networks_builder/processor_nodes_0_grid_nodes_mlp/act_layer_1
mesh2grid_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/act_layer_0
mesh2grid_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/act_layer_1
mesh2grid_gnn/~_networks_builder/decoder_nodes_grid_nodes_mlp/act_layer_0
mesh2grid_gnn/~_networks_builder/decoder_nodes_grid_nodes_mlp/act_layer_1