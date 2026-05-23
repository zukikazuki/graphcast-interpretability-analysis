# 素のデータ構造 (raw NetCDF/Zarr schema)

## pressure

- sample: `/media/dl-box/SSD-SCTU3A/graphcast_data/data/raw/era5/pressure/2021123112.nc`
### dims

| name | size |
|---|---|
| `valid_time` | `1` |
| `pressure_level` | `13` |
| `latitude` | `181` |
| `longitude` | `360` |

### coords

| name | dims | shape | dtype |
|---|---|---|---|
| `valid_time` | `('valid_time',)` | `(1,)` | `int64` |
| `pressure_level` | `('pressure_level',)` | `(13,)` | `float64` |
| `latitude` | `('latitude',)` | `(181,)` | `float64` |
| `longitude` | `('longitude',)` | `(360,)` | `float64` |

### data_vars

| name | dims | shape | dtype |
|---|---|---|---|
| `number` | `()` | `()` | `int64` |
| `expver` | `()` | `()` | `<U4` |
| `t` | `('valid_time', 'pressure_level', 'latitude', 'longitude')` | `(1, 13, 181, 360)` | `float32` |
| `z` | `('valid_time', 'pressure_level', 'latitude', 'longitude')` | `(1, 13, 181, 360)` | `float32` |
| `u` | `('valid_time', 'pressure_level', 'latitude', 'longitude')` | `(1, 13, 181, 360)` | `float32` |
| `v` | `('valid_time', 'pressure_level', 'latitude', 'longitude')` | `(1, 13, 181, 360)` | `float32` |
| `w` | `('valid_time', 'pressure_level', 'latitude', 'longitude')` | `(1, 13, 181, 360)` | `float32` |
| `q` | `('valid_time', 'pressure_level', 'latitude', 'longitude')` | `(1, 13, 181, 360)` | `float32` |

### global_attrs

- `GRIB_centre`: `ecmf`
- `GRIB_centreDescription`: `European Centre for Medium-Range Weather Forecasts`
- `GRIB_subCentre`: `0`
- `Conventions`: `CF-1.7`
- `institution`: `European Centre for Medium-Range Weather Forecasts`
- `history`: `2025-11-09T02:31 GRIB to CDM+CF via cfgrib-0.9.15.0/ecCodes-2.42.0 with {"source": "tmp35fxt0b6/data.grib", "filter_by_keys": {"stream": ["oper"], "stepType": ["instant"]}, "encode_cf": ["parameter", "time", "geography", "vertical"]}`

## single/accum

- sample: `/media/dl-box/SSD-SCTU3A/graphcast_data/data/raw/era5/single/accum/2021123112.nc`
### dims

| name | size |
|---|---|
| `valid_time` | `1` |
| `latitude` | `181` |
| `longitude` | `360` |

### coords

| name | dims | shape | dtype |
|---|---|---|---|
| `valid_time` | `('valid_time',)` | `(1,)` | `int64` |
| `latitude` | `('latitude',)` | `(181,)` | `float64` |
| `longitude` | `('longitude',)` | `(360,)` | `float64` |

### data_vars

| name | dims | shape | dtype |
|---|---|---|---|
| `number` | `()` | `()` | `int64` |
| `expver` | `()` | `()` | `<U4` |
| `tp` | `('valid_time', 'latitude', 'longitude')` | `(1, 181, 360)` | `float32` |

### global_attrs

- `GRIB_centre`: `ecmf`
- `GRIB_centreDescription`: `European Centre for Medium-Range Weather Forecasts`
- `GRIB_subCentre`: `0`
- `Conventions`: `CF-1.7`
- `institution`: `European Centre for Medium-Range Weather Forecasts`
- `history`: `2025-11-09T02:30 GRIB to CDM+CF via cfgrib-0.9.15.0/ecCodes-2.42.0 with {"source": "tmp5y2m5eh5/data.grib", "filter_by_keys": {"stream": ["oper"], "stepType": ["accum"]}, "encode_cf": ["parameter", "time", "geography", "vertical"]}`

## single/instant

- sample: `/media/dl-box/SSD-SCTU3A/graphcast_data/data/raw/era5/single/instant/2021123112.nc`
### dims

| name | size |
|---|---|
| `valid_time` | `1` |
| `latitude` | `181` |
| `longitude` | `360` |

### coords

| name | dims | shape | dtype |
|---|---|---|---|
| `valid_time` | `('valid_time',)` | `(1,)` | `int64` |
| `latitude` | `('latitude',)` | `(181,)` | `float64` |
| `longitude` | `('longitude',)` | `(360,)` | `float64` |

### data_vars

| name | dims | shape | dtype |
|---|---|---|---|
| `number` | `()` | `()` | `int64` |
| `expver` | `()` | `()` | `<U4` |
| `t2m` | `('valid_time', 'latitude', 'longitude')` | `(1, 181, 360)` | `float32` |
| `msl` | `('valid_time', 'latitude', 'longitude')` | `(1, 181, 360)` | `float32` |
| `u10` | `('valid_time', 'latitude', 'longitude')` | `(1, 181, 360)` | `float32` |
| `v10` | `('valid_time', 'latitude', 'longitude')` | `(1, 181, 360)` | `float32` |
| `z` | `('valid_time', 'latitude', 'longitude')` | `(1, 181, 360)` | `float32` |
| `lsm` | `('valid_time', 'latitude', 'longitude')` | `(1, 181, 360)` | `float32` |

### global_attrs

- `GRIB_centre`: `ecmf`
- `GRIB_centreDescription`: `European Centre for Medium-Range Weather Forecasts`
- `GRIB_subCentre`: `0`
- `Conventions`: `CF-1.7`
- `institution`: `European Centre for Medium-Range Weather Forecasts`
- `history`: `2025-11-09T02:29 GRIB to CDM+CF via cfgrib-0.9.15.0/ecCodes-2.42.0 with {"source": "tmpqgnowzl5/data.grib", "filter_by_keys": {"stream": ["oper"], "stepType": ["instant"]}, "encode_cf": ["parameter", "time", "geography", "vertical"]}`

## cache

- sample: `/home/dl-box/Fujikawa/research/machine_learning/graphcast/graphcast_cache/source-era5_date-2022-01-01_res-1.0_levels-13_steps-01.nc`
### dims

| name | size |
|---|---|
| `lon` | `360` |
| `lat` | `181` |
| `level` | `13` |
| `time` | `3` |
| `batch` | `1` |

### coords

| name | dims | shape | dtype |
|---|---|---|---|
| `lon` | `('lon',)` | `(360,)` | `float32` |
| `lat` | `('lat',)` | `(181,)` | `float32` |
| `level` | `('level',)` | `(13,)` | `int32` |
| `time` | `('time',)` | `(3,)` | `int32` |

### data_vars

| name | dims | shape | dtype |
|---|---|---|---|
| `geopotential_at_surface` | `('lat', 'lon')` | `(181, 360)` | `float32` |
| `land_sea_mask` | `('lat', 'lon')` | `(181, 360)` | `float32` |
| `2m_temperature` | `('batch', 'time', 'lat', 'lon')` | `(1, 3, 181, 360)` | `float32` |
| `mean_sea_level_pressure` | `('batch', 'time', 'lat', 'lon')` | `(1, 3, 181, 360)` | `float32` |
| `10m_v_component_of_wind` | `('batch', 'time', 'lat', 'lon')` | `(1, 3, 181, 360)` | `float32` |
| `10m_u_component_of_wind` | `('batch', 'time', 'lat', 'lon')` | `(1, 3, 181, 360)` | `float32` |
| `total_precipitation_6hr` | `('batch', 'time', 'lat', 'lon')` | `(1, 3, 181, 360)` | `float32` |
| `toa_incident_solar_radiation` | `('batch', 'time', 'lat', 'lon')` | `(1, 3, 181, 360)` | `float32` |
| `temperature` | `('batch', 'time', 'level', 'lat', 'lon')` | `(1, 3, 13, 181, 360)` | `float32` |
| `geopotential` | `('batch', 'time', 'level', 'lat', 'lon')` | `(1, 3, 13, 181, 360)` | `float32` |
| `u_component_of_wind` | `('batch', 'time', 'level', 'lat', 'lon')` | `(1, 3, 13, 181, 360)` | `float32` |
| `v_component_of_wind` | `('batch', 'time', 'level', 'lat', 'lon')` | `(1, 3, 13, 181, 360)` | `float32` |
| `vertical_velocity` | `('batch', 'time', 'level', 'lat', 'lon')` | `(1, 3, 13, 181, 360)` | `float32` |
| `specific_humidity` | `('batch', 'time', 'level', 'lat', 'lon')` | `(1, 3, 13, 181, 360)` | `float32` |
| `datetime` | `('batch', 'time')` | `(1, 3)` | `int32` |

### global_attrs

(none)

## processed

- sample: `/media/dl-box/SSD-SCTU3A/graphcast_data/data/processed/graphcast_inputs/source-era5_date-2022-01-01T00_res-1.0_levels-13_steps-01.nc`
### dims

| name | size |
|---|---|
| `time` | `3` |
| `level` | `13` |
| `lat` | `181` |
| `lon` | `360` |
| `batch` | `1` |

### coords

| name | dims | shape | dtype |
|---|---|---|---|
| `time` | `('time',)` | `(3,)` | `int32` |
| `level` | `('level',)` | `(13,)` | `int32` |
| `lat` | `('lat',)` | `(181,)` | `float32` |
| `lon` | `('lon',)` | `(360,)` | `float32` |

### data_vars

| name | dims | shape | dtype |
|---|---|---|---|
| `temperature` | `('batch', 'time', 'level', 'lat', 'lon')` | `(1, 3, 13, 181, 360)` | `float32` |
| `geopotential` | `('batch', 'time', 'level', 'lat', 'lon')` | `(1, 3, 13, 181, 360)` | `float32` |
| `u_component_of_wind` | `('batch', 'time', 'level', 'lat', 'lon')` | `(1, 3, 13, 181, 360)` | `float32` |
| `v_component_of_wind` | `('batch', 'time', 'level', 'lat', 'lon')` | `(1, 3, 13, 181, 360)` | `float32` |
| `vertical_velocity` | `('batch', 'time', 'level', 'lat', 'lon')` | `(1, 3, 13, 181, 360)` | `float32` |
| `specific_humidity` | `('batch', 'time', 'level', 'lat', 'lon')` | `(1, 3, 13, 181, 360)` | `float32` |
| `2m_temperature` | `('batch', 'time', 'lat', 'lon')` | `(1, 3, 181, 360)` | `float32` |
| `mean_sea_level_pressure` | `('batch', 'time', 'lat', 'lon')` | `(1, 3, 181, 360)` | `float32` |
| `10m_u_component_of_wind` | `('batch', 'time', 'lat', 'lon')` | `(1, 3, 181, 360)` | `float32` |
| `10m_v_component_of_wind` | `('batch', 'time', 'lat', 'lon')` | `(1, 3, 181, 360)` | `float32` |
| `total_precipitation_6hr` | `('batch', 'time', 'lat', 'lon')` | `(1, 3, 181, 360)` | `float32` |
| `geopotential_at_surface` | `('lat', 'lon')` | `(181, 360)` | `float32` |
| `land_sea_mask` | `('lat', 'lon')` | `(181, 360)` | `float32` |
| `toa_incident_solar_radiation` | `('batch', 'time', 'lat', 'lon')` | `(1, 3, 181, 360)` | `float32` |
| `datetime` | `('batch', 'time')` | `(1, 3)` | `int32` |

### global_attrs

(none)
