from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .dataset_utils import (
    open_nc,
    normalize_dims,
    expand_batch_time,
    concat_time,
    to_unix_seconds,
)


@dataclass
class TransformConfig:
    root: Path
    raw_dir: Path
    processed_dir: Path
    resolution: float
    levels: int


class Era5ToGraphcastTransformer:
    def __init__(self, cfg: TransformConfig):
        self.cfg = cfg

    def _paths_for_time(self, ts: str) -> Dict[str, Path]:
        base = self.cfg.raw_dir
        return {
            "pressure": base / "pressure" / f"{ts}.nc",
            "single_instant": base / "single" / "instant" / f"{ts}.nc",
            "single_accum": base / "single" / "accum" / f"{ts}.nc",
        }

    def _load_pressure_vars(self, timestamps: Sequence[str]) -> Dict[str, np.ndarray]:
        import xarray as xr  # type: ignore
        vars_map = {"t": "temperature", "z": "geopotential", "u": "u_component_of_wind",
                    "v": "v_component_of_wind", "w": "vertical_velocity", "q": "specific_humidity"}
        stacked: Dict[str, xr.DataArray] = {}
        for ts in timestamps:
            p = self._paths_for_time(ts)["pressure"]
            if not p.exists():
                raise FileNotFoundError(f"pressure file missing: {p}")
            ds = open_nc(p)
            try:
                ds = normalize_dims(ds)
                # Sort level dimension in ascending order if present
                if "level" in ds.coords:
                    level_values = ds.coords["level"].values
                    sort_idx = np.argsort(level_values)
                    ds = ds.isel(level=sort_idx)
                
                for src, dst in vars_map.items():
                    if src in ds.data_vars:
                        da = ds[src]
                        da = da.rename({k: k for k in da.dims})
                        da = expand_batch_time(da)
                        stacked.setdefault(dst, []).append(da)
            finally:
                try:
                    ds.close()
                except Exception:
                    pass

        out: Dict[str, xr.DataArray] = {}
        for dst, arrays in stacked.items():
            out[dst] = concat_time(arrays)
        return out

    def _load_single_instant_vars(self, timestamps: Sequence[str]) -> Dict[str, np.ndarray]:
        import xarray as xr  # type: ignore
        vars_map = {"t2m": "2m_temperature", "msl": "mean_sea_level_pressure",
                    "u10": "10m_u_component_of_wind", "v10": "10m_v_component_of_wind"}
        stacked: Dict[str, xr.DataArray] = {}
        for ts in timestamps:
            p = self._paths_for_time(ts)["single_instant"]
            if not p.exists():
                raise FileNotFoundError(f"single/instant file missing: {p}")
            ds = open_nc(p)
            try:
                ds = normalize_dims(ds)
                for src, dst in vars_map.items():
                    if src in ds.data_vars:
                        da = ds[src]
                        da = expand_batch_time(da)
                        stacked.setdefault(dst, []).append(da)
            finally:
                try:
                    ds.close()
                except Exception:
                    pass

        out: Dict[str, xr.DataArray] = {}
        for dst, arrays in stacked.items():
            out[dst] = concat_time(arrays)
        return out

    def _load_single_accum_vars(self, timestamps: Sequence[str]) -> Dict[str, np.ndarray]:
        import xarray as xr  # type: ignore
        stacked: Dict[str, xr.DataArray] = {}
        # Build 6-hour accumulation by summing 6 consecutive 1-hour 'tp' files
        # ending at each timestamp (inclusive).
        for ts in timestamps:
            # ts: YYYYMMDDHH string
            import pandas as pd  # type: ignore
            t = pd.to_datetime(ts, format="%Y%m%d%H", utc=True)
            hourly_paths = []
            for i in range(0, 6):  # ts, ts-1h, ..., ts-5h
                tt = t - pd.Timedelta(hours=i)
                ts_hour = tt.strftime("%Y%m%d%H")
                p = self._paths_for_time(ts_hour)["single_accum"]
                if not p.exists():
                    raise FileNotFoundError(f"hourly tp file missing: {p}")
                hourly_paths.append(p)
            # Load 6 hourly files and sum
            hourly_arrays: list[xr.DataArray] = []
            for p in hourly_paths:
                ds = open_nc(p)
                try:
                    ds = normalize_dims(ds)
                    if "tp" not in ds.data_vars:
                        raise KeyError(f"'tp' not found in {p}")
                    da = ds["tp"].astype("float32")
                    # If a time dim exists (length 1), drop it for accumulation to avoid collision.
                    if "time" in da.dims:
                        da = da.isel(time=0, drop=True)
                    # Add a temporary aggregation dimension and collect
                    hourly_arrays.append(da.expand_dims({"agg": 1}))
                finally:
                    try:
                        ds.close()
                    except Exception:
                        pass
            # Sum over the temporary aggregation dimension to get 6-hour accumulation
            tp6 = xr.concat(hourly_arrays, dim="agg").sum(dim="agg")
            tp6 = expand_batch_time(tp6)  # add batch/time dims (time will be length 1 here)
            stacked.setdefault("total_precipitation_6hr", []).append(tp6)

        out: Dict[str, xr.DataArray] = {}
        for dst, arrays in stacked.items():
            out[dst] = concat_time(arrays)  # concat across the 3 target times
        return out

    def _build_coords(self, timestamps: Sequence[str]):
        import xarray as xr  # type: ignore
        import pandas as pd  # type: ignore
        # Use one of the pressure files to grab coords
        p0 = self._paths_for_time(timestamps[-1])["pressure"]
        ds = open_nc(p0)
        try:
            ds = normalize_dims(ds)
            lat = ds["lat"].astype("float32")
            lon = ds["lon"].astype("float32")
            level = None
            if "level" in ds.coords:
                # Sort level in ascending order (50, 100, ..., 1000) to match cache format
                level_values = ds.coords["level"].values
                level = np.sort(level_values).astype("int32")
        finally:
            try:
                ds.close()
            except Exception:
                pass
        time_vals = to_unix_seconds(timestamps)
        
        # Convert to hours offset format to match cache format
        # Cache stores time as int32 with units='hours' (0, 6, 12), which xarray decodes as timedelta64[ns]
        # Calculate hours offset from the first timestamp
        first_time = time_vals[0]
        time_vals_hours = ((time_vals - first_time) / 3600).astype("int32")
        
        # Convert timestamps to pandas Timestamp for datetime coordinate
        datetime_vals = []
        for ts in timestamps:
            if isinstance(ts, (int, np.integer)):
                s = str(ts)
            else:
                s = str(ts)
            # Parse YYYYMMDDHH format
            if len(s) == 10:
                dt = pd.Timestamp(s[:4] + "-" + s[4:6] + "-" + s[6:8] + " " + s[8:10] + ":00:00", tz="UTC")
            else:
                dt = pd.to_datetime(s, utc=True)
            # Convert to tz-naive (solar_radiation.py expects tz-naive datetime)
            datetime_vals.append(dt.tz_localize(None) if dt.tz is not None else dt)
        
        coords = {
            "lon": ("lon", lon.values.astype("float32"), {"units": "degrees_east"}),
            "lat": ("lat", lat.values.astype("float32"), {"units": "degrees_north"}),
            "time": ("time", time_vals_hours, {"units": "hours"}),  # int32 with units='hours' to match cache format
            "datetime": ("time", datetime_vals),  # datetime coordinate for toa_incident_solar_radiation
        }
        if level is not None:
            coords["level"] = ("level", level, {})
        # Store datetime_vals for later use
        coords["_datetime_vals"] = datetime_vals
        return coords

    def _static_fields(self, timestamps: Sequence[str]):
        import xarray as xr  # type: ignore
        # Use single/instant z as geopotential_at_surface
        p = self._paths_for_time(timestamps[-1])["single_instant"]
        ds = open_nc(p)
        try:
            ds = normalize_dims(ds)
            z_sfc = ds["z"] if "z" in ds.data_vars else None
            if z_sfc is not None and "time" in z_sfc.dims:
                z_sfc = z_sfc.isel(time=0, drop=True)
            lsm = ds["lsm"] if "lsm" in ds.data_vars else None
            if lsm is not None and "time" in lsm.dims:
                lsm = lsm.isel(time=0, drop=True)
        finally:
            try:
                ds.close()
            except Exception:
                pass

        fields: Dict[str, xr.DataArray] = {}
        if z_sfc is not None:
            fields["geopotential_at_surface"] = z_sfc.astype("float32").transpose("lat", "lon")
        if lsm is not None:
            fields["land_sea_mask"] = lsm.astype("float32").transpose("lat", "lon")
        return fields

    def build_dataset(self, timestamps: Sequence[str]):
        import xarray as xr  # type: ignore
        coords = self._build_coords(timestamps)
        data_vars: Dict[str, xr.DataArray] = {}
        data_vars.update(self._load_pressure_vars(timestamps))
        data_vars.update(self._load_single_instant_vars(timestamps))
        data_vars.update(self._load_single_accum_vars(timestamps))
        data_vars.update(self._static_fields(timestamps))

        # Extract datetime_vals before creating Dataset (it's stored in coords dict)
        datetime_vals = coords.pop("_datetime_vals", None)
        
        # Update time coordinates in all data_vars to match the new time coordinate
        # The data_vars have time coordinates from concat_time, but we need to align them
        # with the new time coordinate [0, 6, 12]
        # coords["time"] is a tuple: (dim_name, values, attrs)
        new_time_values = coords["time"][1]  # Get the time coordinate values
        for var_name, var_data in data_vars.items():
            if "time" in var_data.dims:
                # Replace time coordinate with the new one
                var_data = var_data.assign_coords(time=new_time_values)
                data_vars[var_name] = var_data
        
        ds_out = xr.Dataset(data_vars=data_vars, coords=coords)
        
        # Add toa_incident_solar_radiation first (needs datetime coordinate)
        if datetime_vals is not None:
            # Check if dataset has batch dimension
            # If batch dimension exists, create datetime coord with (batch, time) shape
            # to match cache format and avoid dimension mismatch in add_derived_vars
            if "batch" in ds_out.dims:
                # Reshape datetime_vals to (batch, time) format
                # datetime_vals is a list of length time, need to add batch dimension
                batch_size = ds_out.dims["batch"]
                # Create 2D array: shape (batch, time)
                datetime_vals_2d = [datetime_vals] * batch_size  # List of lists
                datetime_coord = xr.DataArray(datetime_vals_2d, dims=("batch", "time"))
            else:
                # No batch dimension, use 1D format
                datetime_coord = xr.DataArray(datetime_vals, dims=("time",))
            ds_out = ds_out.assign_coords(datetime=datetime_coord)
            
            try:
                from graphcast.data_utils import add_tisr_var
                add_tisr_var(ds_out)
            except ImportError:
                # If graphcast module is not available, skip adding toa_incident_solar_radiation
                pass
        
        # Keep datetime as coordinate to match cache format
        # Cache files store datetime as data_var in NetCDF, but xarray decodes it as coords
        # We keep it as coords directly to ensure it's available for add_derived_vars
        # The datetime coords should have (batch, time) shape with datetime64[ns] dtype
        # This matches what xarray decodes from cache files
        
        return ds_out

    def output_path_for_anchor(self, anchor_ts_str: str) -> Path:
        out_dir = self.cfg.processed_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        # anchor_ts_str expected format: YYYY-MM-DDTHH
        return out_dir / f"source-era5_date-{anchor_ts_str}_res-{self.cfg.resolution}_levels-{self.cfg.levels}_steps-01.nc"


