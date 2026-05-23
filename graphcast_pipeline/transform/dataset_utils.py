from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _try_import_xarray():
    import xarray as xr  # type: ignore
    return xr


def open_nc(path: Path):
    xr = _try_import_xarray()
    return xr.open_dataset(str(path), decode_cf=False)


def ensure_lat_ascending(ds):
    xr = _try_import_xarray()
    lat_name = None
    for cand in ["lat", "latitude"]:
        if cand in ds.coords:
            lat_name = cand
            break
    if lat_name is None:
        return ds
    lat_vals = ds[lat_name].values
    if lat_vals[0] > lat_vals[-1]:
        return ds.sortby(lat_name)
    return ds


DIM_RENAMES = {
    "valid_time": "time",
    "pressure_level": "level",
    "latitude": "lat",
    "longitude": "lon",
}


def normalize_dims(ds):
    # Rename known dims/coords
    rename_map = {k: v for k, v in DIM_RENAMES.items() if k in ds.dims or k in ds.coords}
    if rename_map:
        ds = ds.rename(rename_map)
    ds = ensure_lat_ascending(ds)
    return ds


def expand_batch_time(da, batch: int = 1):
    import xarray as xr  # type: ignore
    if "time" not in da.dims:
        da = da.expand_dims({"time": 1})
    da = da.expand_dims({"batch": batch})
    # Reorder: batch, time, [level], lat, lon
    dims_order = [d for d in ["batch", "time", "level", "lat", "lon"] if d in da.dims]
    return da.transpose(*dims_order)


def concat_time(arrays: Sequence):
    import xarray as xr  # type: ignore
    arrays = [a if "time" in a.dims else a.expand_dims({"time": 1}) for a in arrays]
    return xr.concat(arrays, dim="time")


def to_unix_seconds(timestamps: Sequence) -> np.ndarray:
    import pandas as pd  # type: ignore
    values: List[int] = []
    for t in timestamps:
        if isinstance(t, (int, np.integer)):
            s = str(t)
        else:
            s = str(t)
        # Accept YYYYMMDDHH or YYYY-MM-DD or full ISO strings
        try:
            if re.fullmatch(r"\d{10}", s):
                dt = pd.to_datetime(s, format="%Y%m%d%H", utc=True)
            elif re.fullmatch(r"\d{8}", s):
                dt = pd.to_datetime(s, format="%Y%m%d", utc=True)
            else:
                dt = pd.to_datetime(s, utc=True)
        except Exception:
            # Fallback
            dt = pd.Timestamp(s)
        values.append(int(dt.timestamp()))
    return np.asarray(values, dtype=np.int32)


def safe_get(ds, name: str):
    return ds[name] if name in ds.variables else None


def read_single_variable(ds_path: Path, var_name: str):
    ds = open_nc(ds_path)
    try:
        ds = normalize_dims(ds)
        da = safe_get(ds, var_name)
        return da
    finally:
        try:
            ds.close()
        except Exception:
            pass


def try_read_first(paths: Sequence[Path], var_name: str):
    for p in paths:
        if not p.exists():
            continue
        da = read_single_variable(p, var_name)
        if da is not None:
            return da
    return None


