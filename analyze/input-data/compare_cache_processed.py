#!/usr/bin/env python3
"""Compare cache and processed datasets to verify they are identical.

結果をMarkdown形式で出力します。
"""

import xarray as xr
import numpy as np
from pathlib import Path
from typing import List, Tuple
from datetime import datetime


# パス設定
CACHE_DIR = Path("/home/dl-box/Fujikawa/research/machine_learning/graphcast/graphcast_cache")
PROCESSED_DIR = Path("/media/dl-box/SSD-SCTU3A/graphcast_data/data/processed/graphcast_inputs")
OUTPUT_MD = Path(__file__).parent / "compare_cache_processed_result.md"


def compare_datasets(cache_path: Path, processed_path: Path, rtol: float = 1e-5, atol: float = 1e-8) -> dict:
    """Compare two datasets and return comparison results."""
    
    results = {
        "cache_path": str(cache_path),
        "processed_path": str(processed_path),
        "structure_match": True,
        "values_match": True,
        "errors": [],
        "warnings": [],
        "var_comparisons": {},
        "coord_comparisons": {},
    }
    
    cache_ds = xr.open_dataset(cache_path, decode_cf=False)
    processed_ds = xr.open_dataset(processed_path, decode_cf=False)
    
    try:
        # 1. Structure comparison
        cache_dims = set(cache_ds.dims.keys())
        processed_dims = set(processed_ds.dims.keys())
        if cache_dims != processed_dims:
            results["structure_match"] = False
            results["errors"].append(f"Different dims: cache={cache_dims}, processed={processed_dims}")
        
        cache_coords = set(cache_ds.coords.keys())
        processed_coords = set(processed_ds.coords.keys())
        if cache_coords != processed_coords:
            results["structure_match"] = False
            results["errors"].append(f"Different coords: cache={cache_coords}, processed={processed_coords}")
        
        cache_vars = set(cache_ds.data_vars.keys())
        processed_vars = set(processed_ds.data_vars.keys())
        if cache_vars != processed_vars:
            results["structure_match"] = False
            only_cache = cache_vars - processed_vars
            only_processed = processed_vars - cache_vars
            if only_cache:
                results["errors"].append(f"Only in cache: {only_cache}")
            if only_processed:
                results["errors"].append(f"Only in processed: {only_processed}")
        
        # 2. Compare values for each variable
        common_vars = cache_vars & processed_vars
        for var_name in sorted(common_vars):
            cache_var = cache_ds[var_name]
            processed_var = processed_ds[var_name]
            
            var_result = {
                "dims_match": False,
                "shape_match": False,
                "dtype_match": False,
                "values_match": False,
                "max_diff": None,
                "mean_diff": None,
                "cache_mean": None,
                "processed_mean": None,
                "errors": [],
                "cache_sample": None,
                "processed_sample": None,
                "cache_coords": None,
                "processed_coords": None,
            }
            
            # Store coords for each variable
            try:
                cache_coords_list = sorted(list(cache_var.coords.keys()))
                var_result["cache_coords"] = cache_coords_list
            except Exception:
                var_result["cache_coords"] = []
            
            try:
                processed_coords_list = sorted(list(processed_var.coords.keys()))
                var_result["processed_coords"] = processed_coords_list
            except Exception:
                var_result["processed_coords"] = []
            
            # Check dims
            if cache_var.dims != processed_var.dims:
                var_result["errors"].append(f"Different dims: cache={cache_var.dims}, processed={processed_var.dims}")
            else:
                var_result["dims_match"] = True
            
            # Check shape
            if cache_var.shape != processed_var.shape:
                var_result["errors"].append(f"Different shapes: cache={cache_var.shape}, processed={processed_var.shape}")
            else:
                var_result["shape_match"] = True
            
            # Check dtype
            if cache_var.dtype != processed_var.dtype:
                var_result["warnings"] = [f"Different dtypes: cache={cache_var.dtype}, processed={processed_var.dtype}"]
            else:
                var_result["dtype_match"] = True
            
            # Store sample values for detailed comparison (especially for datetime)
            try:
                if var_name == "datetime" or cache_var.size <= 10:
                    var_result["cache_sample"] = cache_var.values.flatten().tolist()
                    var_result["processed_sample"] = processed_var.values.flatten().tolist()
            except Exception:
                pass
            
            # Compare values
            if var_result["dims_match"] and var_result["shape_match"]:
                try:
                    # Check NaN counts
                    cache_nan_count = int(np.isnan(cache_var.values).sum())
                    processed_nan_count = int(np.isnan(processed_var.values).sum())
                    var_result["cache_nan_count"] = cache_nan_count
                    var_result["processed_nan_count"] = processed_nan_count
                    var_result["cache_total"] = int(cache_var.size)
                    var_result["processed_total"] = int(processed_var.size)
                    
                    # Align dimensions
                    cache_aligned = cache_var.transpose(*processed_var.dims)
                    
                    # Cache/Processed means (for context)
                    try:
                        var_result["cache_mean"] = float(np.nanmean(cache_aligned.values))
                    except Exception:
                        var_result["cache_mean"] = None
                    try:
                        var_result["processed_mean"] = float(np.nanmean(processed_var.values))
                    except Exception:
                        var_result["processed_mean"] = None
                    
                    if np.issubdtype(cache_var.dtype, np.floating):
                        match = np.allclose(
                            cache_aligned.values,
                            processed_var.values,
                            rtol=rtol,
                            atol=atol,
                            equal_nan=True
                        )
                        if not match:
                            diff = np.abs(cache_aligned.values - processed_var.values)
                            # Check if diff contains any valid (non-NaN) values
                            if np.any(~np.isnan(diff)):
                                try:
                                    var_result["max_diff"] = float(np.nanmax(diff))
                                except Exception:
                                    var_result["max_diff"] = None
                                try:
                                    var_result["mean_diff"] = float(np.nanmean(diff))
                                except Exception:
                                    var_result["mean_diff"] = None
                            else:
                                # Both are NaN or one is NaN
                                both_nan = np.isnan(cache_aligned.values) & np.isnan(processed_var.values)
                                cache_only_nan = np.isnan(cache_aligned.values) & ~np.isnan(processed_var.values)
                                processed_only_nan = ~np.isnan(cache_aligned.values) & np.isnan(processed_var.values)
                                
                                if np.any(cache_only_nan):
                                    var_result["errors"].append(f"Cache has NaN where processed has values ({np.sum(cache_only_nan)} locations)")
                                elif np.any(processed_only_nan):
                                    var_result["errors"].append(f"Processed has NaN where cache has values ({np.sum(processed_only_nan)} locations)")
                                else:
                                    var_result["errors"].append("All differences are NaN - both datasets have NaN at same locations")
                                
                                # Show sample values for debugging
                                if cache_var.size <= 100:
                                    var_result["cache_sample"] = cache_aligned.values.flatten().tolist()[:20]
                                    var_result["processed_sample"] = processed_var.values.flatten().tolist()[:20]
                                else:
                                    # Show first few values
                                    cache_flat = cache_aligned.values.flatten()
                                    processed_flat = processed_var.values.flatten()
                                    var_result["cache_sample"] = cache_flat[:10].tolist()
                                    var_result["processed_sample"] = processed_flat[:10].tolist()
                                    var_result["cache_sample"] += ["..."] + cache_flat[-5:].tolist()
                                    var_result["processed_sample"] += ["..."] + processed_flat[-5:].tolist()
                        
                        # Add relative mean diff (%) if available
                        try:
                            if (var_result.get("mean_diff") is not None
                                and var_result.get("cache_mean") not in (None, 0.0)):
                                denom = abs(var_result["cache_mean"])
                                var_result["rel_mean_diff_pct"] = float((var_result["mean_diff"] / denom) * 100.0) if denom != 0.0 else None
                            else:
                                var_result["rel_mean_diff_pct"] = None
                        except Exception:
                            var_result["rel_mean_diff_pct"] = None
                    else:
                        match = np.array_equal(cache_aligned.values, processed_var.values)
                        if not match and var_name == "datetime":
                            # For datetime, show the actual values
                            var_result["cache_sample"] = cache_aligned.values.flatten().tolist()
                            var_result["processed_sample"] = processed_var.values.flatten().tolist()
                    
                    var_result["values_match"] = match
                    if not match:
                        results["values_match"] = False
                except Exception as e:
                    var_result["errors"].append(f"Error comparing values: {e}")
                    import traceback
                    var_result["errors"].append(f"Traceback: {traceback.format_exc()}")
                    results["values_match"] = False
            
            results["var_comparisons"][var_name] = var_result
        
        # 3. Compare coordinates
        # Compare common coordinates
        common_coords = cache_coords & processed_coords
        only_cache_coords = cache_coords - processed_coords
        only_processed_coords = processed_coords - cache_coords
        
        # Track missing coordinates
        if only_cache_coords:
            results["warnings"].append(f"Coordinates only in cache: {only_cache_coords}")
        if only_processed_coords:
            results["warnings"].append(f"Coordinates only in processed: {only_processed_coords}")
        
        # Compare common coordinates
        for coord_name in sorted(common_coords):
            cache_coord = cache_ds.coords[coord_name]
            processed_coord = processed_ds.coords[coord_name]
            
            coord_result = {
                "exists_in_cache": True,
                "exists_in_processed": True,
                "dims_match": cache_coord.dims == processed_coord.dims,
                "shape_match": cache_coord.shape == processed_coord.shape,
                "dtype_match": cache_coord.dtype == processed_coord.dtype,
                "values_match": False,
                "max_diff": None,
                "mean_diff": None,
                "cache_dims": cache_coord.dims,
                "processed_dims": processed_coord.dims,
                "cache_shape": cache_coord.shape,
                "processed_shape": processed_coord.shape,
                "cache_dtype": str(cache_coord.dtype),
                "processed_dtype": str(processed_coord.dtype),
                "cache_size": int(cache_coord.size),
                "processed_size": int(processed_coord.size),
                "cache_values": None,
                "processed_values": None,
                "errors": [],
            }
            
            # Store actual values for detailed comparison
            try:
                if cache_coord.size <= 20:
                    coord_result["cache_values"] = cache_coord.values.tolist()
                else:
                    coord_result["cache_values"] = f"[{cache_coord.values[0]}, ..., {cache_coord.values[-1]}] (length={cache_coord.size})"
            except Exception as e:
                coord_result["cache_values"] = f"Error: {e}"
            
            try:
                if processed_coord.size <= 20:
                    coord_result["processed_values"] = processed_coord.values.tolist()
                else:
                    coord_result["processed_values"] = f"[{processed_coord.values[0]}, ..., {processed_coord.values[-1]}] (length={processed_coord.size})"
            except Exception as e:
                coord_result["processed_values"] = f"Error: {e}"
            
            if coord_result["dims_match"] and coord_result["shape_match"]:
                try:
                    if np.issubdtype(cache_coord.dtype, np.floating):
                        match = np.allclose(
                            cache_coord.values,
                            processed_coord.values,
                            rtol=rtol,
                            atol=atol,
                            equal_nan=True
                        )
                        if not match:
                            diff = np.abs(cache_coord.values - processed_coord.values)
                            coord_result["max_diff"] = float(np.nanmax(diff))
                            coord_result["mean_diff"] = float(np.nanmean(diff))
                    else:
                        match = np.array_equal(cache_coord.values, processed_coord.values)
                        if not match:
                            # For integer types, show differences
                            diff = np.abs(cache_coord.values.astype(np.float64) - processed_coord.values.astype(np.float64))
                            coord_result["max_diff"] = float(np.nanmax(diff))
                            coord_result["mean_diff"] = float(np.nanmean(diff))
                    
                    coord_result["values_match"] = match
                    if not match:
                        results["values_match"] = False
                except Exception as e:
                    coord_result["errors"].append(f"Error comparing values: {e}")
                    results["values_match"] = False
            else:
                coord_result["errors"].append(
                    f"Dims or shape mismatch: cache_dims={cache_coord.dims}, processed_dims={processed_coord.dims}, "
                    f"cache_shape={cache_coord.shape}, processed_shape={processed_coord.shape}"
                )
                results["values_match"] = False
            
            results["coord_comparisons"][coord_name] = coord_result
        
        # Add coordinates that only exist in one dataset
        for coord_name in sorted(only_cache_coords):
            cache_coord = cache_ds.coords[coord_name]
            coord_result = {
                "exists_in_cache": True,
                "exists_in_processed": False,
                "dims_match": False,
                "shape_match": False,
                "dtype_match": False,
                "values_match": False,
                "cache_dims": cache_coord.dims,
                "processed_dims": None,
                "cache_shape": cache_coord.shape,
                "processed_shape": None,
                "cache_dtype": str(cache_coord.dtype),
                "processed_dtype": None,
                "cache_size": int(cache_coord.size),
                "processed_size": None,
                "cache_values": None,
                "processed_values": None,
                "errors": [f"Only exists in cache dataset"],
            }
            try:
                if cache_coord.size <= 20:
                    coord_result["cache_values"] = cache_coord.values.tolist()
                else:
                    coord_result["cache_values"] = f"[{cache_coord.values[0]}, ..., {cache_coord.values[-1]}] (length={cache_coord.size})"
            except Exception:
                pass
            results["coord_comparisons"][coord_name] = coord_result
        
        for coord_name in sorted(only_processed_coords):
            processed_coord = processed_ds.coords[coord_name]
            coord_result = {
                "exists_in_cache": False,
                "exists_in_processed": True,
                "dims_match": False,
                "shape_match": False,
                "dtype_match": False,
                "values_match": False,
                "cache_dims": None,
                "processed_dims": processed_coord.dims,
                "cache_shape": None,
                "processed_shape": processed_coord.shape,
                "cache_dtype": None,
                "processed_dtype": str(processed_coord.dtype),
                "cache_size": None,
                "processed_size": int(processed_coord.size),
                "cache_values": None,
                "processed_values": None,
                "errors": [f"Only exists in processed dataset"],
            }
            try:
                if processed_coord.size <= 20:
                    coord_result["processed_values"] = processed_coord.values.tolist()
                else:
                    coord_result["processed_values"] = f"[{processed_coord.values[0]}, ..., {processed_coord.values[-1]}] (length={processed_coord.size})"
            except Exception:
                pass
            results["coord_comparisons"][coord_name] = coord_result
        
    finally:
        cache_ds.close()
        processed_ds.close()
    
    return results


def generate_markdown(results: dict) -> str:
    """Generate Markdown report from comparison results."""
    
    lines = []
    lines.append("# Cache vs Processed Dataset Comparison")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Files Compared")
    lines.append("")
    lines.append(f"- **Cache:** `{results['cache_path']}`")
    lines.append(f"- **Processed:** `{results['processed_path']}`")
    lines.append("")
    
    # Summary
    lines.append("## Summary")
    lines.append("")
    structure_status = "✅ Match" if results["structure_match"] else "❌ Mismatch"
    values_status = "✅ Match" if results["values_match"] else "❌ Mismatch"
    overall_status = "✅ **PASS**" if (results["structure_match"] and results["values_match"]) else "❌ **FAIL**"
    
    lines.append(f"- **Structure:** {structure_status}")
    lines.append(f"- **Values:** {values_status}")
    lines.append(f"- **Overall:** {overall_status}")
    lines.append("")
    
    # Errors
    if results["errors"]:
        lines.append("## Errors")
        lines.append("")
        for error in results["errors"]:
            lines.append(f"- ❌ {error}")
        lines.append("")
    
    # Warnings
    if results["warnings"]:
        lines.append("## Warnings")
        lines.append("")
        for warning in results["warnings"]:
            lines.append(f"- ⚠️  {warning}")
        lines.append("")
    
    # Variable comparisons
    lines.append("## Variable Comparisons")
    lines.append("")
    lines.append("| Variable | Dims Match | Shape Match | Dtype Match | Values Match | Max Diff | Mean Diff | Rel Mean Diff (%) | Cache Mean | Processed Mean | NaN Info | Cache Coords | Processed Coords |")
    lines.append("|----------|------------|-------------|-------------|--------------|----------|-----------|-------------------|------------|----------------|----------|--------------|-----------------|")
    
    def _fmt_float(val):
        if val is None:
            return "-"
        try:
            # Fixed-point with sufficient precision; avoid scientific notation.
            return f"{float(val):.12f}"
        except Exception:
            return str(val)
    
    for var_name, var_result in sorted(results["var_comparisons"].items()):
        dims_ok = "✅" if var_result["dims_match"] else "❌"
        shape_ok = "✅" if var_result["shape_match"] else "❌"
        dtype_ok = "✅" if var_result.get("dtype_match", True) else "⚠️"
        values_ok = "✅" if var_result["values_match"] else "❌"
        max_diff = _fmt_float(var_result["max_diff"])
        mean_diff = _fmt_float(var_result["mean_diff"])
        cache_mean = _fmt_float(var_result.get("cache_mean"))
        rel_mean_diff_pct = _fmt_float(var_result.get("rel_mean_diff_pct"))
        processed_mean = _fmt_float(var_result.get("processed_mean"))
        
        # NaN info
        nan_info = "-"
        if "cache_nan_count" in var_result and "processed_nan_count" in var_result:
            cache_nan = var_result["cache_nan_count"]
            processed_nan = var_result["processed_nan_count"]
            cache_total = var_result.get("cache_total", 0)
            processed_total = var_result.get("processed_total", 0)
            if cache_nan > 0 or processed_nan > 0:
                nan_info = f"C:{cache_nan}/{cache_total} P:{processed_nan}/{processed_total}"
        
        # Coords info
        cache_coords = var_result.get("cache_coords", [])
        processed_coords = var_result.get("processed_coords", [])
        cache_coords_str = ", ".join(cache_coords) if cache_coords else "-"
        processed_coords_str = ", ".join(processed_coords) if processed_coords else "-"
        
        lines.append(f"| `{var_name}` | {dims_ok} | {shape_ok} | {dtype_ok} | {values_ok} | {max_diff} | {mean_diff} | {rel_mean_diff_pct} | {cache_mean} | {processed_mean} | {nan_info} | {cache_coords_str} | {processed_coords_str} |")
        
        # Show sample values for debugging
        if not var_result["values_match"] and var_result.get("cache_sample") is not None:
            cache_sample = var_result.get("cache_sample")
            processed_sample = var_result.get("processed_sample")
            if cache_sample is not None and processed_sample is not None:
                lines.append("")
                lines.append(f"### `{var_name}` Sample Values")
                lines.append("")
                lines.append(f"- **Cache:** `{cache_sample}`")
                lines.append(f"- **Processed:** `{processed_sample}`")
                lines.append("")
        
        if var_result.get("errors"):
            for error in var_result["errors"]:
                lines.append(f"|   └─ ❌ {error} | | | | | | | |")
        if var_result.get("warnings"):
            for warning in var_result["warnings"]:
                lines.append(f"|   └─ ⚠️  {warning} | | | | | | | |")
    
    lines.append("")
    
    # Coordinate comparisons
    lines.append("## Coordinate Comparisons")
    lines.append("")
    lines.append("| Coordinate | In Cache | In Processed | Dims Match | Shape Match | Dtype Match | Values Match | Max Diff | Mean Diff | Cache Dims | Processed Dims | Cache Shape | Processed Shape | Cache Dtype | Processed Dtype | Cache Size | Processed Size |")
    lines.append("|------------|----------|---------------|------------|-------------|-------------|--------------|----------|-----------|------------|----------------|-------------|-----------------|------------|-----------------|------------|----------------|")
    
    for coord_name, coord_result in sorted(results["coord_comparisons"].items()):
        in_cache = "✅" if coord_result.get("exists_in_cache", False) else "❌"
        in_processed = "✅" if coord_result.get("exists_in_processed", False) else "❌"
        dims_ok = "✅" if coord_result.get("dims_match", False) else "❌"
        shape_ok = "✅" if coord_result.get("shape_match", False) else "❌"
        dtype_ok = "✅" if coord_result.get("dtype_match", False) else "❌"
        values_ok = "✅" if coord_result.get("values_match", False) else "❌"
        
        max_diff = "-"
        if coord_result.get("max_diff") is not None:
            try:
                max_diff = f"{coord_result['max_diff']:.6e}"
            except Exception:
                max_diff = str(coord_result.get("max_diff"))
        
        mean_diff = "-"
        if coord_result.get("mean_diff") is not None:
            try:
                mean_diff = f"{coord_result['mean_diff']:.6e}"
            except Exception:
                mean_diff = str(coord_result.get("mean_diff"))
        
        cache_dims = str(coord_result.get("cache_dims", "-"))
        processed_dims = str(coord_result.get("processed_dims", "-"))
        cache_shape = str(coord_result.get("cache_shape", "-"))
        processed_shape = str(coord_result.get("processed_shape", "-"))
        cache_dtype = coord_result.get("cache_dtype", "-")
        processed_dtype = coord_result.get("processed_dtype", "-")
        cache_size = str(coord_result.get("cache_size", "-"))
        processed_size = str(coord_result.get("processed_size", "-"))
        
        lines.append(f"| `{coord_name}` | {in_cache} | {in_processed} | {dims_ok} | {shape_ok} | {dtype_ok} | {values_ok} | {max_diff} | {mean_diff} | {cache_dims} | {processed_dims} | {cache_shape} | {processed_shape} | {cache_dtype} | {processed_dtype} | {cache_size} | {processed_size} |")
        
        # Show errors if any
        if coord_result.get("errors"):
            for error in coord_result["errors"]:
                lines.append(f"|   └─ ❌ {error} | | | | | | | | | | | | | | | |")
        
        # Show actual values for detailed comparison
        cache_vals = coord_result.get("cache_values")
        processed_vals = coord_result.get("processed_values")
        if cache_vals is not None or processed_vals is not None:
            lines.append("")
            lines.append(f"### `{coord_name}` Coordinate Details")
            lines.append("")
            if cache_vals is not None:
                lines.append(f"- **Cache Values:** `{cache_vals}`")
            if processed_vals is not None:
                lines.append(f"- **Processed Values:** `{processed_vals}`")
            lines.append("")
    
    lines.append("")
    
    return "\n".join(lines)


def main():
    """Main function."""
    # Find matching files
    cache_file = CACHE_DIR / "source-era5_date-2022-01-01_res-1.0_levels-13_steps-01.nc"
    processed_file = PROCESSED_DIR / "source-era5_date-2022-01-01T12_res-1.0_levels-13_steps-01.nc"
    
    if not cache_file.exists():
        print(f"Error: Cache file not found: {cache_file}")
        return 1
    
    if not processed_file.exists():
        print(f"Error: Processed file not found: {processed_file}")
        return 1
    
    print(f"Comparing:")
    print(f"  Cache: {cache_file}")
    print(f"  Processed: {processed_file}")
    print()
    
    # Compare datasets
    results = compare_datasets(cache_file, processed_file)
    
    # Generate markdown
    markdown = generate_markdown(results)
    
    # Write to file
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text(markdown, encoding="utf-8")
    
    print(f"Comparison complete. Results written to: {OUTPUT_MD}")
    
    # Print summary
    if results["structure_match"] and results["values_match"]:
        print("✅ All checks passed! Datasets are identical.")
        return 0
    else:
        print("❌ Differences found. See markdown file for details.")
        return 1


if __name__ == "__main__":
    exit(main())

