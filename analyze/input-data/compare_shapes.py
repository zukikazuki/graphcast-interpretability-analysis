import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set


def vlog(verbose: bool, message: str) -> None:
    if verbose:
        print(message)


@dataclass
class VariableShape:
    dims: Tuple[str, ...]
    shape: Tuple[int, ...]
    source: str  # e.g., "raw-single", "raw-pressure", "cache"


@dataclass
class PathsConfig:
    root: Path
    raw_dir: Path
    cache_dir: Optional[Path]
    config_path: Path

# -----------------------------------------------------------------------------
# 固定パス（ユーザー指定の確定値）
# -----------------------------------------------------------------------------
CONFIG_PATH = Path(
    "/home/dl-box/Fujikawa/research/machine_learning/graphcast/graphcast_pipeline/configs/config.yaml"
)
ROOT_PATH = Path("/media/dl-box/SSD-SCTU3A/graphcast_data")
RAW_DIR = Path("/media/dl-box/SSD-SCTU3A/graphcast_data/data/raw/era5")
CACHE_DIR = Path(
    "/home/dl-box/Fujikawa/research/machine_learning/graphcast/graphcast_cache"
)
PROCESSED_DIR = Path(
    "/media/dl-box/SSD-SCTU3A/graphcast_data/data/processed/graphcast_inputs"
)


def try_import_yaml():
    return None  # YAML 読みは不要になったため無効化


def load_config(config_path: Path) -> Dict:
    # ハードコード指定のため YAML を実際には読みません
    return {}


def resolve_paths_from_config(verbose: bool = False) -> PathsConfig:
    root = ROOT_PATH.resolve()
    raw_dir = RAW_DIR.resolve()
    cache_dir = CACHE_DIR.resolve()

    vlog(verbose, f"[config] config_path={CONFIG_PATH}")
    vlog(verbose, f"[config] root={root}")
    vlog(verbose, f"[config] raw_dir={raw_dir}")
    vlog(verbose, f"[config] cache_dir={cache_dir}")

    return PathsConfig(root=root, raw_dir=raw_dir, cache_dir=cache_dir, config_path=CONFIG_PATH)


def auto_find_cache_dir(*args, **kwargs) -> Optional[Path]:
    # 使わなくなったが互換のため残す
    return CACHE_DIR


def is_coord_or_aux(name: str) -> bool:
    lowered = name.lower()
    return lowered in {"time", "times", "valid_time", "latitude", "lat", "longitude", "lon", "level", "levels", "step"}


def open_xarray_dataset(path: Path):
    import xarray as xr  # type: ignore  # local import to avoid hard dependency if unused
    if path.suffix == ".zarr" or path.is_dir() and path.name.endswith(".zarr"):
        return xr.open_zarr(str(path))
    return xr.open_dataset(str(path), decode_cf=False)


def extract_shapes_from_nc_or_zarr(path: Path, source_tag: str) -> Dict[str, VariableShape]:
    shapes: Dict[str, VariableShape] = {}
    try:
        ds = open_xarray_dataset(path)
    except Exception:
        return shapes
    try:
        for var_name, da in ds.data_vars.items():
            if is_coord_or_aux(var_name):
                continue
            dims = tuple(map(str, da.dims))
            shape = tuple(int(s) for s in da.shape)
            shapes[var_name] = VariableShape(dims=dims, shape=shape, source=source_tag)
    finally:
        try:
            ds.close()
        except Exception:
            pass
    return shapes


def extract_raw_shapes(raw_dir: Path, max_files_per_group: int = 3) -> Dict[str, VariableShape]:
    shapes: Dict[str, VariableShape] = {}

    groups = []
    # Try to use known sub-structure if present
    single_dir = raw_dir / "single"
    pressure_dir = raw_dir / "pressure"
    if single_dir.exists():
        groups.append((single_dir, "raw-single"))
    if pressure_dir.exists():
        groups.append((pressure_dir, "raw-pressure"))
    if not groups:
        groups.append((raw_dir, "raw"))

    for base, tag in groups:
        count = 0
        # Search for .nc and .zarr
        for p in base.rglob("*.nc"):
            shapes.update(extract_shapes_from_nc_or_zarr(p, tag))
            count += 1
            if count >= max_files_per_group:
                break
        if count < max_files_per_group:
            for p in base.rglob("*.zarr"):
                if p.is_dir():
                    shapes.update(extract_shapes_from_nc_or_zarr(p, tag))
                    count += 1
                    if count >= max_files_per_group:
                        break

    return shapes


def extract_cache_shapes(cache_dir: Path, max_stores: int = 6, verbose: bool = False) -> Dict[str, VariableShape]:
    shapes: Dict[str, VariableShape] = {}
    if not cache_dir.exists():
        vlog(verbose, f"[cache-scan] cache_dir not found: {cache_dir}")
        return shapes

    count = 0
    seen: Set[Path] = set()

    def _consume(path: Path) -> None:
        nonlocal count
        if count >= max_stores:
            return
        rp = path.resolve()
        if rp in seen:
            return
        seen.add(rp)
        vlog(verbose, f"[cache-scan] consume: {path}")
        shapes.update(extract_shapes_from_nc_or_zarr(path, "cache"))
        count += 1

    # 1) Top-level files/stores like source-era5*.nc or source-era5*.zarr
    try:
        vlog(verbose, f"[cache-scan] scan top-level: {cache_dir}")
        for p in sorted(cache_dir.iterdir()):
            if count >= max_stores:
                break
            name = p.name
            if not name.startswith("source-era5"):
                continue
            if p.is_file() and p.suffix == ".nc":
                vlog(verbose, f"[cache-scan] top-level file match: {p}")
                _consume(p)
            elif p.is_dir() and name.endswith(".zarr"):
                vlog(verbose, f"[cache-scan] top-level zarr match: {p}")
                _consume(p)
    except Exception:
        pass

    if count >= max_stores:
        return shapes

    # 2) Subdirectories named source-era5-*
    try:
        vlog(verbose, f"[cache-scan] scan subdirs: {cache_dir}")
        for sub in sorted(cache_dir.iterdir()):
            if count >= max_stores:
                break
            if not sub.is_dir() or not sub.name.startswith("source-era5-"):
                continue
            vlog(verbose, f"[cache-scan] subdir: {sub}")
            zarrs = list(sub.rglob("*.zarr"))
            ncs = list(sub.rglob("*.nc"))
            targets = [*zarrs, *ncs]
            if not targets and sub.name.endswith(".zarr"):
                targets = [sub]
            for p in targets:
                if count >= max_stores:
                    break
                vlog(verbose, f"[cache-scan] subdir target: {p}")
                _consume(p)
    except Exception:
        pass

    if count >= max_stores:
        return shapes

    # 3) Fallback: recursive search anywhere under cache_dir (e.g., in 'unzipped')
    try:
        vlog(verbose, f"[cache-scan] fallback recursive search under: {cache_dir}")
        for p in cache_dir.rglob("source-era5*.nc"):
            if count >= max_stores:
                break
            if p.is_file():
                vlog(verbose, f"[cache-scan] fallback .nc: {p}")
                _consume(p)
        for p in cache_dir.rglob("source-era5*.zarr"):
            if count >= max_stores:
                break
            if p.is_dir():
                vlog(verbose, f"[cache-scan] fallback .zarr: {p}")
                _consume(p)
    except Exception:
        pass

    vlog(verbose, f"[cache-scan] total consumed: {count}")
    return shapes


def normalize_lat_lon_dim_names(dims: Tuple[str, ...]) -> Tuple[str, ...]:
    mapped = []
    for d in dims:
        dl = d.lower()
        if dl in {"latitude", "lat"}:
            mapped.append("lat")
        elif dl in {"longitude", "lon"}:
            mapped.append("lon")
        elif dl in {"levels", "level"}:
            mapped.append("level")
        else:
            mapped.append(d)
    return tuple(mapped)


def compare_var_shapes(a: VariableShape, b: VariableShape) -> Dict:
    dims_a = normalize_lat_lon_dim_names(a.dims)
    dims_b = normalize_lat_lon_dim_names(b.dims)

    same_dims = dims_a == dims_b
    same_shape = a.shape == b.shape

    # Also check spatial grid sizes regardless of time/level positions
    def lat_lon_size(dims: Tuple[str, ...], shape: Tuple[int, ...]) -> Tuple[Optional[int], Optional[int]]:
        lat_size = None
        lon_size = None
        for d, s in zip(dims, shape):
            dl = d.lower()
            if dl.endswith("lat"):
                lat_size = s
            if dl.endswith("lon"):
                lon_size = s
        return lat_size, lon_size

    lat_a, lon_a = lat_lon_size(dims_a, a.shape)
    lat_b, lon_b = lat_lon_size(dims_b, b.shape)
    same_spatial = (lat_a == lat_b) and (lon_a == lon_b)

    return {
        "same_dims": bool(same_dims),
        "same_shape": bool(same_shape),
        "same_spatial": bool(same_spatial),
        "dims_a": a.dims,
        "dims_b": b.dims,
        "shape_a": a.shape,
        "shape_b": b.shape,
    }


def build_summary(
    paths: PathsConfig,
    raw_shapes: Dict[str, VariableShape],
    cache_shapes: Dict[str, VariableShape],
) -> Tuple[str, Dict]:
    all_vars = sorted(set(raw_shapes.keys()) | set(cache_shapes.keys()))
    lines: List[str] = []
    lines.append("## GraphCast 入力データ 形状比較")
    lines.append("")
    lines.append(f"- **Config**: `{paths.config_path}`")
    lines.append(f"- **root**: `{paths.root}`")
    lines.append(f"- **raw_dir**: `{paths.raw_dir}`")
    lines.append(f"- **cache_dir**: `{paths.cache_dir if paths.cache_dir else 'NOT FOUND'}`")
    lines.append("")

    summary_json: Dict = {
        "config_path": str(paths.config_path),
        "root": str(paths.root),
        "raw_dir": str(paths.raw_dir),
        "cache_dir": str(paths.cache_dir) if paths.cache_dir else None,
        "variables": {},
    }

    # Markdown table
    lines.append("| variable | raw (dims → shape) | cache (dims → shape) | match (shape/dims/spatial) |")
    lines.append("|---|---|---|---|")

    for var in all_vars:
        a = raw_shapes.get(var)
        b = cache_shapes.get(var)
        if a and b:
            comp = compare_var_shapes(a, b)
            match_str = f"{comp['same_shape']}/{comp['same_dims']}/{comp['same_spatial']}"
            lines.append(
                f"| `{var}` | `{a.dims}` → `{a.shape}` | `{b.dims}` → `{b.shape}` | {match_str} |"
            )
            summary_json["variables"][var] = {
                "raw": {"dims": a.dims, "shape": a.shape, "source": a.source},
                "cache": {"dims": b.dims, "shape": b.shape, "source": b.source},
                "comparison": comp,
            }
        elif a and not b:
            lines.append(f"| `{var}` | `{a.dims}` → `{a.shape}` | MISSING | false/false/false |")
            summary_json["variables"][var] = {
                "raw": {"dims": a.dims, "shape": a.shape, "source": a.source},
                "cache": None,
            }
        elif b and not a:
            lines.append(f"| `{var}` | MISSING | `{b.dims}` → `{b.shape}` | false/false/false |")
            summary_json["variables"][var] = {
                "raw": None,
                "cache": {"dims": b.dims, "shape": b.shape, "source": b.source},
            }

    text = "\n".join(lines) + "\n"
    return text, summary_json


def write_outputs(output_dir: Path, text: str, summary_json: Dict, base_name: str = "shapes_summary") -> Tuple[Path, Path]:
    output_md = output_dir / f"{base_name}.md"
    output_json = output_dir / f"{base_name}.json"
    output_md.write_text(text, encoding="utf-8")
    output_json.write_text(json.dumps(summary_json, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_md, output_json


def infer_repo_root_from_script() -> Path:
    # analyze/input-data/compare_shapes.py -> repo root is two parents up
    return Path(__file__).resolve().parents[2]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ERA5 raw shapes vs GraphCast cache shapes.")
    parser.add_argument("--output-name", default="shapes_summary", help="Base name of output files (no extension)")
    parser.add_argument("--max-raw-files", type=int, default=3, help="Max raw files to sample per group")
    parser.add_argument("--max-cache-stores", type=int, default=6, help="Max cache stores to sample")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args(argv)


def render_dir_tree(root: Path, max_depth: int = 3, max_entries: int = 2000) -> str:
    lines: List[str] = []
    entries_count = 0

    def _walk(path: Path, depth: int, prefix: str) -> None:
        nonlocal entries_count
        if entries_count >= max_entries:
            return
        try:
            items = sorted([p for p in path.iterdir()], key=lambda p: (not p.is_dir(), p.name.lower()))
        except Exception:
            return
        for idx, p in enumerate(items):
            if entries_count >= max_entries:
                return
            connector = "└──" if idx == len(items) - 1 else "├──"
            name = p.name + ("/" if p.is_dir() else "")
            lines.append(f"{prefix}{connector} {name}")
            entries_count += 1
            if p.is_dir() and depth < max_depth:
                extension = "    " if idx == len(items) - 1 else "│   "
                _walk(p, depth + 1, prefix + extension)

    lines.append(f"{root}:")
    _walk(root, 1, "")
    if entries_count >= max_entries:
        lines.append(f"... (truncated after {max_entries} entries)")
    return "\n".join(lines)


def build_structure_text(paths: PathsConfig, structure_depth: int = 3) -> str:
    sections: List[str] = []
    sections.append("## ディレクトリ構造 (概要)")
    sections.append("")
    sections.append(f"### raw_dir (depth={structure_depth})")
    sections.append("")
    sections.append("```\n" + render_dir_tree(paths.raw_dir, max_depth=structure_depth) + "\n```")
    sections.append("")
    if paths.cache_dir and paths.cache_dir.exists():
        sections.append(f"### cache_dir (depth={structure_depth})")
        sections.append("")
        sections.append("```\n" + render_dir_tree(paths.cache_dir, max_depth=structure_depth) + "\n```")
    else:
        sections.append("### cache_dir")
        sections.append("")
        sections.append("NOT FOUND")
    sections.append("")
    return "\n".join(sections)


def build_single_structure_text(title: str, target_dir: Path, structure_depth: int = 3) -> str:
    sections: List[str] = []
    sections.append(f"## {title}")
    sections.append("")
    if target_dir.exists():
        sections.append("```")
        sections.append(render_dir_tree(target_dir, max_depth=structure_depth))
        sections.append("```")
    else:
        sections.append("NOT FOUND")
    sections.append("")
    return "\n".join(sections)


def write_simple_structure_files(paths: PathsConfig, structure_depth: int = 3) -> List[Path]:
    output_dir = Path(__file__).resolve().parent
    outputs: List[Path] = []

    mapping = [
        ("pressure_structure.md", "pressure", paths.raw_dir / "pressure"),
        ("single_accum_structure.md", "single/accum", paths.raw_dir / "single" / "accum"),
        ("single_instant_structure.md", "single/instant", paths.raw_dir / "single" / "instant"),
        ("cache_structure.md", "cache", paths.cache_dir if paths.cache_dir else paths.root / "graphcast_cache"),
    ]

    for filename, title, dir_path in mapping:
        text = build_single_structure_text(title, dir_path, structure_depth=structure_depth)
        p = output_dir / filename
        p.write_text(text, encoding="utf-8")
        outputs.append(p)

    return outputs


def find_sample_file_in_dir(target_dir: Path, prefer_pattern: Optional[str] = None) -> Optional[Path]:
    if not target_dir.exists():
        return None
    try:
        if prefer_pattern:
            for p in sorted(target_dir.glob(prefer_pattern)):
                if p.is_file() or p.is_dir():
                    return p
        # Prefer top-level .nc, then .zarr
        for p in sorted(target_dir.glob("*.nc")):
            if p.is_file():
                return p
        for p in sorted(target_dir.glob("*.zarr")):
            if p.is_dir():
                return p
        # Fallback recursive
        for p in target_dir.rglob("*.nc"):
            if p.is_file():
                return p
        for p in target_dir.rglob("*.zarr"):
            if p.is_dir():
                return p
    except Exception:
        return None
    return None


def build_data_structure_md(title: str, sample_path: Optional[Path]) -> str:
    lines: List[str] = []
    lines.append(f"## {title}")
    lines.append("")
    if not sample_path:
        lines.append("サンプルファイルが見つかりませんでした")
        lines.append("")
        return "\n".join(lines)

    shapes = extract_shapes_from_nc_or_zarr(sample_path, source_tag=title)
    lines.append(f"- sample: `{sample_path}`")
    lines.append("")
    lines.append("| variable | dims | shape |")
    lines.append("|---|---|---|")
    for var in sorted(shapes.keys()):
        s = shapes[var]
        lines.append(f"| `{var}` | `{s.dims}` | `{s.shape}` |")
    lines.append("")
    return "\n".join(lines)


def write_data_structure_files(paths: PathsConfig) -> List[Path]:
    output_dir = Path(__file__).resolve().parent
    outputs: List[Path] = []

    pressure_dir = paths.raw_dir / "pressure"
    single_accum_dir = paths.raw_dir / "single" / "accum"
    single_instant_dir = paths.raw_dir / "single" / "instant"
    cache_dir = paths.cache_dir if paths.cache_dir else CACHE_DIR

    # Find representative files
    pressure_sample = find_sample_file_in_dir(pressure_dir)
    single_accum_sample = find_sample_file_in_dir(single_accum_dir)
    single_instant_sample = find_sample_file_in_dir(single_instant_dir)
    cache_sample = None
    if cache_dir and cache_dir.exists():
        # Prefer top-level source-era5*.nc first
        cache_sample = find_sample_file_in_dir(cache_dir, prefer_pattern="source-era5*.nc")
        if cache_sample is None:
            cache_sample = find_sample_file_in_dir(cache_dir)

    # Build combined Markdown content
    parts: List[str] = []
    parts.append("# データ構造 (pressure / single/accum / single/instant / cache)")
    parts.append("")
    parts.append(build_data_structure_md("pressure", pressure_sample))
    parts.append(build_data_structure_md("single/accum", single_accum_sample))
    parts.append(build_data_structure_md("single/instant", single_instant_sample))
    parts.append(build_data_structure_md("cache", cache_sample))

    combined_md = "\n".join(parts)
    combined_path = output_dir / "data_structures.md"
    combined_path.write_text(combined_md, encoding="utf-8")
    outputs.append(combined_path)

    return outputs


def attrs_keys_summary(attrs: Dict) -> str:
    try:
        keys = list(attrs.keys())
        return ", ".join(str(k) for k in keys)
    except Exception:
        return ""


def build_raw_schema_md(title: str, sample_path: Optional[Path]) -> str:
    lines: List[str] = []
    lines.append(f"## {title}")
    lines.append("")
    if not sample_path:
        lines.append("サンプルファイルが見つかりませんでした")
        lines.append("")
        return "\n".join(lines)

    lines.append(f"- sample: `{sample_path}`")
    try:
        ds = open_xarray_dataset(sample_path)
    except Exception as e:
        lines.append(f"- open error: `{e}`")
        lines.append("")
        return "\n".join(lines)

    try:
        # Dims
        lines.append("### dims")
        lines.append("")
        lines.append("| name | size |")
        lines.append("|---|---|")
        for name, size in ds.dims.items():
            lines.append(f"| `{name}` | `{int(size)}` |")
        lines.append("")

        # Coords
        lines.append("### coords")
        lines.append("")
        lines.append("| name | dims | shape | dtype |")
        lines.append("|---|---|---|---|")
        for name, da in ds.coords.items():
            dims = tuple(map(str, da.dims))
            shape = tuple(int(s) for s in da.shape)
            dtype = str(getattr(da, "dtype", ""))
            lines.append(f"| `{name}` | `{dims}` | `{shape}` | `{dtype}` |")
        lines.append("")

        # Data variables
        lines.append("### data_vars")
        lines.append("")
        lines.append("| name | dims | shape | dtype |")
        lines.append("|---|---|---|---|")
        for name, da in ds.data_vars.items():
            dims = tuple(map(str, da.dims))
            shape = tuple(int(s) for s in da.shape)
            dtype = str(getattr(da, "dtype", ""))
            lines.append(f"| `{name}` | `{dims}` | `{shape}` | `{dtype}` |")
        lines.append("")

        # Global attrs
        lines.append("### global_attrs")
        lines.append("")
        if hasattr(ds, "attrs") and ds.attrs:
            for k, v in ds.attrs.items():
                val = str(v)
                if len(val) > 300:
                    val = val[:300] + "..."
                lines.append(f"- `{k}`: `{val}`")
        else:
            lines.append("(none)")
        lines.append("")
    finally:
        try:
            ds.close()
        except Exception:
            pass

    return "\n".join(lines)


def write_raw_schema_combined(paths: PathsConfig) -> Path:
    output_dir = Path(__file__).resolve().parent

    pressure_dir = paths.raw_dir / "pressure"
    single_accum_dir = paths.raw_dir / "single" / "accum"
    single_instant_dir = paths.raw_dir / "single" / "instant"
    cache_dir = paths.cache_dir if paths.cache_dir else CACHE_DIR
    processed_dir = PROCESSED_DIR

    pressure_sample = find_sample_file_in_dir(pressure_dir)
    single_accum_sample = find_sample_file_in_dir(single_accum_dir)
    single_instant_sample = find_sample_file_in_dir(single_instant_dir)
    cache_sample = None
    if cache_dir and cache_dir.exists():
        cache_sample = find_sample_file_in_dir(cache_dir, prefer_pattern="source-era5*.nc") or find_sample_file_in_dir(cache_dir)
    processed_sample = None
    if processed_dir.exists():
        processed_sample = find_sample_file_in_dir(processed_dir, prefer_pattern="source-era5*.nc") or find_sample_file_in_dir(processed_dir)

    parts: List[str] = []
    parts.append("# 素のデータ構造 (raw NetCDF/Zarr schema)")
    parts.append("")
    parts.append(build_raw_schema_md("pressure", pressure_sample))
    parts.append(build_raw_schema_md("single/accum", single_accum_sample))
    parts.append(build_raw_schema_md("single/instant", single_instant_sample))
    parts.append(build_raw_schema_md("cache", cache_sample))
    parts.append(build_raw_schema_md("processed", processed_sample))

    combined_md = "\n".join(parts)
    out_path = output_dir / "data_structures.md"
    out_path.write_text(combined_md, encoding="utf-8")
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    paths = resolve_paths_from_config(verbose=args.verbose)

    # Write RAW NetCDF/Zarr schema (dims/coords/data_vars/attrs) into single MD
    out_path = write_raw_schema_combined(paths)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)


