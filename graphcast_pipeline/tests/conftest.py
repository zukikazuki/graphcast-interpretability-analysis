import pathlib
import textwrap
import yaml
import pytest


@pytest.fixture
def tmp_cfg_path(tmp_path: pathlib.Path, tmp_raw_dir: pathlib.Path):
    """Write a minimal YAML config (1 日分) を一時ファイルに生成して返す。"""
    yaml_content = textwrap.dedent(
        f"""
        period:
          start: "2022-01-01"
          end:   "2022-01-01"
        resolution: 1.0
        lookback_hours: 0
        storage:
          raw_dir: "{tmp_raw_dir}"
        compute:
          num_workers: 1
        """
    )
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml_content)
    return cfg_path


@pytest.fixture
def tmp_raw_dir(tmp_path: pathlib.Path):
    d = tmp_path / "raw"
    d.mkdir()
    return d 