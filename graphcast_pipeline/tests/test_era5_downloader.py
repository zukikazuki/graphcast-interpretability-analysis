import pathlib
from typing import Any

import pytest

from graphcast_pipeline.io.era5_downloader import Cfg, Era5Downloader


class _DummyClient:  # noqa: D401
    """cdsapi.Client のダミー。retrieve で空ファイルを生成するだけ。"""

    def retrieve(self, _dataset: str, _request: dict[str, Any], target: str):
        pathlib.Path(target).write_text("dummy")


def test_download_one_day(monkeypatch: pytest.MonkeyPatch, tmp_cfg_path: pathlib.Path, tmp_raw_dir: pathlib.Path):
    """期間 1 日・変数最小で 2 ファイルが生成されることを確認。"""

    # cdsapi.Client をモック置換
    import graphcast_pipeline.io.era5_downloader as mod

    monkeypatch.setattr(mod, "cdsapi", type("X", (), {"Client": _DummyClient}))

    cfg = Cfg.from_yaml(tmp_cfg_path)
    downloader = Era5Downloader(cfg)
    downloader.download_range()

    files = sorted(tmp_raw_dir.glob("*.nc"))
    assert len(files) == 2
    assert any("single" in f.name for f in files)
    assert any("pressure" in f.name for f in files) 