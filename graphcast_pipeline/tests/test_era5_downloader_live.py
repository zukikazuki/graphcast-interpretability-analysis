"""Integration test that hits the real CDS API.
This test is skipped by default because it requires valid ~/.cdsapirc credentials and
makes network calls which can take several minutes.
"""
import os
import datetime as _dt
import pathlib
from typing import Any

import pytest  # type: ignore

from graphcast_pipeline.io.era5_downloader import Cfg, Era5Downloader


@pytest.mark.integration  # type: ignore[attr-defined]
@pytest.mark.slow  # mark as slow so it can be deselected easily
def test_live_download_two_days(tmp_path: pathlib.Path):
    """Download 2 days (2022-01-01 to 2022-01-02) and verify file count & dimensions.

    Expected timestamps (lookback 12h, 6-hour grid):
      2021-12-31 12, 18,
      2022-01-01 00, 06, 12, 18,
      2022-01-02 00   → 7 slots
      7 slots × (single + pressure) = 14 files expected.
    """

    # 簡易的な認証ファイルチェック (.cdsapirc が存在しないと Client() が失敗する)
    if not pathlib.Path.home().joinpath(".cdsapirc").exists():
        pytest.skip("~/.cdsapirc not found – skip live API test", allow_module_level=False)

    # ------------------------------------------------------------------
    # YAML を一時生成
    # ------------------------------------------------------------------
    raw_dir = tmp_path / "raw"
    cfg_yaml = f"""
period:
  start: "2022-01-01"
  end:   "2022-01-02"
lookback_hours: 12
resolution: 1.0
root: "{tmp_path.as_posix()}"
storage:
  raw_dir: "{raw_dir.as_posix()}"
  log_dir: "{(tmp_path / 'logs').as_posix()}"
compute:
  num_workers: 2
"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg_yaml)

    # ------------------------------------------------------------------
    # 実ダウンロード実行
    # ------------------------------------------------------------------
    cfg = Cfg.from_yaml(cfg_path)
    Era5Downloader(cfg).download_range()

    # ------------------------------------------------------------------
    # 検証 – ファイル数とファイルサイズ
    # ------------------------------------------------------------------
    files = sorted(raw_dir.glob("*.nc"))

    # Downloader のロジックと同じ計算で期待ファイル数を算出
    period_start = _dt.datetime.fromisoformat("2022-01-01")
    period_end = _dt.datetime.fromisoformat("2022-01-02")
    lookback = 12
    start_ts = period_start - _dt.timedelta(hours=lookback)
    n_hours = int(((period_end - start_ts).total_seconds()) // 3600)
    expected_slots = n_hours // 6 + 1  # inclusive
    expected_files = expected_slots * 2

    assert len(files) == expected_files, f"期待ファイル数 {expected_files} に対し {len(files)} 個生成"

    assert any("single" in f.name for f in files)
    assert any("pressure" in f.name for f in files)

    # 各ファイルが 0 バイトではないことを確認 (CDS が実データを返した証拠)
    for f in files:
        assert f.stat().st_size > 0, f"ファイル {f.name} が空です"

    # ------------------------------------------------------------------
    # ログ出力を表示
    # ------------------------------------------------------------------
    import json, glob
    log_files = sorted(cfg.log_dir.glob("download_log_*.json"))
    if log_files:
        latest_log = log_files[-1]
        print("\nDownload summary JSON:")
        print(latest_log.read_text()) 