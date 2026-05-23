"""Utility for executing a single GraphCast forecast against prepared inputs."""

from __future__ import annotations

import dataclasses
import datetime as dt
import functools
import csv
import json
import time
from pathlib import Path
from typing import Any

import haiku as hk
from tqdm import tqdm
import jax
import jax.numpy as jnp
import numpy as np
import xarray
import yaml

from graphcast import autoregressive
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
from graphcast import xarray_tree


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_path(template: str, root: Path) -> Path:
    """Resolves a config path that may contain a {root} placeholder."""
    return Path(template.format(root=str(root))).expanduser()


def _parse_period_to_times(cfg: dict[str, Any]) -> list[dt.datetime]:
    """期間設定を解析して6時間間隔の時刻リストを生成する。"""
    period_cfg = cfg.get("period", {})
    if not period_cfg:
        raise ValueError("period section (start, end) is required")
    start_str = period_cfg.get("start")
    end_str = period_cfg.get("end")
    if not start_str or not end_str:
        raise ValueError("period.start and period.end are required")
    
    start_dt = dt.datetime.fromisoformat(start_str.replace("Z", "+00:00"))
    end_dt = dt.datetime.fromisoformat(end_str.replace("Z", "+00:00"))
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=dt.timezone.utc)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=dt.timezone.utc)
    start_dt = start_dt.astimezone(dt.timezone.utc)
    end_dt = end_dt.astimezone(dt.timezone.utc)
    
    # 6時間間隔で時刻リストを生成
    times = []
    current = start_dt
    while current <= end_dt:
        times.append(current)
        current += dt.timedelta(hours=6)
    
    return times


class GraphCastSingleRun:
    """Encapsulates a one-off GraphCast inference for a given timestamp."""

    def __init__(self):
        # デフォルトの設定ファイルパスを固定
        self.config_path = Path("graphcast_pipeline/configs/config.yaml")
        self.cfg = _load_yaml(self.config_path)
        self.root = Path(self.cfg.get("root")).expanduser()

        # 最終的な時刻リストのみを保持
        self.target_times = _parse_period_to_times(self.cfg)

        storage_cfg = self.cfg.get("storage", {})

        self.processed_dir = _resolve_path(
            storage_cfg.get(
                "processed_dir",
                "{root}/data/processed/graphcast_inputs",
            ),
            self.root,
        )
        self.forecast_dir = _resolve_path(
            storage_cfg.get("forecast_dir", "{root}/outputs/forecasts"),
            self.root,
        )
        self.vjp_output_dir = _resolve_path(
            storage_cfg.get("vjp_output_dir", "{root}/outputs/vjp"),
            self.root,
        )
        # ログ出力用のディレクトリ（例: {root}/logs/inference）
        self.log_dir = _resolve_path(
            storage_cfg.get("log_dir", "{root}/logs"),
            self.root,
        )
        self.inference_log_dir = self.log_dir / "inference"

        self.forecast_dir.mkdir(parents=True, exist_ok=True)
        self.vjp_output_dir.mkdir(parents=True, exist_ok=True)
        self.inference_log_dir.mkdir(parents=True, exist_ok=True)

        # この SingleRun 実行全体で共有するログファイルパスを決定
        self.inference_log_path = self._build_inference_log_path()

        model_cfg = self.cfg.get("model", {})
        if not model_cfg:
            raise ValueError("model section (checkpoint_path, stats_dir) is required")

        self.checkpoint_path = Path(model_cfg["checkpoint_path"]).expanduser()
        self.stats_dir = Path(model_cfg["stats_dir"]).expanduser()
        self.mode = model_cfg.get("mode", "standard")

        self.resolution = float(self.cfg.get("resolution", 1.0))
        self.levels = int(self.cfg.get("levels", 13))

        self._load_checkpoint()
        self._load_normalization_stats()
        self._rng_seed = 0  # 固定値として保持
        if self.mode == "vjp":
            self._run_forward = self._build_vjp_forward_transform()
        else:
            self._run_forward = self._build_forward_transform()
        # 推論ログを後でまとめて1ファイルに書き出すためのバッファ
        self._inference_records: list[dict[str, Any]] = []

    def _build_inference_log_path(self) -> Path:
        """period と現在時刻から一意なログファイルパスを生成する。"""
        period_cfg = self.cfg.get("period", {})
        start_raw = str(period_cfg.get("start", "unknown"))
        end_raw = str(period_cfg.get("end", "unknown"))
        start_str = start_raw.replace(":", "").replace("Z", "")
        end_str = end_raw.replace(":", "").replace("Z", "")
        timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = f"inference_{start_str}_{end_str}_{timestamp}.json"
        return self.inference_log_dir / filename

    def _load_checkpoint(self) -> None:
        with self.checkpoint_path.open("rb") as f:
            ckpt = checkpoint.load(f, graphcast.CheckPoint)
        self.params = ckpt.params
        self.model_config = ckpt.model_config
        self.task_config = ckpt.task_config

    def _load_normalization_stats(self) -> None:
        stats_names = {
            "diffs": "diffs_stddev_by_level.nc",
            "mean": "mean_by_level.nc",
            "stddev": "stddev_by_level.nc",
        }
        self.diffs_stddev_by_level = xarray.load_dataset(
            self.stats_dir / stats_names["diffs"]
        ).compute()
        self.mean_by_level = xarray.load_dataset(
            self.stats_dir / stats_names["mean"]
        ).compute()
        self.stddev_by_level = xarray.load_dataset(
            self.stats_dir / stats_names["stddev"]
        ).compute()

    def _build_forward_transform(self):
        diffs = self.diffs_stddev_by_level
        mean = self.mean_by_level
        stddev = self.stddev_by_level
        @hk.transform_with_state
        def run_forward(
            model_config,
            task_config,
            inputs,
            targets_template,
            forcings,
        ):
            predictor = graphcast.GraphCast(model_config, task_config)
            predictor = normalization.InputsAndResiduals(
                predictor,
                diffs_stddev_by_level=diffs,
                mean_by_level=mean,
                stddev_by_level=stddev,
            )
            predictor = autoregressive.Predictor(
                predictor,
                gradient_checkpointing=True,
            )
            return predictor(
                inputs,
                targets_template=targets_template,
                forcings=forcings,
            )

        return run_forward

    def _build_vjp_forward_transform(self):
        diffs = self.diffs_stddev_by_level
        mean = self.mean_by_level
        stddev = self.stddev_by_level

        @hk.transform_with_state
        def run_forward_vjp(
            model_config,
            task_config,
            inputs,
            targets_template,
            forcings,
        ):
            predictor = graphcast.GraphCast(model_config, task_config)
            norm_inputs = normalization.normalize(inputs, stddev, mean)
            norm_forcings = normalization.normalize(forcings, stddev, mean)

            predictor._maybe_init(norm_inputs)

            grid_node_features = predictor._inputs_to_grid_node_features(
                norm_inputs, norm_forcings
            )
            latent_mesh, latent_grid = predictor._run_grid2mesh_gnn(
                grid_node_features
            )

            def decoder_output(mesh_latent):
                return predictor._run_mesh2grid_gnn(mesh_latent, latent_grid)

            decoder_output_ckpt = hk.remat(decoder_output)

            base_key = hk.next_rng_key()

            def vjp_score(mesh_latent, layer_idx):
                output_grid_nodes, vjp_fun = jax.vjp(
                    decoder_output_ckpt, mesh_latent
                )
                layer_key = jax.random.fold_in(base_key, layer_idx)
                def body(k, acc):
                    key = jax.random.fold_in(layer_key, k)
                    v = jax.random.rademacher(
                        key, output_grid_nodes.shape, dtype=output_grid_nodes.dtype
                    )
                    (g,) = vjp_fun(v)
                    return acc + jnp.sum(jnp.square(g))

                total = jax.lax.fori_loop(0, 8, body, jnp.array(0.0, dtype=output_grid_nodes.dtype))
                return total / 8.0

            scores = [vjp_score(latent_mesh, 0)]

            mesh_graph = predictor._mesh_graph_structure
            assert mesh_graph is not None
            mesh_edges_key = mesh_graph.edge_key_by_name("mesh")
            edges = mesh_graph.edges[mesh_edges_key]
            batch_size = latent_mesh.shape[1]

            new_edges = edges._replace(
                features=graphcast._add_batch_second_axis(
                    edges.features.astype(latent_mesh.dtype), batch_size
                )
            )
            nodes = mesh_graph.nodes["mesh_nodes"]._replace(features=latent_mesh)
            input_graph = mesh_graph._replace(
                edges={mesh_edges_key: new_edges}, nodes={"mesh_nodes": nodes}
            )

            embedder_network, processor_networks, _ = (
                predictor._mesh_gnn._networks_builder(input_graph, None)
            )
            latent_graph = predictor._mesh_gnn._embed(input_graph, embedder_network)

            layer_idx = 1
            mesh_latent = latent_mesh
            for _ in range(predictor._mesh_gnn._num_processor_repetitions):
                for processor_network in processor_networks:
                    latent_graph = predictor._mesh_gnn._process_step(
                        processor_network, latent_graph
                    )
                    mesh_latent = latent_graph.nodes["mesh_nodes"].features
                    scores.append(vjp_score(mesh_latent, layer_idx))
                    layer_idx += 1

            output_grid_nodes = decoder_output(mesh_latent)
            norm_predictions = predictor._grid_node_outputs_to_prediction(
                output_grid_nodes, targets_template
            )

            def unnormalize_prediction(norm_prediction):
                if norm_prediction.name in inputs:
                    prediction = normalization.unnormalize(
                        norm_prediction, diffs, None
                    )
                    last_input = inputs[norm_prediction.name].isel(time=-1)
                    return prediction + last_input
                return normalization.unnormalize(norm_prediction, stddev, mean)

            predictions = xarray_tree.map_structure(
                unnormalize_prediction, norm_predictions
            )

            return predictions, jnp.stack(scores)

        return run_forward_vjp

    def _rng(self, tag: str) -> jax.Array:
        base = jax.random.PRNGKey(self._rng_seed)
        # fold_in expects data that can be converted to uint32.
        # Hash the tag deterministically and mask to uint32 range.
        tag_hash = np.uint32(hash(tag) & 0xFFFFFFFF)
        return jax.random.fold_in(base, tag_hash)

    def _with_configs(self, fn, model_config):
        return functools.partial(
            fn,
            model_config=model_config,
            task_config=self.task_config,
        )

    @staticmethod
    def _with_params(fn, params, state):
        return functools.partial(fn, params=params, state=state)

    def _dataset_path(self, target_time: dt.datetime):
        if target_time.tzinfo is None:
            anchor = target_time.replace(tzinfo=dt.timezone.utc)
        else:
            anchor = target_time
        anchor = anchor.astimezone(dt.timezone.utc)
        anchor_str = anchor.strftime("%Y-%m-%dT%H")
        filename = (
            f"source-era5_date-{anchor_str}_res-{self.resolution}_"
            f"levels-{self.levels}_steps-01.nc"
        )
        return self.processed_dir / filename

    def _log_single_inference(
        self,
        *,
        target_time: dt.datetime,
        elapsed_sec: float,
    ) -> None:
        """単一推論の結果をバッファに保存（ファイル書き込みはまとめて行う）。"""
        self._inference_records.append(
            {
                "forecast_time": f"{target_time.isoformat()}Z",
                "elapsed_sec": round(elapsed_sec, 3),
            }
        )

    def _log_total_elapsed(
        self,
        *,
        started_at: dt.datetime,
        ended_at: dt.datetime,
        elapsed_sec: float,
    ) -> None:
        """全体の実行時間と推論結果をまとめて JSON に保存する。"""
        summary = {
            "started_at": f"{started_at.isoformat()}Z",
            "ended_at": f"{ended_at.isoformat()}Z",
            "total_elapsed_sec": round(elapsed_sec, 3),
        }
        payload = {
            "inferences": self._inference_records,
            "summary": summary,
        }
        with self.inference_log_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _build_vjp_csv_path(self, started_at: dt.datetime) -> Path:
        period_cfg = self.cfg.get("period", {})
        start_raw = str(period_cfg.get("start", "unknown"))
        end_raw = str(period_cfg.get("end", "unknown"))
        start_str = start_raw.replace(":", "").replace("Z", "")
        end_str = end_raw.replace(":", "").replace("Z", "")
        steps_str = f"{self.model_config.gnn_msg_steps:02d}"
        run_str = started_at.strftime("%Y%m%dT%H%M%SZ")
        filename = f"vjp_{start_str}_{end_str}_gnn{steps_str}_{run_str}.csv"
        return self.vjp_output_dir / filename

    def _build_jitted_fns(self):
        """init/apply の JIT 関数を構築する。"""
        model_config = self.model_config
        print("‼️ JIT compiling for", model_config)

        # init: state を生成する関数（model_config, task_config を束縛）
        init_core = self._with_configs(self._run_forward.init, model_config)
        init_fn = jax.jit(init_core)

        # apply: params / state は後から束縛できるよう、ここでは model_config / task_config のみ束縛して JIT
        apply_core = self._with_configs(self._run_forward.apply, model_config)
        apply_fn = jax.jit(apply_core)

        return init_fn, apply_fn

    def _run_single(
        self,
        target_time: dt.datetime,
        init_fn,
        apply_fn,
    ) -> tuple[xarray.Dataset, Path, np.ndarray | None]:
        """Executes inference for a single timestamp and returns predictions along with artifact paths."""
        ds_path = self._dataset_path(target_time)
        if not ds_path.exists():
            raise FileNotFoundError(ds_path)

        t0 = time.perf_counter()

        dataset = xarray.load_dataset(ds_path).compute()
        if self.mode == "vjp":
            target_slice = "6h"
        else:
            target_steps = max(dataset.sizes["time"] - 2, 1)
            target_slice = slice("6h", f"{target_steps * 6}h")
        inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
            dataset,
            target_lead_times=target_slice,
            **dataclasses.asdict(self.task_config),
        )
        # 既に JIT 済みの init_fn を用いて state を初期化
        _, state = init_fn(
            rng=self._rng("init"),
            inputs=inputs,
            targets_template=targets,
            forcings=forcings,
        )

        # state は毎回変わるので、params とあわせて部分適用で apply_fn に渡す
        # apply_fn は (params, state, rng, inputs, targets_template, forcings, ...) を受け取る想定
        apply_fn_with_state = functools.partial(apply_fn, self.params, state)

        vjp_scores = None
        if self.mode == "vjp":
            (predictions, vjp_scores), final_state = apply_fn_with_state(
                rng=self._rng("vjp"),
                inputs=inputs,
                targets_template=targets,
                forcings=forcings,
            )
        else:
            predictions, final_state = rollout.chunked_prediction(
                apply_fn_with_state,
                rng=self._rng("rollout"),
                inputs=inputs,
                targets_template=targets * np.nan,
                forcings=forcings,
                num_steps_per_chunk=1,
                verbose=False,
            )

        stem = ds_path.stem

        forecast_path = self.forecast_dir / f"{stem}_predictions.nc"
        predictions.to_netcdf(forecast_path)


        elapsed_sec = time.perf_counter() - t0
        self._log_single_inference(
            target_time=target_time,
            elapsed_sec=elapsed_sec,
        )

        return predictions, forecast_path, vjp_scores
    
    def run(self) -> None:
        """Executes inference for all target times from config internally."""
        started_at = dt.datetime.now(dt.timezone.utc)
        t0 = time.perf_counter()

        total_steps = len(self.target_times)
        progress = tqdm(total=total_steps, desc="Inference", unit="step")

        init_fn, apply_fn = self._build_jitted_fns()

        vjp_writer = None
        vjp_file = None
        if self.mode == "vjp":
            vjp_path = self._build_vjp_csv_path(started_at)
            vjp_file = vjp_path.open("w", encoding="utf-8", newline="")
            vjp_writer = csv.writer(vjp_file)
            vjp_writer.writerow(
                ["time"] + [f"S_l{i}" for i in range(17)]
            )
            print(f"Wrote VJP CSV to {vjp_path}")

        try:
            for target_time in self.target_times:
                _, forecast_path, vjp_scores = self._run_single(
                    target_time=target_time,
                    init_fn=init_fn,
                    apply_fn=apply_fn,
                )
                print(f"Wrote predictions to {forecast_path}")
                if vjp_writer is not None and vjp_scores is not None:
                    row = [f"{target_time.isoformat()}Z"]
                    row += np.asarray(vjp_scores).tolist()
                    vjp_writer.writerow(row)
                progress.update(1)
        finally:
            if vjp_file is not None:
                vjp_file.close()

        total_elapsed = time.perf_counter() - t0
        ended_at = dt.datetime.now(dt.timezone.utc)
        progress.close()
        self._log_total_elapsed(
            started_at=started_at,
            ended_at=ended_at,
            elapsed_sec=total_elapsed,
        )
    


def main() -> int:
    # デフォルトの設定ファイルを使用（引数なし）
    runner = GraphCastSingleRun()
    
    # 内部で保持している時刻リストを使用してループ処理（外部からは触れない）
    runner.run()
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
