# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""モデルのロールアウトのためのユーティリティ。"""

from typing import Iterator, Optional, Sequence

from absl import logging
import chex
import dask.array
from graphcast import xarray_jax
from graphcast import xarray_tree
import jax
import numpy as np
import typing_extensions
import xarray


class PredictorFn(typing_extensions.Protocol):
  """明示的なrngを持つbase.Predictor.__call__の関数バージョン。"""

  def __call__(
      self, rng: chex.PRNGKey, inputs: xarray.Dataset,
      targets_template: xarray.Dataset,
      forcings: xarray.Dataset,
      **optional_kwargs,
      ) -> xarray.Dataset:
    ...


def _replicate_dataset(
    data: xarray.Dataset, replica_dim: str,
    replicate_to_device: bool,
    devices: Sequence[jax.Device],
    ) -> xarray.Dataset:
  """xarray_jax.pmapの準備に使用されます。"""

  def replicate_variable(variable: xarray.Variable) -> xarray.Variable:
    if replica_dim in variable.dims:
      # TODO(pricei): replicate_to_device==Trueの場合、device_put_replicatedを呼び出す
      return variable.transpose(replica_dim, ...)
    else:
      data = len(devices) * [variable.data]
      if replicate_to_device:
        assert devices is not None
        # TODO(pricei): device_put_shardedの代わりに「device_put_replicated」を使用するようにコードをリファクタリングする。
        data = jax.device_put_sharded(data, devices)
      else:
        data = np.stack(data, axis=0)
      return xarray_jax.Variable(
          data=data, dims=(replica_dim,) + variable.dims, attrs=variable.attrs
      )

  def replicate_dataset(dataset: xarray.Dataset) -> xarray.Dataset:
    if dataset is None:
      return None
    data_variables = {
        name: replicate_variable(var)
        for name, var in dataset.data_vars.variables.items()
    }
    coords = {name: coord.variable for name, coord in dataset.coords.items()}
    return xarray.Dataset(data_variables, coords=coords, attrs=dataset.attrs)

  return replicate_dataset(data)


def chunked_prediction_generator_multiple_runs(
    predictor_fn: PredictorFn,
    rngs: chex.PRNGKey,
    inputs: xarray.Dataset,
    targets_template: xarray.Dataset,
    forcings: Optional[xarray.Dataset],
    num_samples: Optional[int],
    pmap_devices: Optional[Sequence[jax.Device]] = None,
    **chunked_prediction_kwargs,
) -> Iterator[xarray.Dataset]:
  """チャンク化された予測を生成することで、複数のサンプルの軌跡を出力します。

  引数:
    predictor_fn: 各チャンクの予測に使用する関数。
    rngs: 各アンサンブルメンバーに使用されるRNGシーケンス。
    inputs: モデルの入力。
    targets_template: ターゲット予測のテンプレート、時間的に等間隔のターゲットが必要です。
    forcings: モデルのオプションの強制力。
    num_samples: ロールアウトする実行/サンプルの数。
    pmap_devices: predictor_fnがpmapされるデバイスのリスト、またはpmapされていない場合はNone。
    **chunked_prediction_kwargs:
      chunked_predictionを参照、これらの一部は必須引数です。

  生成:
    チャンク化されたロールアウトの各チャンクステップの予測。すべての予測が時間的に連結され、
    サンプル次元が圧縮された場合、これは構造的にターゲットテンプレートと一致します。

  """
  if pmap_devices is not None:
    assert (
        num_samples % len(pmap_devices) == 0
    ), "num_samplesはlen(pmap_devices)の倍数でなければなりません"

    def predictor_fn_pmap_named_args(rng, inputs, targets_template, forcings):
      targets_template = _replicate_dataset(
          targets_template,
          replica_dim="sample",
          replicate_to_device=True,
          devices=pmap_devices,
      )
      return predictor_fn(rng, inputs, targets_template, forcings)

    for i in range(0, num_samples, len(pmap_devices)):
      sample_idx = slice(i, i + len(pmap_devices))
      logging.info("サンプル %s / %s", sample_idx, num_samples)
      logging.flush()
      sample_group_rngs = rngs[sample_idx]

      if "sample" not in inputs.dims:
        sample_inputs = inputs
      else:
        sample_inputs = inputs.isel(sample=sample_idx, drop=True)

      sample_inputs = _replicate_dataset(
          sample_inputs,
          replica_dim="sample",
          replicate_to_device=True,
          devices=pmap_devices,
      )

      if forcings is not None:
        if "sample" not in forcings.dims:
          sample_forcings = forcings
        else:
          sample_forcings = forcings.isel(sample=sample_idx, drop=True)

        # TODO(pricei): ここでは、上記のtargets_templateのように`predictor_fn_pmap_named_args`内ではなく、
        # ロールアウトのすべてのタイムステップに対する完全な強制力を複製しています。これは、入力が既に複製されている
        # 入力と連結されるためです。これをリファクタリングして、チャンク予測がpmapで実行されているかどうかを認識し、
        # その場合は必要なタイムステップのみの複製とdevice_putをチャンク予測関数の一部として行うべきです。
        sample_forcings = _replicate_dataset(
            sample_forcings,
            replica_dim="sample",
            replicate_to_device=False,
            devices=pmap_devices,
        )
      else:
        sample_forcings = None

      for prediction_chunk in chunked_prediction_generator(
          predictor_fn=predictor_fn_pmap_named_args,
          rng=sample_group_rngs,
          inputs=sample_inputs,
          targets_template=targets_template,
          forcings=sample_forcings,
          pmap_devices=pmap_devices,
          **chunked_prediction_kwargs,
      ):
        prediction_chunk.coords["sample"] = np.arange(
            sample_idx.start, sample_idx.stop, sample_idx.step
        )
        yield prediction_chunk
        del prediction_chunk
  else:
    for i in range(num_samples):
      logging.info("サンプル %d/%d", i, num_samples)
      logging.flush()
      this_sample_rng = rngs[i]

      if "sample" in inputs.dims:
        sample_inputs = inputs.isel(sample=i, drop=True)
      else:
        sample_inputs = inputs

      sample_forcings = forcings
      if sample_forcings is not None:
        if "sample" in sample_forcings.dims:
          sample_forcings = sample_forcings.isel(sample=i, drop=True)

      for prediction_chunk in chunked_prediction_generator(
          predictor_fn=predictor_fn,
          rng=this_sample_rng,
          inputs=sample_inputs,
          targets_template=targets_template,
          forcings=sample_forcings,
          **chunked_prediction_kwargs):
        prediction_chunk.coords["sample"] = i
        yield prediction_chunk
        del prediction_chunk


def chunked_prediction(
    predictor_fn: PredictorFn,
    rng: chex.PRNGKey,
    inputs: xarray.Dataset,
    targets_template: xarray.Dataset,
    forcings: xarray.Dataset,
    num_steps_per_chunk: int = 1,
    verbose: bool = False,
) -> xarray.Dataset:
  """チャンク化された予測を反復的に連結することで長い軌跡を出力します。

  引数:
    predictor_fn: 各チャンクの予測に使用する関数。
    rng: ランダムキー。
    inputs: モデルの入力。
    targets_template: ターゲット予測のテンプレート、時間的に等間隔のターゲットが必要です。
    forcings: モデルのオプションの強制力。
    num_steps_per_chunk: `predictor_fn`の各呼び出しで予測する`targets_template`のステップ数。
        これは`targets_template`のステップ数を均等に分割する必要があります。
    verbose: 予測中のチャンクをログに記録するかどうか。

  戻り値:
    ターゲットテンプレートの予測。

  """
  chunks_list = []
  for prediction_chunk, state in chunked_prediction_generator(
      predictor_fn=predictor_fn,
      rng=rng,
      inputs=inputs,
      targets_template=targets_template,
      forcings=forcings,
      num_steps_per_chunk=num_steps_per_chunk,
      verbose=verbose):
    chunks_list.append(jax.device_get(prediction_chunk))
  return xarray.concat(chunks_list, dim="time"), state


def chunked_prediction_generator(
    predictor_fn: PredictorFn,
    rng: chex.PRNGKey,
    inputs: xarray.Dataset,
    targets_template: xarray.Dataset,
    forcings: xarray.Dataset,
    num_steps_per_chunk: int = 1,
    verbose: bool = False,
    pmap_devices: Optional[Sequence[jax.Device]] = None
) -> Iterator[xarray.Dataset]:
  """チャンク化された予測を生成することで長い軌跡を出力します。

  引数:
    predictor_fn: 各チャンクの予測に使用する関数。
    rng: ランダムキー。
    inputs: モデルの入力。
    targets_template: ターゲット予測のテンプレート、時間的に等間隔のターゲットが必要です。
    forcings: モデルのオプションの強制力。
    num_steps_per_chunk: `predictor_fn`の各呼び出しで予測する`targets_template`のステップ数。
        これは`targets_template`のステップ数を均等に分割する必要があります。
    verbose: 予測中のチャンクをログに記録するかどうか。
    pmap_devices: predictor_fnがpmapされるデバイスのリスト、またはpmapされていない場合はNone。

  生成:
    チャンク化されたロールアウトの各チャンクステップの予測。すべての予測が時間的に連結された場合、
    これは構造的にターゲットテンプレートと一致します。

  """

  # 入力を変更しないようにコピーを作成します。
  inputs = xarray.Dataset(inputs)
  targets_template = xarray.Dataset(targets_template)
  forcings = xarray.Dataset(forcings)

  if "datetime" in inputs.coords:
    del inputs.coords["datetime"]

  if "datetime" in targets_template.coords:
    output_datetime = targets_template.coords["datetime"]
    del targets_template.coords["datetime"]
  else:
    output_datetime = None

  if "datetime" in forcings.coords:
    del forcings.coords["datetime"]

  num_target_steps = targets_template.dims["time"]
  num_chunks, remainder = divmod(num_target_steps, num_steps_per_chunk)
  if remainder != 0:
    raise ValueError(
        f"チャンクごとのステップ数 {num_steps_per_chunk} は "
        f"ターゲットステップ数 {num_target_steps} を均等に分割する必要があります")

  if len(np.unique(np.diff(targets_template.coords["time"].data))) > 1:
    raise ValueError("ターゲットの時間座標は均等に間隔を空ける必要があります")

  # テンプレートターゲットには常に、最初のチャンクのタイムデルタに対応する時間軸があります。
  targets_chunk_time = targets_template.time.isel(
      time=slice(0, num_steps_per_chunk))

  current_inputs = inputs

  def split_rng_fn(rng):
    # 注意：これは `return jax.random.split(rng)` と同等ではありません。なぜなら、
    # タプルに割り当てることで、`jax.random.split` によって返される単一のnumpy配列が
    # 実際に2つの配列に分割されるからです。そのため、pmapで関数を呼び出すと、
    # 出力はTuple[Array, Array]となり、各配列の先頭軸は `num devices` です。
    rng1, rng2 = jax.random.split(rng)
    return rng1, rng2

  if pmap_devices is not None:
    split_rng_fn = jax.pmap(split_rng_fn, devices=pmap_devices)

  for chunk_index in range(num_chunks):
    if verbose:
      logging.info("チャンク %d/%d", chunk_index, num_chunks)
      logging.flush()

    # このチャンクで予測している時間期間のターゲットを選択します。
    target_offset = num_steps_per_chunk * chunk_index
    target_slice = slice(target_offset, target_offset + num_steps_per_chunk)
    current_targets_template = targets_template.isel(time=target_slice)

    # タイムデルタを最初のチャンクに対応するものに置き換えて、
    # 毎回再コンパイルしないようにします。
    actual_target_time = current_targets_template.coords["time"]
    current_targets_template = current_targets_template.assign_coords(
        time=targets_chunk_time).compute()

    current_forcings = forcings.isel(time=target_slice)
    current_forcings = current_forcings.assign_coords(time=targets_chunk_time)
    current_forcings = current_forcings.compute()
    # チャンクの予測を行います。
    rng, this_rng = split_rng_fn(rng)
    predictions, state = predictor_fn(
        rng=this_rng,
        inputs=current_inputs,
        targets_template=current_targets_template,
        forcings=current_forcings)

    # pmapされた場合、プロファイリングにより予測、強制力、入力がすべて単一のTPUにコピーされ、
    # OOMを引き起こすことが明らかになっています。これを避けるために、
    # すべての入出力データをデバイスから取り出します。これにはパフォーマンスへの
    # 影響がありますが、メモリ効率を最大化します。
    # TODO(aelkadi): pmapの下で実行する場合は `_get_next_inputs` をpmapし、
    # device_getを削除します。
    if pmap_devices is not None:
      predictions = jax.device_get(predictions)
      current_forcings = jax.device_get(current_forcings)
      current_inputs = jax.device_get(current_inputs)

    next_frame = xarray.merge([predictions, current_forcings])

    next_inputs = _get_next_inputs(current_inputs, next_frame)

    # 毎回再コンパイルしないように、タイムデルタ座標をシフトします。
    next_inputs = next_inputs.assign_coords(time=current_inputs.coords["time"])
    current_inputs = next_inputs

    # この時点で、実際のターゲット時間座標を割り当てることができます。
    predictions = predictions.assign_coords(time=actual_target_time)
    if output_datetime is not None:
      predictions.coords["datetime"] = output_datetime.isel(
          time=target_slice)
    yield predictions, state
    del predictions


def _get_next_inputs(
    prev_inputs: xarray.Dataset, next_frame: xarray.Dataset,
    ) -> xarray.Dataset:
  """前の入力と予測から次の入力を計算します。"""

  # 時間軸を持つすべての入力を予測していることを確認します。
  non_predicted_or_forced_inputs = list(
      set(prev_inputs.keys()) - set(next_frame.keys()))
  if "time" in prev_inputs[non_predicted_or_forced_inputs].dims:
    raise ValueError(
        "予測または強制されていない時間インデックスを持つ入力が見つかりました。")

  # 予測から入力にコピーする必要があるキー。
  next_inputs_keys = list(
      set(next_frame.keys()).intersection(set(prev_inputs.keys())))
  next_inputs = next_frame[next_inputs_keys]

  # 次のフレームを入力と連結し、必要のないものを切り取ります。
  num_inputs = prev_inputs.dims["time"]
  return (
      xarray.concat(
          [prev_inputs, next_inputs], dim="time", data_vars="different")
      .tail(time=num_inputs))


def extend_targets_template(
    targets_template: xarray.Dataset,
    required_num_steps: int) -> xarray.Dataset:
  """遅延配列を使用して`targets_template`を`required_num_steps`に拡張します。

  メモリ内で配列をインスタンス化する必要がないように、遅延dask配列のゼロを使用します。

  引数:
    targets_template: 拡張する入力テンプレート。
    required_num_steps: 返されるテンプレートで必要なステップ数。

  戻り値:
    `targets_template`と変数とタイムステップが同一の`xarray.Dataset`で、
    時間軸が`required_num_steps`を持つ`dask.array.zeros`で満たされています。

  """

  # "time"と"datetime"座標を拡張します
  time = targets_template.coords["time"]

  # 最初のターゲット時間がタイムステップに対応することを確認します。
  timestep = time[0].data
  if time.shape[0] > 1:
    assert np.all(timestep == time[1:] - time[:-1])

  extended_time = (np.arange(required_num_steps) + 1) * timestep

  if "datetime" in targets_template.coords:
    datetime = targets_template.coords["datetime"]
    extended_datetime = (datetime[0].data - timestep) + extended_time
  else:
    extended_datetime = None

  # 値を空のdask配列に置き換えて、時間座標を拡張します。
  datetime = targets_template.coords["time"]

  def extend_time(data_array: xarray.DataArray) -> xarray.DataArray:
    dims = data_array.dims
    shape = list(data_array.shape)
    shape[dims.index("time")] = required_num_steps
    dask_data = dask.array.zeros(
        shape=tuple(shape),
        chunks=-1,  # チャンク情報を直接`ChunksToZarr`に提供します。
        dtype=data_array.dtype)

    coords = dict(data_array.coords)
    coords["time"] = extended_time

    if extended_datetime is not None:
      coords["datetime"] = ("time", extended_datetime)

    return xarray.DataArray(
        dims=dims,
        data=dask_data,
        coords=coords)

  return xarray_tree.map_structure(extend_time, targets_template)
