# ベースイメージ（ダイジェスト固定で再現性確保）
FROM ghcr.io/nvidia/jax:jax@sha256:0c837e4936b5e349eac8fe6d1d1ea2cc7baf3648e28bac4fcd22c9d3891cf27a

# 作業ディレクトリ設定（オプション）
WORKDIR /workspace
COPY graphcast_pipeline/requirements_graphcast.txt .
RUN pip install --no-cache-dir -r requirements_graphcast.txt

# XLA 関連メモリ設定（必要に応じて docker run -e で上書き可能）
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 \
    TF_GPU_ALLOCATOR=cuda_malloc_async

CMD ["/bin/bash"]
