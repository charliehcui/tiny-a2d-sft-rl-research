# Draft reproduction image; not built or GPU-tested during packaging.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
WORKDIR /workspace/bluewhale
COPY requirements.txt ./
COPY third_party/ ./third_party/
COPY scripts/install.sh ./scripts/install.sh
RUN bash scripts/install.sh
COPY . .
ENV HF_HOME=/opt/bluewhale_cache/huggingface \
    HF_DATASETS_CACHE=/opt/bluewhale_cache/datasets \
    MPLCONFIGDIR=/opt/bluewhale_cache/matplotlib \
    TOKENIZERS_PARALLELISM=false \
    WANDB_MODE=disabled \
    PYTHONDONTWRITEBYTECODE=1
RUN mkdir -p /opt/bluewhale_cache/exports
CMD ["bash"]
