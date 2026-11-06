#!/usr/bin/env bash
# From the submission root: bash scripts/install.sh
set -euo pipefail
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
mkdir -p third_party/dllm third_party/lm-evaluation-harness
tar -xzf third_party/dllm.tar.gz -C third_party/dllm --strip-components=1
tar -xzf third_party/lm-evaluation-harness.tar.gz -C third_party/lm-evaluation-harness --strip-components=1
python -m pip install -r requirements.txt
python -m pip install -e third_party/lm-evaluation-harness
python -m pip install --no-deps -e third_party/dllm
python -c 'import torch, transformers, dllm, lm_eval; print("torch", torch.__version__, "transformers", transformers.__version__, "CUDA", torch.cuda.is_available())'
