#!/usr/bin/env bash
set -euo pipefail

# Required:
#   export NSD_DATA_PATH=/path/to/naturalscenesdataset
#   export QWEN_MODEL_PATH=/path/to/Qwen3-VL-Embedding-8B
#
# This code-only release does not ship model weights.
# Usage:
#   bash scripts/eval_pretrained_subj01.sh /path/to/checkpoint.pth
#
# The evaluation configuration corresponds to the run described in
# docs/training_config_subj01.json (subj01, layer 8, 982-image test gallery).

CKPT="${1:?checkpoint path required (no weights are bundled with this package)}"

python -m brain2token.eval_brain2token_retrieval \
  --ckpt "$CKPT" \
  --subj 1 \
  --target_layer 8 \
  --batch_size 16 \
  --out "$(dirname "$CKPT")/retrieval_eval_982_batch1.json"
