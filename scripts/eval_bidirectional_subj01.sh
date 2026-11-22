#!/usr/bin/env bash
set -euo pipefail

# Required:
#   export NSD_DATA_PATH=/path/to/naturalscenesdataset
#   export QWEN_MODEL_PATH=/path/to/Qwen3-VL-Embedding-8B
#
# Usage:
#   bash scripts/eval_bidirectional_subj01.sh /path/to/best.pth

CKPT="${1:?checkpoint path required}"

python -m brain2token.eval_bidirectional_retrieval \
  --method brain2token \
  --brain2token_ckpt "$CKPT" \
  --subj 1 \
  --target_layer 8 \
  --batch_size 16 \
  --out "$(dirname "$CKPT")/retrieval_eval_bidirectional_test982.json"
