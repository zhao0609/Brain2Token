#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/eval_test982_subj01.sh runs/Brain2Token_subj01_block8_L8_4/best.pth

CKPT="${1:?checkpoint path required}"

python -m brain2token.eval_brain2token_retrieval \
  --ckpt "$CKPT" \
  --subj 1 \
  --target_layer 8 \
  --batch_size 16 \
  --out "$(dirname "$CKPT")/retrieval_eval_test982.json"
