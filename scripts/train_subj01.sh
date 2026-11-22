#!/usr/bin/env bash
set -euo pipefail

# Required:
#   export NSD_DATA_PATH=/path/to/naturalscenesdataset
#   export QWEN_MODEL_PATH=/path/to/Qwen3-VL-Embedding-8B

python -m brain2token.train_brain2token \
  --subj 1 \
  --target_layer 8 \
  --batch_size 4 \
  --val_batch_size 16 \
  --num_epochs 240 \
  --model_name Brain2Token_subj01_block8 \
  --out_root runs \
  --no-use_image_aug \
  --no-prior
