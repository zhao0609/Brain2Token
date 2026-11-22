#!/usr/bin/env bash
set -euo pipefail

# Required:
#   export NSD_DATA_PATH=/path/to/naturalscenesdataset

python -m brain2token.audit_splits \
  --subj 1 \
  --out docs/split_audit_subj01.json
