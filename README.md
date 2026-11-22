# Brain2Token

Brain2Token aligns fMRI responses with intermediate visual-token representations from
Qwen3-VL-Embedding-8B. This package contains the cleaned subject-1 (subj01) training and
retrieval code together with the training configuration of the subject-1 run reported in
the paper. 

## Contents

```
brain2token/
  train_brain2token.py          # fMRI-to-token training entry point
  eval_brain2token_retrieval.py # full-gallery / 300-way retrieval evaluation
  eval_bidirectional_retrieval.py
  models.py                     # BrainNetwork, projector, diffusion prior
  qwen_vision_bridge.py         # frozen Qwen vision-block feature extraction
  qwen3_vl_embedding.py         # Qwen3-VL embeddings helper
  utils.py, configurator.py, audit_splits.py
scripts/                        # one-command training / evaluation / split audit
docs/                           # training configuration and split-overlap audit
results_examples/               # example retrieval outputs
requirements.txt
```

NSD data, Qwen3-VL weights, and the trained Brain2Token weights are not included.

## Environment

```bash
export NSD_DATA_PATH=/path/to/naturalscenesdataset
export QWEN_MODEL_PATH=/path/to/Qwen3-VL-Embedding-8B
pip install -r requirements.txt
```

Expected NSD WebDataset layout:

```
$NSD_DATA_PATH/webdataset_avg_split/
  train/train_subj01_{0..17}.tar
  val/val_subj01_0.tar
  test/test_subj01_{0..1}.tar
  metadata_subj01.json
```

## Training

```bash
bash scripts/train_subj01.sh
```

Training uses the 8,559 images in the subject-1 training split; the 300-image validation
split is used for training-time monitoring. The 982-image test split is used only by the
evaluation scripts below. Outputs are written to `runs/Brain2Token_subj01_block8_L8_4/`,
including periodic epoch checkpoints at `epoch_160.pth`, `epoch_180.pth`, etc.

`docs/training_config_subj01.json` records the exact hyperparameters of the reported
subject-1 block-8 run (seed 42, max LR 1e-4, batch size 4, 240 epochs, `mixup_pct` 0.33,
loss weights λ_NCE = λ_MSE = λ_tok-cos = 1, diffusion prior disabled, projector enabled).
The reference result below uses the fixed epoch-180 snapshot of this run; the fingerprint
is listed only to identify the weights used for the reported numbers, and the file itself
is not distributed here.

- Reference checkpoint: `subj01_block8_epoch180.pth` (5,395,046,812 bytes)
- SHA-256: `9acef11c7d817fa8d306c5116ce1979ac6250acada85320ac84d982d60f46e9b`

The checkpoint contains only the `voxel2clip` decoder/projector weights needed for
inference; the diffusion-prior branch is disabled in this configuration.

Retrieval on the 982-image NSD test gallery (recomputed with the code in this package):

| Branch | top-1 | top-5 | errors | 95% CI (top-1) |
|---|---:|---:|---:|---|
| raw token prediction | 57.23% | 82.18% | 420 | [54.28%, 60.29%] |
| projector prediction | 99.29% | 100.00% | 7 | [98.78%, 99.80%] |

The projector readout matches the 99.29% / 7-error result reported in the paper.

## Retrieval evaluation

All evaluation scripts take a checkpoint path as an argument. Pass either a checkpoint
produced by `scripts/train_subj01.sh` or a separately obtained reference checkpoint:

```bash
bash scripts/eval_pretrained_subj01.sh /path/to/checkpoint.pth
# equivalent explicit-config variant:
bash scripts/eval_test982_subj01.sh /path/to/checkpoint.pth
```

Bidirectional (image-to-fMRI and fMRI-to-image) retrieval:

```bash
bash scripts/eval_bidirectional_subj01.sh /path/to/checkpoint.pth
```

Example outputs are in `results_examples/`. The projector branch is the readout reported
in the paper.

## Split audit

```bash
bash scripts/audit_splits_subj01.sh
```

A precomputed local audit is in `docs/split_audit_subj01.local.json`; train, validation,
and test tar files have no COCO-ID or sample-key overlap.
