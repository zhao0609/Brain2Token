import argparse
import io
import json
import os
import tarfile
from pathlib import Path

import numpy as np


def expand_braces(pattern):
    if "{" not in pattern:
        return [pattern]
    prefix, rest = pattern.split("{", 1)
    body, suffix = rest.split("}", 1)
    if ".." not in body:
        return [pattern]
    start, end = body.split("..", 1)
    pad = (len(start) > 1 and start.startswith("0")) or (len(end) > 1 and end.startswith("0"))
    width = max(len(start), len(end)) if pad else 0
    if width:
        return [f"{prefix}{i:0{width}d}{suffix}" for i in range(int(start), int(end) + 1)]
    return [f"{prefix}{i}{suffix}" for i in range(int(start), int(end) + 1)]


def read_split(pattern):
    files = [Path(p) for p in expand_braces(pattern)]
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing tar files: {missing[:3]}")

    coco_ids = []
    sample_keys = []
    for path in files:
        with tarfile.open(path, "r") as tar:
            for member in tar:
                name = os.path.basename(member.name)
                if name.endswith(".coco73k.npy"):
                    raw = tar.extractfile(member).read()
                    arr = np.load(io.BytesIO(raw), allow_pickle=True)
                    coco_ids.append(int(np.asarray(arr).reshape(-1)[0]))
                    sample_keys.append(name.split(".", 1)[0])
    return {
        "pattern": pattern,
        "num_tar_files": len(files),
        "tar_files": [p.name for p in files],
        "num_samples": len(coco_ids),
        "num_unique_coco_ids": len(set(coco_ids)),
        "num_unique_sample_keys": len(set(sample_keys)),
        "coco_ids": set(coco_ids),
        "sample_keys": set(sample_keys),
        "first10_coco_ids": coco_ids[:10],
    }


def main():
    parser = argparse.ArgumentParser(description="Audit NSD WebDataset train/val/test split overlap.")
    parser.add_argument("--data_path", default=os.environ.get("NSD_DATA_PATH", "/path/to/naturalscenesdataset"))
    parser.add_argument("--subj", type=int, default=1)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(args.data_path) / "webdataset_avg_split"
    patterns = {
        "train": str(root / "train" / (f"train_subj0{args.subj}_" + "{0..17}.tar")),
        "val": str(root / "val" / f"val_subj0{args.subj}_0.tar"),
        "test": str(root / "test" / (f"test_subj0{args.subj}_" + "{0..1}.tar")),
    }
    splits = {name: read_split(pattern) for name, pattern in patterns.items()}

    overlaps = {}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        coco = splits[a]["coco_ids"] & splits[b]["coco_ids"]
        sample = splits[a]["sample_keys"] & splits[b]["sample_keys"]
        overlaps[f"{a}_vs_{b}"] = {
            "num_coco_id_overlap": len(coco),
            "num_sample_key_overlap": len(sample),
            "example_coco_ids": sorted(coco)[:20],
            "example_sample_keys": sorted(sample)[:20],
        }

    public = {}
    for name, info in splits.items():
        public[name] = {k: v for k, v in info.items() if k not in {"coco_ids", "sample_keys"}}
    result = {
        "subject": args.subj,
        "data_path": args.data_path,
        "splits": public,
        "overlaps": overlaps,
        "pass_no_overlap": all(v["num_coco_id_overlap"] == 0 and v["num_sample_key_overlap"] == 0 for v in overlaps.values()),
    }

    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
