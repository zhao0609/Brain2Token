import argparse
import json
import os
import sys
import types
from pathlib import Path

import huggingface_hub


class FakeHfFolder:
    @classmethod
    def get_token(cls):
        return None


huggingface_hub.HfFolder = FakeHfFolder
if not hasattr(huggingface_hub, "utils"):
    huggingface_hub.utils = types.ModuleType("huggingface_hub.utils")
huggingface_hub.utils.HfFolder = FakeHfFolder
sys.modules["huggingface_hub.utils._hf_folder"] = types.ModuleType("mock_folder")
sys.modules["huggingface_hub.utils._hf_folder"].HfFolder = FakeHfFolder

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import utils
from models import BrainNetwork
from qwen_vision_bridge import extract_vision_block_features
from qwen3_vl_embedding import Qwen3VLEmbedder


def qwen_block_target(qwen_extractor, images, target_layer, num_tokens, target_dim, device, to_pil):
    pil_images = [to_pil(images[i].detach().cpu().clip(0, 1)).convert("RGB") for i in range(images.shape[0])]
    conversations = [qwen_extractor.format_model_input(image=img) for img in pil_images]
    processed = qwen_extractor._preprocess_inputs(conversations)
    processed = {key: value.to(device) for key, value in processed.items()}
    features = extract_vision_block_features(
        qwen_extractor.model,
        pixel_values=processed["pixel_values"],
        image_grid_thw=processed["image_grid_thw"],
        target_layers=(target_layer,),
    )[target_layer]
    return features.reshape(images.shape[0], num_tokens, target_dim)


def load_voxel2clip(ckpt_path, subj, num_tokens, target_dim, device):
    voxel_dict = {1: 15724, 2: 14278, 3: 15226, 4: 13153, 5: 13039, 6: 17907, 7: 12682, 8: 14386}
    model = BrainNetwork(
        in_dim=voxel_dict[subj],
        out_dim=num_tokens * target_dim,
        clip_size=target_dim,
        use_projector=True,
    ).to(device)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    v2c_state = {}
    for key, value in state_dict.items():
        if key.startswith("voxel2clip."):
            v2c_state[key.replace("voxel2clip.", "", 1)] = value
    missing, unexpected = model.load_state_dict(v2c_state, strict=False)
    print(f"Loaded voxel2clip from {ckpt_path}")
    print(f"Missing keys: {len(missing)} Unexpected keys: {len(unexpected)}")
    model.eval()
    return model


def retrieval_metrics(preds, targets, random_k=300, repeats=100, seed=42):
    preds = F.normalize(preds.float(), dim=-1, eps=1e-6)
    targets = F.normalize(targets.float(), dim=-1, eps=1e-6)
    sims = preds @ targets.T
    labels = torch.arange(sims.shape[0], device=sims.device)
    ranks = torch.argsort(sims, dim=1, descending=True)
    top1 = (ranks[:, 0] == labels).float().mean().item()
    top5 = (ranks[:, :5] == labels[:, None]).any(dim=1).float().mean().item()

    generator = torch.Generator(device=sims.device).manual_seed(seed)
    correct = []
    n = sims.shape[0]
    for _ in range(repeats):
        rows = torch.arange(n, device=sims.device)
        distractors = torch.randint(0, n, (n, random_k - 1), generator=generator, device=sims.device)
        candidates = torch.cat([labels[:, None], distractors], dim=1)
        cand_scores = sims[rows[:, None], candidates]
        correct.append((cand_scores.argmax(dim=1) == 0).float().mean().item())
    return {
        "top1_full": top1,
        "top5_full": top5,
        f"top1_{random_k}way_mean": sum(correct) / len(correct),
        f"top1_{random_k}way_repeats": repeats,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate fMRI -> Qwen block8 feature retrieval.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--qwen_model_path", default=os.environ.get("QWEN_MODEL_PATH", "Qwen/Qwen3-VL-Embedding-8B"))
    parser.add_argument("--data_path", default=os.environ.get("NSD_DATA_PATH", "/path/to/naturalscenesdataset"))
    parser.add_argument("--subj", type=int, default=1)
    parser.add_argument("--target_layer", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_tokens = 256
    target_dim = 1152
    to_pil = T.ToPILImage()

    val_url = f"{args.data_path}/webdataset_avg_split/test/test_subj0{args.subj}_" + "{0..1}.tar"
    meta_url = f"{args.data_path}/webdataset_avg_split/metadata_subj0{args.subj}.json"
    _, val_dl, _, _ = utils.get_dataloaders(
        args.batch_size,
        "images",
        num_devices=1,
        num_workers=1,
        train_url=val_url,
        val_url=val_url,
        meta_url=meta_url,
        num_train=0,
        num_val=982,
        val_batch_size=args.batch_size,
        cache_dir=args.data_path,
        seed=42,
        voxels_key="nsdgeneral.npy",
        to_tuple=["voxels", "images", "coco"],
        local_rank=0,
        world_size=1,
    )

    qwen = Qwen3VLEmbedder(model_name_or_path=args.qwen_model_path, torch_dtype=torch.bfloat16)
    qwen.model.eval().to(device)
    qwen.model.requires_grad_(False)
    voxel2clip = load_voxel2clip(args.ckpt, args.subj, num_tokens, target_dim, device)

    raw_preds, proj_preds, targets = [], [], []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        for batch_i, (voxel, image, _coco) in enumerate(tqdm(val_dl)):
            if args.max_batches is not None and batch_i >= args.max_batches:
                break
            voxel = torch.mean(voxel, axis=1).float().to(device)
            target = qwen_block_target(qwen, image, args.target_layer, num_tokens, target_dim, device, to_pil)
            raw, proj = voxel2clip(voxel)
            raw_preds.append(raw.reshape(len(voxel), -1).float().cpu())
            proj_preds.append(proj.reshape(len(voxel), -1).float().cpu())
            targets.append(target.reshape(len(voxel), -1).float().cpu())

    raw_preds = torch.cat(raw_preds)
    proj_preds = torch.cat(proj_preds)
    targets = torch.cat(targets)
    results = {
        "ckpt": args.ckpt,
        "num_samples": int(targets.shape[0]),
        "raw_prediction": retrieval_metrics(raw_preds, targets),
        "projector_prediction": retrieval_metrics(proj_preds, targets),
    }
    out = Path(args.out) if args.out else Path(args.ckpt).with_name("retrieval_eval_test982.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
