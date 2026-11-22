#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import utils
from eval_brain2token_retrieval import load_voxel2clip as load_brain2token
from eval_brain2token_retrieval import qwen_block_target
from models import BrainDiffusionPrior, BrainNetwork, Clipper, VersatileDiffusionPriorNetwork
from qwen3_vl_embedding import Qwen3VLEmbedder


VOXELS = {1: 15724, 2: 14278, 3: 15226, 4: 13153, 5: 13039, 6: 17907, 7: 12682, 8: 14386}


def build_mindeye(subj, device, use_projector=True):
    clip_size = 768
    num_tokens = 257
    voxel2clip = BrainNetwork(
        in_dim=VOXELS[subj],
        out_dim=num_tokens * clip_size,
        clip_size=clip_size,
        use_projector=use_projector,
    )
    prior_network = VersatileDiffusionPriorNetwork(
        dim=clip_size,
        depth=6,
        dim_head=64,
        heads=clip_size // 64,
        causal=False,
        num_tokens=num_tokens,
        learned_query_mode="pos_emb",
    ).to(device)
    return BrainDiffusionPrior(
        net=prior_network,
        image_embed_dim=clip_size,
        condition_on_text_encodings=False,
        timesteps=100,
        cond_drop_prob=0.2,
        image_embed_scale=None,
        voxel2clip=voxel2clip,
    ).to(device)


def normalize(x):
    return F.normalize(x.float(), dim=-1, eps=1e-6)


def full_metrics(sims):
    labels = torch.arange(sims.shape[0], device=sims.device)
    ranks = torch.argsort(sims, dim=1, descending=True)
    return {
        "top1": float((ranks[:, 0] == labels).float().mean().item()),
        "top5": float((ranks[:, :5] == labels[:, None]).any(dim=1).float().mean().item()),
        "num_top1_errors": int((ranks[:, 0] != labels).sum().item()),
    }


def random_k_top1(sims, random_k=300, repeats=100, seed=42):
    device = sims.device
    n = sims.shape[0]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    scores = []
    for _ in range(repeats):
        correct = 0
        for i in range(n):
            pool = torch.randperm(n - 1, generator=generator)[: random_k - 1]
            pool = torch.where(pool >= i, pool + 1, pool).to(device)
            candidates = torch.cat([torch.tensor([i], device=device), pool])
            if int(torch.argmax(sims[i, candidates]).item()) == 0:
                correct += 1
        scores.append(correct / n)
    return {
        "top1": float(sum(scores) / len(scores)),
        "repeats": repeats,
        "way": random_k,
    }


def bidirectional_metrics(preds, targets, random_k=300, repeats=100, seed=42, device="cuda"):
    preds = normalize(preds).to(device)
    targets = normalize(targets).to(device)
    sims = preds @ targets.T
    forward = {
        "query": "fmri",
        "gallery": "image",
        "full": full_metrics(sims),
        f"{random_k}way": random_k_top1(sims, random_k=random_k, repeats=repeats, seed=seed),
    }
    backward_sims = sims.T.contiguous()
    backward = {
        "query": "image",
        "gallery": "fmri",
        "full": full_metrics(backward_sims),
        f"{random_k}way": random_k_top1(backward_sims, random_k=random_k, repeats=repeats, seed=seed),
    }
    return {"fmri_to_image": forward, "image_to_fmri": backward}


def build_val_loader(args):
    val_url = f"{args.data_path}/webdataset_avg_split/test/test_subj0{args.subj}_" + "{0..1}.tar"
    meta_url = f"{args.data_path}/webdataset_avg_split/metadata_subj0{args.subj}.json"
    _, val_dl, _, _ = utils.get_dataloaders(
        args.batch_size,
        "images",
        num_devices=1,
        num_workers=args.num_workers,
        train_url=val_url,
        val_url=val_url,
        meta_url=meta_url,
        num_train=0,
        num_val=982,
        val_batch_size=args.batch_size,
        cache_dir=args.data_path,
        seed=args.seed,
        voxels_key="nsdgeneral.npy",
        to_tuple=["voxels", "images", "coco"],
        local_rank=0,
        world_size=1,
    )
    return val_dl


def extract_brain2token(args, device):
    num_tokens, target_dim = 256, 1152
    to_pil = T.ToPILImage()
    val_dl = build_val_loader(args)
    qwen = Qwen3VLEmbedder(model_name_or_path=args.qwen_model_path, torch_dtype=torch.bfloat16)
    qwen.model.eval().to(device).requires_grad_(False)
    voxel2clip = load_brain2token(args.brain2token_ckpt, args.subj, num_tokens, target_dim, device)

    raw_preds, proj_preds, targets = [], [], []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for voxel, image, _coco in tqdm(val_dl, desc="Brain2Token features"):
            voxel = torch.mean(voxel, axis=1).float().to(device)
            target = qwen_block_target(qwen, image, args.target_layer, num_tokens, target_dim, device, to_pil)
            raw, proj = voxel2clip(voxel)
            raw_preds.append(raw.reshape(len(voxel), -1).float().cpu())
            proj_preds.append(proj.reshape(len(voxel), -1).float().cpu())
            targets.append(target.reshape(len(voxel), -1).float().cpu())
    return {
        "raw_prediction": (torch.cat(raw_preds)[:982], torch.cat(targets)[:982]),
        "projector_prediction": (torch.cat(proj_preds)[:982], torch.cat(targets)[:982]),
    }


def extract_mindeye(args, device):
    val_dl = build_val_loader(args)
    model = build_mindeye(args.subj, device)
    ckpt = torch.load(args.mindeye_ckpt, map_location="cpu")
    state_dict = ckpt["model_state_dict"]
    state_dict.pop("voxel2clip.temp", None)
    model.load_state_dict(state_dict, strict=True)
    model.eval().requires_grad_(False)
    clipper = Clipper("ViT-L/14", device=device, hidden_state=True, norm_embs=True)
    clipper.eval().requires_grad_(False)

    raw_preds, proj_preds, targets = [], [], []
    with torch.no_grad():
        for voxel, image, _coco in tqdm(val_dl, desc="MindEye features"):
            voxel = torch.mean(voxel, axis=1).float().to(device)
            target = clipper.embed_image(image.to(device)).float()
            raw, proj = model.voxel2clip(voxel)
            raw_preds.append(raw.reshape(len(voxel), -1).float().cpu())
            proj_preds.append(proj.reshape(len(voxel), -1).float().cpu())
            targets.append(target.reshape(len(voxel), -1).float().cpu())
    return {
        "raw_prediction": (torch.cat(raw_preds)[:982], torch.cat(targets)[:982]),
        "projector_prediction": (torch.cat(proj_preds)[:982], torch.cat(targets)[:982]),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate bidirectional fMRI-image retrieval.")
    parser.add_argument("--method", choices=["brain2token", "mindeye"], required=True)
    parser.add_argument("--subj", type=int, default=1)
    parser.add_argument("--data_path", default=os.environ.get("NSD_DATA_PATH", "/path/to/naturalscenesdataset"))
    parser.add_argument("--qwen_model_path", default=os.environ.get("QWEN_MODEL_PATH", "Qwen/Qwen3-VL-Embedding-8B"))
    parser.add_argument("--brain2token_ckpt", default=None)
    parser.add_argument("--mindeye_ckpt", default=None)
    parser.add_argument("--target_layer", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--random_k", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.method == "brain2token" and not args.brain2token_ckpt:
        parser.error("--brain2token_ckpt is required for --method brain2token")
    if args.method == "mindeye" and not args.mindeye_ckpt:
        parser.error("--mindeye_ckpt is required for --method mindeye")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    utils.seed_everything(args.seed, cudnn_deterministic=False)

    if args.method == "brain2token":
        features = extract_brain2token(args, device)
    else:
        features = extract_mindeye(args, device)

    result = {
        "method": args.method,
        "subj": args.subj,
        "num_samples": 982,
        "random_k": args.random_k,
        "repeats": args.repeats,
    }
    for name, (preds, targets) in features.items():
        result[name] = bidirectional_metrics(
            preds,
            targets,
            random_k=args.random_k,
            repeats=args.repeats,
            seed=args.seed,
            device=device,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
