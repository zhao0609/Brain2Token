import argparse
import json
import os
import sys
import types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from accelerate import Accelerator
from tqdm import tqdm

from qwen3_vl_embedding import Qwen3VLEmbedder

import utils
from models import BrainDiffusionPrior, BrainNetwork, VersatileDiffusionPriorNetwork
from qwen_vision_bridge import extract_vision_block_features


def parse_args():
    parser = argparse.ArgumentParser(description="Train fMRI -> Qwen3-VL intermediate vision-block features.")
    parser.add_argument("--model_name", type=str, default="Brain2Token_Block8")
    parser.add_argument("--qwen_model_path", type=str, default=os.environ.get("QWEN_MODEL_PATH", "Qwen/Qwen3-VL-Embedding-8B"))
    parser.add_argument("--data_path", type=str, default=os.environ.get("NSD_DATA_PATH", "/path/to/naturalscenesdataset"))
    parser.add_argument("--out_root", type=str, default=os.environ.get("BRAIN2TOKEN_OUT", "runs"))
    parser.add_argument("--subj", type=int, default=1, choices=[1, 2, 5, 7])
    parser.add_argument("--target_layer", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=240)
    parser.add_argument("--mixup_pct", type=float, default=0.33)
    parser.add_argument("--use_image_aug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt_interval", type=int, default=20)
    parser.add_argument("--epoch_ckpt_start", type=int, default=150)
    parser.add_argument("--save_epoch_ckpts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--light_epoch_ckpts", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--light_best_ckpt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--early_stop_min_epoch", type=int, default=150)
    parser.add_argument("--early_stop_patience", type=int, default=30)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    parser.add_argument("--prior_mult", type=float, default=30.0)
    parser.add_argument("--nce_mult", type=float, default=1.0)
    parser.add_argument("--token_mse_mult", type=float, default=1.0)
    parser.add_argument("--token_cos_mult", type=float, default=1.0)
    parser.add_argument("--soft_clip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prior", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--use_projector", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--init_from_ckpt", type=str, default=None)
    return parser.parse_args([]) if utils.is_interactive() else parser.parse_args()


def save_ckpt(path, epoch, diffusion_prior, optimizer, lr_scheduler, losses, val_losses, accelerator, include_optimizer=True):
    unwrapped = accelerator.unwrap_model(diffusion_prior)
    payload = {
        "epoch": epoch,
        "model_state_dict": unwrapped.state_dict(),
        "train_losses": losses,
        "val_losses": val_losses,
    }
    if include_optimizer:
        payload["optimizer_state_dict"] = optimizer.state_dict()
        payload["lr_scheduler"] = lr_scheduler.state_dict()
    torch.save(payload, path)


def load_compatible_state_dict(model, state_dict):
    """Load checkpoint tensors whose names and shapes match the current model.

    This enables cross-subject transfer: downstream Qwen-space layers/projectors
    can be reused while subject-specific voxel input weights with different
    shapes are safely skipped.
    """
    current = model.state_dict()
    compatible = {}
    skipped_shape = []
    skipped_missing = []
    for key, value in state_dict.items():
        if key not in current:
            skipped_missing.append(key)
            continue
        if current[key].shape != value.shape:
            skipped_shape.append((key, tuple(value.shape), tuple(current[key].shape)))
            continue
        compatible[key] = value
    current.update(compatible)
    model.load_state_dict(current, strict=True)
    return compatible, skipped_shape, skipped_missing


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

    actual_tokens = features.shape[0] // images.shape[0]
    if actual_tokens != num_tokens:
        raise ValueError(f"Expected {num_tokens} Qwen vision tokens, got {actual_tokens}. Check preprocessing size.")
    return features.reshape(images.shape[0], num_tokens, target_dim)


def flatten_normalize(x):
    flat = torch.nan_to_num(x.to(torch.float32).flatten(1), nan=0.0, posinf=1e4, neginf=-1e4)
    denom = torch.linalg.vector_norm(flat, dim=-1, keepdim=True).clamp_min(1e-6)
    return flat / denom


def stable_logits(preds, targs, temp):
    logits = (preds @ targs.T).to(torch.float32) / float(temp)
    return torch.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100.0, 100.0)


def stable_mixco_nce(preds, targs, temp=0.006, perm=None, betas=None, select=None):
    logits = stable_logits(preds, targs, temp)
    if perm is None or betas is None or select is None:
        labels = torch.arange(logits.shape[0], device=logits.device)
        return (nn.functional.cross_entropy(logits, labels) + nn.functional.cross_entropy(logits.T, labels)) / 2

    betas = betas.to(logits.device, dtype=logits.dtype)
    perm = perm.to(logits.device)
    probs = torch.diag(betas)
    probs[torch.arange(preds.shape[0], device=logits.device), perm] = 1 - betas
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    log_probs_t = nn.functional.log_softmax(logits.T, dim=-1)
    return (-(log_probs * probs).sum(-1).mean() - (log_probs_t * probs.T).sum(-1).mean()) / 2


def stable_soft_clip_loss(preds, targs, temp):
    target_logits = stable_logits(targs, targs, temp)
    pred_logits = stable_logits(preds, targs, temp)
    target_probs = target_logits.softmax(dim=-1)
    loss1 = -(pred_logits.log_softmax(dim=-1) * target_probs).sum(-1).mean()
    loss2 = -(pred_logits.T.log_softmax(dim=-1) * target_probs.T).sum(-1).mean()
    return (loss1 + loss2) / 2


def token_cos_loss(pred_tokens, target_tokens):
    pred = nn.functional.normalize(pred_tokens.float(), dim=-1, eps=1e-6)
    target = nn.functional.normalize(target_tokens.float(), dim=-1, eps=1e-6)
    return 1 - nn.functional.cosine_similarity(pred, target, dim=-1).mean()


def main():
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    utils.seed_everything(args.seed, cudnn_deterministic=False)

    accelerator = Accelerator(split_batches=False, mixed_precision="bf16")
    print = accelerator.print
    device = accelerator.device
    distributed = accelerator.state.distributed_type != "NO"
    local_rank = accelerator.state.local_process_index
    world_size = accelerator.state.num_processes

    target_dim = 1152
    num_tokens = 256
    out_dim = num_tokens * target_dim
    model_name = f"{args.model_name}_L{args.target_layer}_{args.batch_size}"
    outdir = os.path.join(args.out_root, model_name)
    if local_rank == 0:
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "config.json"), "w") as f:
            json.dump(
                vars(args)
                | {
                    "num_tokens": num_tokens,
                    "target_dim": target_dim,
                    "out_dim": out_dim,
                },
                f,
                indent=2,
            )

    if args.use_image_aug:
        import kornia
        from kornia.augmentation.container import AugmentationSequential

        img_augment = AugmentationSequential(
            kornia.augmentation.RandomResizedCrop((256, 256), (0.6, 1), p=0.3),
            kornia.augmentation.Resize((256, 256)),
            kornia.augmentation.RandomHorizontalFlip(p=0.5),
            kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.3),
            kornia.augmentation.RandomGrayscale(p=0.3),
            data_keys=["input"],
        )

    train_url = f"{args.data_path}/webdataset_avg_split/train/train_subj0{args.subj}_" + "{0..17}.tar"
    val_url = f"{args.data_path}/webdataset_avg_split/val/val_subj0{args.subj}_0.tar"
    meta_url = f"{args.data_path}/webdataset_avg_split/metadata_subj0{args.subj}.json"
    num_train = 8559
    num_val = 300
    print(f"train_url: {train_url}")
    print(f"validation url: {val_url}")

    train_dl, val_dl, num_train, num_val = utils.get_dataloaders(
        args.batch_size,
        "images",
        num_devices=1,
        num_workers=args.num_workers,
        train_url=train_url,
        val_url=val_url,
        meta_url=meta_url,
        num_train=num_train,
        num_val=num_val,
        val_batch_size=args.val_batch_size,
        cache_dir=args.data_path,
        seed=args.seed,
        voxels_key="nsdgeneral.npy",
        to_tuple=["voxels", "images", "coco"],
        local_rank=local_rank,
        world_size=world_size,
    )

    print(f"Loading Qwen target extractor from {args.qwen_model_path}")
    qwen_extractor = Qwen3VLEmbedder(model_name_or_path=args.qwen_model_path, torch_dtype=torch.bfloat16)
    qwen_extractor.model.eval().to(device)
    qwen_extractor.model.requires_grad_(False)

    voxel_dict = {1: 15724, 2: 14278, 3: 15226, 4: 13153, 5: 13039, 6: 17907, 7: 12682, 8: 14386}
    voxel2clip = BrainNetwork(
        in_dim=voxel_dict[args.subj],
        out_dim=out_dim,
        clip_size=target_dim,
        use_projector=args.use_projector,
    )

    prior_network = VersatileDiffusionPriorNetwork(
        dim=target_dim,
        depth=6,
        dim_head=64,
        heads=target_dim // 64,
        causal=False,
        num_tokens=num_tokens,
        learned_query_mode="pos_emb",
    ).to(device)

    diffusion_prior = BrainDiffusionPrior(
        net=prior_network,
        image_embed_dim=target_dim,
        condition_on_text_encodings=False,
        timesteps=100,
        cond_drop_prob=0.2,
        image_embed_scale=None,
        voxel2clip=voxel2clip,
    ).to(device)
    diffusion_prior.requires_grad_(args.prior)
    diffusion_prior.voxel2clip.requires_grad_(True)

    if args.init_from_ckpt:
        checkpoint = torch.load(args.init_from_ckpt, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        compatible, skipped_shape, skipped_missing = load_compatible_state_dict(diffusion_prior, state_dict)
        print(f"Loaded init checkpoint from {args.init_from_ckpt}")
        print(f"Compatible tensors loaded: {len(compatible)}")
        print(f"Skipped missing-name tensors: {len(skipped_missing)}")
        print(f"Skipped shape-mismatch tensors: {len(skipped_shape)}")
        if skipped_shape:
            print(f"First skipped shape mismatch: {skipped_shape[0]}")
        del checkpoint

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for n, p in diffusion_prior.net.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 1e-2,
            },
            {
                "params": [p for n, p in diffusion_prior.net.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in diffusion_prior.voxel2clip.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 1e-2,
            },
            {
                "params": [p for n, p in diffusion_prior.voxel2clip.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ],
        lr=args.max_lr * accelerator.num_processes,
    )

    total_steps = int(args.num_epochs * (num_train // args.batch_size))
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.max_lr * accelerator.num_processes,
        total_steps=total_steps,
        final_div_factor=1000,
        pct_start=(2 / args.num_epochs) if args.num_epochs > 2 else 0.3,
    )

    diffusion_prior, optimizer, train_dl, val_dl, lr_scheduler = accelerator.prepare(
        diffusion_prior, optimizer, train_dl, val_dl, lr_scheduler
    )

    losses, val_losses, records = [], [], []
    best_val_loss = float("inf")
    best_epoch = -1
    soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, args.num_epochs - int(args.mixup_pct * args.num_epochs))
    to_pil = T.ToPILImage()

    progress_bar = tqdm(range(args.num_epochs), ncols=160, disable=(local_rank != 0))
    for epoch in progress_bar:
        should_stop = False
        diffusion_prior.train()
        train_nce_sum = train_prior_sum = train_sim_sum = train_mse_sum = train_token_cos_sum = 0.0

        for train_i, (voxel, image, _coco) in enumerate(train_dl):
            if args.max_train_batches is not None and train_i >= args.max_train_batches:
                break
            optimizer.zero_grad()
            repeat_index = train_i % 3
            if args.use_image_aug:
                image = img_augment(image)

            voxel = voxel[:, repeat_index].float()
            if epoch < int(args.mixup_pct * args.num_epochs):
                voxel, perm, betas, select = utils.mixco(voxel)
            else:
                perm = betas = select = None

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                target = qwen_block_target(qwen_extractor, image, args.target_layer, num_tokens, target_dim, device, to_pil)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                clip_voxels, clip_voxels_proj = diffusion_prior.module.voxel2clip(voxel) if distributed else diffusion_prior.voxel2clip(voxel)
                clip_voxels = clip_voxels.view(len(voxel), num_tokens, target_dim)
                if args.prior:
                    loss_prior, _ = diffusion_prior(text_embed=clip_voxels, image_embed=target)
                    loss_prior = torch.nan_to_num(loss_prior.float(), nan=0.0, posinf=1e4, neginf=1e4)
                else:
                    loss_prior = torch.zeros((), device=device)

            with torch.amp.autocast("cuda", enabled=False):
                pred_norm = flatten_normalize(clip_voxels_proj)
                target_norm = flatten_normalize(target)
                loss_mse = nn.functional.mse_loss(clip_voxels.float(), target.float())
                loss_token_cos = token_cos_loss(clip_voxels, target)
                if epoch < int(args.mixup_pct * args.num_epochs):
                    loss_nce = stable_mixco_nce(pred_norm, target_norm, temp=0.006, perm=perm, betas=betas, select=select)
                else:
                    epoch_temp = soft_loss_temps[epoch - int(args.mixup_pct * args.num_epochs)]
                    if args.soft_clip:
                        loss_nce = stable_soft_clip_loss(pred_norm, target_norm, temp=float(epoch_temp))
                    else:
                        loss_nce = stable_mixco_nce(pred_norm, target_norm, temp=float(epoch_temp))

            loss = (
                args.nce_mult * loss_nce
                + args.token_mse_mult * loss_mse
                + args.token_cos_mult * loss_token_cos
                + (args.prior_mult * loss_prior if args.prior else 0)
            )

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()

            losses.append(float(loss.detach().cpu()))
            train_nce_sum += float(loss_nce.detach().cpu())
            train_prior_sum += float(loss_prior.detach().cpu())
            train_mse_sum += float(loss_mse.detach().cpu())
            train_token_cos_sum += float(loss_token_cos.detach().cpu())
            train_sim_sum += nn.functional.cosine_similarity(pred_norm, target_norm).mean().item()

        diffusion_prior.eval()
        val_nce_sum = val_prior_sum = val_sim_sum = val_mse_sum = val_token_cos_sum = 0.0
        for val_i, (voxel, image, _coco) in enumerate(val_dl):
            if args.max_val_batches is not None and val_i >= args.max_val_batches:
                break
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                voxel = torch.mean(voxel, axis=1).float()
                target = qwen_block_target(qwen_extractor, image, args.target_layer, num_tokens, target_dim, device, to_pil)
                clip_voxels, clip_voxels_proj = diffusion_prior.module.voxel2clip(voxel) if distributed else diffusion_prior.voxel2clip(voxel)
                clip_voxels = clip_voxels.view(len(voxel), num_tokens, target_dim)
                if args.prior:
                    val_loss_prior, _ = diffusion_prior(text_embed=clip_voxels, image_embed=target)
                    val_loss_prior = torch.nan_to_num(val_loss_prior.float(), nan=0.0, posinf=1e4, neginf=1e4)
                else:
                    val_loss_prior = torch.zeros((), device=device)

                with torch.amp.autocast("cuda", enabled=False):
                    pred_norm = flatten_normalize(clip_voxels_proj)
                    target_norm = flatten_normalize(target)
                    val_loss_mse = nn.functional.mse_loss(clip_voxels.float(), target.float())
                    val_loss_token_cos = token_cos_loss(clip_voxels, target)
                    if epoch < int(args.mixup_pct * args.num_epochs):
                        val_loss_nce = stable_mixco_nce(pred_norm, target_norm, temp=0.006)
                    else:
                        if args.soft_clip:
                            val_loss_nce = stable_soft_clip_loss(pred_norm, target_norm, temp=float(epoch_temp))
                        else:
                            val_loss_nce = stable_mixco_nce(pred_norm, target_norm, temp=float(epoch_temp))
                val_loss = (
                    args.nce_mult * val_loss_nce
                    + args.token_mse_mult * val_loss_mse
                    + args.token_cos_mult * val_loss_token_cos
                    + (args.prior_mult * val_loss_prior if args.prior else 0)
                )

                val_losses.append(float(val_loss.detach().cpu()))
                val_nce_sum += float(val_loss_nce.detach().cpu())
                val_prior_sum += float(val_loss_prior.detach().cpu())
                val_mse_sum += float(val_loss_mse.detach().cpu())
                val_token_cos_sum += float(val_loss_token_cos.detach().cpu())
                val_sim_sum += nn.functional.cosine_similarity(pred_norm, target_norm).mean().item()

        if local_rank == 0:
            train_count = train_i + 1
            val_count = val_i + 1
            val_loss_avg = float(np.mean(val_losses[-val_count:]))
            record = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses[-train_count:])),
                "val_loss": val_loss_avg,
                "train_nce": train_nce_sum / train_count,
                "train_mse": train_mse_sum / train_count,
                "train_token_cos_loss": train_token_cos_sum / train_count,
                "train_prior": train_prior_sum / train_count,
                "train_cosine": train_sim_sum / train_count,
                "val_nce": val_nce_sum / val_count,
                "val_mse": val_mse_sum / val_count,
                "val_token_cos_loss": val_token_cos_sum / val_count,
                "val_prior": val_prior_sum / val_count,
                "val_cosine": val_sim_sum / val_count,
                "lr": optimizer.param_groups[0]["lr"],
            }
            records.append(record)
            with open(os.path.join(outdir, "results.json"), "w") as f:
                json.dump(records, f, indent=2)

            improved = val_loss_avg < (best_val_loss - args.early_stop_min_delta)
            if improved:
                best_val_loss = val_loss_avg
                best_epoch = epoch
                save_ckpt(
                    os.path.join(outdir, "best.pth"),
                    epoch,
                    diffusion_prior,
                    optimizer,
                    lr_scheduler,
                    losses,
                    val_losses,
                    accelerator,
                    include_optimizer=not args.light_best_ckpt,
                )
            if args.save_epoch_ckpts and epoch >= args.epoch_ckpt_start and epoch % args.ckpt_interval == 0:
                save_ckpt(
                    os.path.join(outdir, f"epoch_{epoch}.pth"),
                    epoch,
                    diffusion_prior,
                    optimizer,
                    lr_scheduler,
                    losses,
                    val_losses,
                    accelerator,
                    include_optimizer=not args.light_epoch_ckpts,
                )
            save_ckpt(os.path.join(outdir, "rolling_last.pth"), epoch, diffusion_prior, optimizer, lr_scheduler, losses, val_losses, accelerator)
            if (
                args.early_stop_patience > 0
                and epoch >= args.early_stop_min_epoch
                and best_epoch >= 0
                and (epoch - best_epoch) >= args.early_stop_patience
            ):
                should_stop = True
                print(
                    f"early stopping at epoch {epoch}: best_epoch={best_epoch}, "
                    f"best_val_loss={best_val_loss:.6f}, current_val_loss={val_loss_avg:.6f}"
                )
            progress_bar.set_postfix(**record)

        stop_tensor = torch.tensor(int(should_stop), device=device)
        if distributed:
            torch.distributed.broadcast(stop_tensor, src=0)
        should_stop = bool(stop_tensor.item())
        accelerator.wait_for_everyone()
        if should_stop:
            break


if __name__ == "__main__":
    main()
