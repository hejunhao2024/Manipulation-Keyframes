#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

try:
    import deepspeed
except ImportError as exc:
    raise ImportError("deepspeed is required for this trainer") from exc

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from diffsynth.core import ModelConfig


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def read_manifest(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.lstrip().startswith("#")
        ]


def read_sample_id(sample_dir: str) -> str:
    sample_path = Path(sample_dir)
    ann_path = sample_path / "annotation.json"
    if not ann_path.exists():
        return sample_path.name
    try:
        ann = load_json(str(ann_path))
        return str(ann.get("id", sample_path.name))
    except Exception:
        return sample_path.name


def init_distributed() -> Tuple[bool, int, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if distributed and not torch.distributed.is_initialized():
        from datetime import timedelta
        torch.distributed.init_process_group(
            backend="nccl",
            timeout=timedelta(hours=2),
        )
    return distributed, rank, world_size, local_rank


def barrier(distributed: bool, local_rank: int) -> None:
    if not distributed:
        return
    if torch.cuda.is_available():
        torch.distributed.barrier(device_ids=[local_rank])
    else:
        torch.distributed.barrier()


def rank0_print(rank: int, *args, **kwargs) -> None:
    if rank == 0:
        print(*args, **kwargs, flush=True)


def seed_everything(seed: int, rank: int) -> None:
    seed = int(seed) + rank
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def make_epoch_order(num_samples: int, epoch: int, seed: int, shuffle: bool) -> List[int]:
    order = list(range(num_samples))
    if shuffle:
        random.Random(seed + epoch).shuffle(order)
    return order


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.alpha = int(alpha)
        self.scale = self.alpha / self.rank
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(
            torch.empty(
                self.rank,
                base.in_features,
                dtype=base.weight.dtype,
                device=base.weight.device,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                base.out_features,
                self.rank,
                dtype=base.weight.dtype,
                device=base.weight.device,
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        z = self.dropout(x).to(self.lora_A)
        z = z @ self.lora_A.t()
        z = z @ self.lora_B.t()
        return base_out + z.to(base_out) * self.scale


def get_parent_module(root: nn.Module, module_name: str):
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_lora(
    model: nn.Module,
    target_keywords: List[str],
    skip_keywords: List[str],
    rank: int,
    alpha: int,
    dropout: float,
) -> List[str]:
    replaced: List[str] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(k in name for k in target_keywords):
            continue
        if any(k in name for k in skip_keywords):
            continue
        parent, child_name = get_parent_module(model, name)
        setattr(parent, child_name, LoRALinear(module, rank, alpha, dropout))
        replaced.append(name)
    return replaced


def configure_trainable(model: nn.Module, train_cfg: Dict[str, Any], rank: int):
    for p in model.parameters():
        p.requires_grad_(False)

    mode = train_cfg["train_mode"]
    full_keywords = list(train_cfg.get("full_train_keywords", []))

    if mode == "lora_plus_full_keywords":
        lora_cfg = train_cfg["lora"]
        replaced = inject_lora(
            model=model,
            target_keywords=list(lora_cfg["target_keywords"]),
            skip_keywords=list(lora_cfg.get("skip_keywords", [])),
            rank=int(lora_cfg["rank"]),
            alpha=int(lora_cfg["alpha"]),
            dropout=float(lora_cfg["dropout"]),
        )
        rank0_print(rank, f"[LoRA] injected={len(replaced)}")
    elif mode == "full_keywords_only":
        replaced = []
    else:
        raise ValueError(f"Unsupported train_mode: {mode}")

    groups = {"lora": [], "cross_attn": [], "full_extra": []}
    named_trainable = []

    for name, p in model.named_parameters():
        if any(k in name for k in full_keywords):
            p.requires_grad_(True)
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad_(True)

        if not p.requires_grad:
            continue
        named_trainable.append((name, p.numel()))
        if "lora_A" in name or "lora_B" in name:
            groups["lora"].append(p)
        elif "cross_attn" in name:
            groups["cross_attn"].append(p)
        else:
            groups["full_extra"].append(p)

    if not named_trainable:
        raise RuntimeError("No trainable parameters selected.")

    rank0_print(
        rank,
        "[trainable] total="
        f"{sum(n for _, n in named_trainable) / 1e6:.3f}M "
        f"tensors={len(named_trainable)}",
    )
    return groups, named_trainable


def build_optimizer(groups: Dict[str, List[nn.Parameter]], train_cfg: Dict[str, Any]):
    optim_groups = []
    lr_map = {
        "lora": float(train_cfg["lora_lr"]),
        "cross_attn": float(train_cfg["cross_attn_lr"]),
        "full_extra": float(train_cfg["full_extra_lr"]),
    }
    for name in ("lora", "cross_attn", "full_extra"):
        if groups[name]:
            optim_groups.append(
                {"params": groups[name], "lr": lr_map[name], "name": name}
            )

    optimizer = torch.optim.AdamW(
        optim_groups,
        weight_decay=float(train_cfg["weight_decay"]),
        betas=tuple(train_cfg["betas"]),
        eps=float(train_cfg["eps"]),
        foreach=False,
        fused=False,
    )
    parameters = [p for group in optim_groups for p in group["params"]]
    return optimizer, parameters


def import_pipeline(conditioning_mode: str):
    if conditioning_mode == "local_only":
        from diffsynth.pipelines.keyframe_local_context import WanVideoPipeline
        return WanVideoPipeline
    if conditioning_mode == "dual_context":
        from diffsynth.pipelines.keyframe_dual_context import WanVideoPipeline
        return WanVideoPipeline
    raise ValueError(
        f"conditioning_mode must be local_only or dual_context, got {conditioning_mode}"
    )


def build_pipe(cfg: Dict[str, Any], conditioning_mode: str, local_rank: int):
    WanVideoPipeline = import_pipeline(conditioning_mode)
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    model_id = model_cfg["model_id"]
    dtype = torch.bfloat16
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    vram_config = {
        "offload_dtype": dtype,
        "offload_device": model_cfg["offload_device"],
        "onload_dtype": dtype,
        "onload_device": model_cfg["onload_device"],
        "computation_dtype": dtype,
        "computation_device": device,
    }

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=dtype,
        device=device,
        model_configs=[
            ModelConfig(
                model_id=model_id,
                origin_file_pattern="diffusion_pytorch_model*.safetensors",
                **vram_config,
            ),
        ],
        tokenizer_config=None,
        redirect_common_files=False,
        vram_limit=model_cfg.get("vram_limit_gb"),
        **(
            {
                "dual_context_global_scale_init": float(
                    train_cfg.get("dual_context", {}).get("global_scale_init", 1.0)
                ),
                "dual_context_local_scale_init": float(
                    train_cfg.get("dual_context", {}).get("local_scale_init", 1.0)
                ),
                "dual_context_max_context_scale": float(
                    train_cfg.get("dual_context", {}).get("max_context_scale", 2.0)
                ),
            }
            if conditioning_mode == "dual_context"
            else {}
        ),
    )
    return pipe


def as_5d(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.ndim == 4:
        return x.unsqueeze(0)
    if x.ndim == 5:
        return x
    raise ValueError(f"{name} must be 4D/5D, got {tuple(x.shape)}")


def as_local_context(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        return x.unsqueeze(0)
    if x.ndim == 4:
        return x
    if x.ndim == 5 and x.shape[1] == 1:
        return x[:, 0]
    raise ValueError(f"local context must be 3D/4D, got {tuple(x.shape)}")


def as_global_context(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x.unsqueeze(0)
    if x.ndim == 3:
        return x
    raise ValueError(f"global context must be 2D/3D, got {tuple(x.shape)}")


def move(x, device, dtype=torch.bfloat16):
    if x is None:
        return None
    return x.to(device=device, dtype=dtype, non_blocking=True)


def load_cached_item(
    sample_dir: str,
    data_cfg: Dict[str, Any],
    conditioning_mode: str,
    device,
    expected_slots: int,
) -> Dict[str, Any]:
    sample_id = read_sample_id(sample_dir)
    text_path = Path(data_cfg["text_cache_dir"]) / f"{sample_id}.pt"
    vae_path = Path(data_cfg["vae_cache_dir"]) / f"{sample_id}.pt"

    if not text_path.exists():
        raise FileNotFoundError(text_path)
    if not vae_path.exists():
        raise FileNotFoundError(vae_path)

    text_obj = torch.load(text_path, map_location="cpu", weights_only=False)
    vae_obj = torch.load(vae_path, map_location="cpu", weights_only=False)

    num_slots = int(vae_obj.get("num_slots", text_obj.get("num_slots", 0)))
    num_frames = int(vae_obj.get("num_frames", text_obj.get("num_frames", 0)))
    if num_slots != expected_slots:
        raise ValueError(
            f"{sample_id}: expected {expected_slots} slots, cache has {num_slots}"
        )
    expected_frames = 1 + (expected_slots - 1) * 4
    if num_frames != expected_frames:
        raise ValueError(
            f"{sample_id}: expected num_frames={expected_frames}, got {num_frames}"
        )

    target_latents = move(
        as_5d(vae_obj["target_latents"], "target_latents"), device
    )
    y = move(as_5d(vae_obj["y"], "y"), device)
    first_frame_latents = vae_obj.get("first_frame_latents")
    if first_frame_latents is not None:
        first_frame_latents = move(
            as_5d(first_frame_latents, "first_frame_latents"), device
        )
    clip_feature = move(text_obj.get("clip_feature"), device)

    if conditioning_mode == "local_only":
        context = move(as_local_context(text_obj["context"]), device)
        conditions = {"context": context}
    else:
        global_context = move(
            as_global_context(text_obj["global_context"]), device
        )
        local_context = move(
            as_local_context(text_obj["local_context"]), device
        )
        conditions = {
            "global_context": global_context,
            "local_context": local_context,
        }

    return {
        "sample_id": sample_id,
        "conditions": conditions,
        "clip_feature": clip_feature,
        "target_latents": target_latents,
        "y": y,
        "first_frame_latents": first_frame_latents,
        "fuse_vae_embedding_in_latents": bool(
            vae_obj.get("fuse_vae_embedding_in_latents", False)
        ),
        "text_path": str(text_path),
        "vae_path": str(vae_path),
    }


def collect_trainable_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: p.detach().cpu()
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def load_trainable_weights(model: nn.Module, path: str, rank: int) -> int:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    state = obj.get("trainable_state_dict", obj)
    current = model.state_dict()
    compatible = {
        k: v.to(dtype=current[k].dtype)
        for k, v in state.items()
        if k in current and current[k].shape == v.shape
    }
    model.load_state_dict(compatible, strict=False)
    rank0_print(rank, f"[resume weights] loaded={len(compatible)} from {path}")
    return int(obj.get("global_step", obj.get("step", 0)))


def save_lightweight_checkpoint(
    model: nn.Module,
    path: str,
    global_step: int,
    loss: float,
    cfg: Dict[str, Any],
    data_state: Dict[str, Any],
) -> None:
    ensure_dir(str(Path(path).parent))
    tmp = path + ".tmp"
    torch.save(
        {
            "global_step": global_step,
            "loss": float(loss),
            "trainable_state_dict": collect_trainable_state(model),
            "data_state": data_state,
            "config": cfg,
        },
        tmp,
    )
    os.replace(tmp, path)


def reduce_mean(x: torch.Tensor, distributed: bool) -> torch.Tensor:
    out = x.detach().clone()
    if distributed:
        torch.distributed.all_reduce(out, op=torch.distributed.ReduceOp.SUM)
        out /= torch.distributed.get_world_size()
    return out


def validate_config(cfg: Dict[str, Any], world_size: int) -> None:
    required = ["model", "data", "train", "deepspeed"]
    for key in required:
        if key not in cfg:
            raise KeyError(f"Missing config section: {key}")

    mode = cfg["train"]["conditioning_mode"]
    if mode not in {"local_only", "dual_context"}:
        raise ValueError(f"Invalid conditioning_mode: {mode}")

    ds = cfg["deepspeed"]
    micro = int(ds["train_micro_batch_size_per_gpu"])
    accum = int(ds["gradient_accumulation_steps"])
    expected = micro * accum * world_size
    if int(ds["train_batch_size"]) != expected:
        raise ValueError(
            f"deepspeed.train_batch_size={ds['train_batch_size']} but "
            f"world_size*micro*accum={expected}"
        )
    if micro != 1:
        raise ValueError(
            "This cache-streaming trainer currently requires "
            "train_micro_batch_size_per_gpu=1."
        )


def train(cfg: Dict[str, Any]) -> None:
    distributed, rank, world_size, local_rank = init_distributed()
    validate_config(cfg, world_size)

    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    mode = train_cfg["conditioning_mode"]
    seed_everything(int(train_cfg["seed"]), rank)

    out_dir = Path(train_cfg["out_dir"])
    ckpt_root = out_dir / "checkpoints"
    ds_ckpt_dir = ckpt_root / "deepspeed"
    tb_dir = out_dir / "tb"

    if rank == 0:
        ensure_dir(str(out_dir))
        ensure_dir(str(ckpt_root))
        ensure_dir(str(ds_ckpt_dir))
        ensure_dir(str(tb_dir))
        with open(out_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    barrier(distributed, local_rank)

    train_samples = read_manifest(data_cfg["train_manifest"])
    if not train_samples:
        raise ValueError("Empty training manifest.")

    rank0_print(
        rank,
        f"[data] samples={len(train_samples)} mode={mode} "
        f"expected_slots={data_cfg['expected_num_slots']}",
    )

    pipe = build_pipe(cfg, mode, local_rank)
    pipe.load_models_to_device(["dit"])
    pipe.dit.train()

    param_groups, _ = configure_trainable(pipe.dit, train_cfg, rank)
    optimizer, parameters = build_optimizer(param_groups, train_cfg)

    engine, optimizer, _, _ = deepspeed.initialize(
        model=pipe.dit,
        optimizer=optimizer,
        model_parameters=parameters,
        config=cfg["deepspeed"],
    )
    pipe.dit = engine.module
    pipe.dit.train()

    resume_cfg = train_cfg["resume"]
    global_step = 0
    micro_step = 0

    if resume_cfg["mode"] == "full":
        load_path, client_state = engine.load_checkpoint(
            resume_cfg["path"],
            tag=resume_cfg.get("tag"),
            load_module_strict=False,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
        )
        if load_path is None:
            raise RuntimeError(
                f"Failed to load DeepSpeed checkpoint from {resume_cfg['path']}"
            )
        client_state = client_state or {}
        global_step = int(client_state.get("global_step", 0))
        micro_step = int(client_state.get("micro_step", 0))
        restore_rng_state(client_state.get("rng_state"))
        rank0_print(
            rank,
            f"[resume full] path={load_path} global_step={global_step} "
            f"micro_step={micro_step}",
        )
    elif resume_cfg["mode"] == "weights":
        global_step = load_trainable_weights(
            engine.module, resume_cfg["path"], rank
        )
        if not bool(resume_cfg.get("keep_step", False)):
            global_step = 0
        micro_step = global_step * int(
            cfg["deepspeed"]["gradient_accumulation_steps"]
        )
    elif resume_cfg["mode"] != "none":
        raise ValueError(f"Unknown resume mode: {resume_cfg['mode']}")

    writer = SummaryWriter(str(tb_dir)) if rank == 0 else None

    max_steps = int(train_cfg["max_steps"])
    accum_steps = int(cfg["deepspeed"]["gradient_accumulation_steps"])
    shuffle = bool(data_cfg["shuffle"])
    shuffle_seed = int(data_cfg["shuffle_seed"])
    min_t = float(train_cfg["min_timestep"])
    max_t = float(train_cfg["max_timestep"])
    log_every = int(train_cfg["log_every"])
    save_every = int(train_cfg["save_every"])
    empty_cache_every = int(train_cfg["empty_cache_every"])
    expected_slots = int(data_cfg["expected_num_slots"])

    rank0_print(
        rank,
        f"[train] global_step={global_step} -> {max_steps} "
        f"world_size={world_size} accum={accum_steps}",
    )

    last_loss_value = float("nan")

    while global_step < max_steps:
        # Deterministic distributed stream. Restoring micro_step exactly restores
        # epoch, rank-local sample position, and next sample.
        global_item_index = micro_step * world_size + rank
        epoch = global_item_index // len(train_samples)
        position = global_item_index % len(train_samples)
        order = make_epoch_order(
            len(train_samples), epoch, shuffle_seed, shuffle
        )
        sample_dir = train_samples[order[position]]

        item = load_cached_item(
            sample_dir=sample_dir,
            data_cfg=data_cfg,
            conditioning_mode=mode,
            device=pipe.device,
            expected_slots=expected_slots,
        )

        pipe.load_models_to_device(["dit"])
        engine.train()

        target = item["target_latents"]
        batch_size = target.shape[0]
        timestep = torch.empty(
            batch_size, device=pipe.device, dtype=torch.float32
        ).uniform_(min_t, max_t)
        sigma = (timestep / 1000.0).view(batch_size, 1, 1, 1, 1).to(target)
        noise = torch.randn_like(target)
        noisy = (1.0 - sigma) * target + sigma * noise
        velocity = noise - target
        loss_mask = torch.ones_like(velocity)

        if (
            item["fuse_vae_embedding_in_latents"]
            and item["first_frame_latents"] is not None
        ):
            noisy[:, :, :1] = item["first_frame_latents"]
            loss_mask[:, :, :1] = 0

        model_kwargs = dict(
            dit=engine.module,
            latents=noisy,
            timestep=timestep,
            clip_feature=item["clip_feature"],
            y=item["y"],
            fuse_vae_embedding_in_latents=item[
                "fuse_vae_embedding_in_latents"
            ],
            use_gradient_checkpointing=bool(
                train_cfg["use_gradient_checkpointing"]
            ),
            use_gradient_checkpointing_offload=bool(
                train_cfg["use_gradient_checkpointing_offload"]
            ),
        )
        model_kwargs.update(item["conditions"])

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=torch.cuda.is_available(),
        ):
            pred = pipe.model_fn(**model_kwargs)
            loss = (
                (pred.float() - velocity.float()).pow(2)
                * loss_mask.float()
            ).mean()

        engine.backward(loss)
        accumulation_boundary = engine.is_gradient_accumulation_boundary()
        engine.step()
        micro_step += 1

        if accumulation_boundary:
            global_step += 1
            avg_loss = reduce_mean(loss, distributed)
            last_loss_value = float(avg_loss.item())

            if rank == 0 and global_step % log_every == 0:
                print(
                    f"[step {global_step:06d}] epoch={epoch} "
                    f"position={position} loss={last_loss_value:.6f} "
                    f"sample={item['sample_id']}",
                    flush=True,
                )
                writer.add_scalar("train/loss", last_loss_value, global_step)
                writer.add_scalar("train/epoch", epoch, global_step)
                for group in optimizer.param_groups:
                    writer.add_scalar(
                        f"train/lr/{group.get('name', 'group')}",
                        group["lr"],
                        global_step,
                    )

            if global_step % save_every == 0 or global_step == max_steps:
                data_state = {
                    "global_step": global_step,
                    "micro_step": micro_step,
                    "epoch": epoch,
                    "position": position,
                    "next_global_item_index": micro_step * world_size,
                }
                client_state = {
                    **data_state,
                    "rng_state": capture_rng_state(),
                    "conditioning_mode": mode,
                }
                tag = f"step_{global_step:06d}"
                engine.save_checkpoint(
                    str(ds_ckpt_dir),
                    tag=tag,
                    client_state=client_state,
                )
                if rank == 0:
                    lightweight = ckpt_root / f"trainable_{tag}.pt"
                    save_lightweight_checkpoint(
                        engine.module,
                        str(lightweight),
                        global_step,
                        last_loss_value,
                        cfg,
                        data_state,
                    )
                    print(f"[checkpoint] {tag}", flush=True)

        del (
            item,
            target,
            timestep,
            sigma,
            noise,
            noisy,
            velocity,
            loss_mask,
            pred,
            loss,
        )
        if (
            torch.cuda.is_available()
            and empty_cache_every > 0
            and micro_step % empty_cache_every == 0
        ):
            torch.cuda.empty_cache()

    if writer is not None:
        writer.close()
    barrier(distributed, local_rank)
    rank0_print(rank, f"[done] {out_dir}")
    if distributed and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--local-rank", type=int, default=0)
    args = parser.parse_args()
    train(load_json(args.config))


if __name__ == "__main__":
    main()
