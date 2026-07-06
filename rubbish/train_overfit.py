import os
import sys
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from diffsynth.pipelines.keyframe import WanVideoPipeline
from diffsynth.core import ModelConfig


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


# -------------------------
# basic utils
# -------------------------

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def list_images_in_dir(path: str) -> List[str]:
    files = []
    for name in sorted(os.listdir(path)):
        if name.lower().endswith(IMAGE_EXTS):
            files.append(os.path.join(path, name))
    return files


def load_image(path: str, width: int, height: int) -> Image.Image:
    return Image.open(path).convert("RGB").resize((width, height))


def build_video_from_keyframes(keyframes: List[Image.Image], num_frames: int) -> List[Image.Image]:
    """
    keyframes length = F
    num_frames = 1 + (F - 1) * 4

    video[0]  -> keyframe[0]
    video[4]  -> keyframe[1]
    video[8]  -> keyframe[2]

    中间帧第一版直接重复填充，先用于 overfit 验证。
    """
    frames = []
    f = len(keyframes)
    for t in range(num_frames):
        idx = min(t // 4, f - 1)
        frames.append(keyframes[idx])
    return frames


def resolve_path(base_dir: Path, p: str) -> str:
    p = Path(p)
    if p.is_absolute():
        return str(p)
    return str(base_dir / p)


def load_sample_from_dir(sample_dir: str) -> Dict:
    """
    读取新数据格式：

    sample_xxxxxx/
      images/
        000.jpg
        ...
      annotation.json

    annotation.json:
      {
        "id": "sample_000001",
        "image": "images/000.jpg",
        "keyframes": ["images/000.jpg", ...],
        "prompt": "...",
        "frame_prompts": [...]
      }
    """
    sample_dir = Path(sample_dir)
    ann_path = sample_dir / "annotation.json"
    ann = load_json(str(ann_path))

    keyframes = [resolve_path(sample_dir, p) for p in ann["keyframes"]]

    if len(keyframes) != len(ann["frame_prompts"]):
        raise ValueError(
            f"{sample_dir}: len(keyframes)={len(keyframes)} != "
            f"len(frame_prompts)={len(ann['frame_prompts'])}"
        )

    return {
        "id": ann.get("id", sample_dir.name),
        "sample_dir": str(sample_dir),
        "image": resolve_path(sample_dir, ann["image"]),
        "prompt": ann["prompt"],
        "negative_prompt": ann.get("negative_prompt", ""),
        "frame_prompts": ann["frame_prompts"],
        "target_keyframes": keyframes,
    }


def load_train_samples(cfg: Dict) -> List[Dict]:
    """
    兼容两种配置：
    1. 新版多样本：
       "data": {"sample_dirs": [...]}

    2. 旧版单样本：
       "sample": {...}
    """
    if "data" in cfg and "sample_dirs" in cfg["data"]:
        samples = [load_sample_from_dir(d) for d in cfg["data"]["sample_dirs"]]
    elif "sample" in cfg:
        samples = [cfg["sample"]]
    else:
        raise KeyError('Config must contain either "data.sample_dirs" or "sample".')

    if len(samples) == 0:
        raise ValueError("No training samples found.")

    return samples


# -------------------------
# LoRA
# -------------------------

class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int = 16,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        for p in self.base.parameters():
            p.requires_grad_(False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 保持 LoRA 参数 fp32，训练更稳；forward 里再转 dtype。
        # 注意：device 必须跟 base linear 一致，否则会出现 CPU/CUDA mismatch。
        device = base.weight.device
        self.lora_A = nn.Parameter(
            torch.empty(rank, base.in_features, dtype=torch.float32, device=device)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_features, rank, dtype=torch.float32, device=device)
        )

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        base_out = self.base(x)

        x_lora = self.dropout(x).to(dtype=self.lora_A.dtype, device=self.lora_A.device)
        lora_out = x_lora @ self.lora_A.t()
        lora_out = lora_out @ self.lora_B.t()
        lora_out = lora_out.to(dtype=base_out.dtype, device=base_out.device)

        return base_out + lora_out * self.scale


def get_parent_module(root: nn.Module, module_name: str):
    parts = module_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def inject_lora_to_dit(
    dit: nn.Module,
    target_keywords: List[str],
    rank: int = 16,
    alpha: int = 16,
    dropout: float = 0.05,
    skip_keywords: Optional[List[str]] = None,
):
    if skip_keywords is None:
        skip_keywords = ["cross_attn"]

    replaced = []

    for name, module in list(dit.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        if not any(k in name for k in target_keywords):
            continue

        if any(k in name for k in skip_keywords):
            continue

        parent, child_name = get_parent_module(dit, name)
        setattr(
            parent,
            child_name,
            LoRALinear(
                module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            ),
        )
        replaced.append(name)

    print(f"[LoRA] injected into {len(replaced)} Linear layers")
    for name in replaced[:80]:
        print("  ", name)
    if len(replaced) > 80:
        print(f"  ... and {len(replaced) - 80} more")

    return replaced


def set_trainable_params(pipe: WanVideoPipeline, train_cfg: Dict):
    """
    推荐第一版：
      - self_attn / ffn 上 LoRA
      - cross_attn 全参数训练
      - text_embedding / img_emb / norm3 可选全参数训练

    config:
      train_mode = "lora_plus_full_keywords"
    """
    dit = pipe.dit

    for p in dit.parameters():
        p.requires_grad_(False)

    train_mode = train_cfg.get("train_mode", "lora_plus_full_keywords")

    if train_mode == "cross_attn_only":
        full_train_keywords = train_cfg.get("full_train_keywords", ["cross_attn"])

    elif train_mode == "lora_plus_full_keywords":
        lora_cfg = train_cfg.get("lora", {})
        inject_lora_to_dit(
            dit,
            target_keywords=lora_cfg.get("target_keywords", ["self_attn", "ffn"]),
            rank=lora_cfg.get("rank", 16),
            alpha=lora_cfg.get("alpha", 16),
            dropout=lora_cfg.get("dropout", 0.05),
            skip_keywords=lora_cfg.get("skip_keywords", ["cross_attn"]),
        )
        full_train_keywords = train_cfg.get(
            "full_train_keywords",
            ["cross_attn", "norm3", "text_embedding", "img_emb"],
        )

    else:
        raise ValueError(f"Unknown train_mode: {train_mode}")

    for name, p in dit.named_parameters():
        if any(k in name for k in full_train_keywords):
            p.requires_grad_(True)

        if "lora_A" in name or "lora_B" in name:
            p.requires_grad_(True)

    groups = {
        "lora": [],
        "cross_attn": [],
        "full_extra": [],
    }

    total = 0
    print("\n[Trainable parameters]")
    for name, p in dit.named_parameters():
        if not p.requires_grad:
            continue

        n = p.numel()
        total += n

        if "lora_A" in name or "lora_B" in name:
            groups["lora"].append(p)
            tag = "lora"
        elif "cross_attn" in name:
            groups["cross_attn"].append(p)
            tag = "cross_attn"
        else:
            groups["full_extra"].append(p)
            tag = "full_extra"

        print(f"  [{tag}] {name}: {n / 1e6:.3f}M")

    print(f"Total trainable params: {total / 1e6:.3f}M\n")

    return groups


def build_optimizer(param_groups: Dict[str, List[nn.Parameter]], train_cfg: Dict):
    optim_groups = []

    if len(param_groups["lora"]) > 0:
        optim_groups.append({
            "params": param_groups["lora"],
            "lr": train_cfg.get("lora_lr", train_cfg.get("lr", 1e-4)),
            "name": "lora",
        })

    if len(param_groups["cross_attn"]) > 0:
        optim_groups.append({
            "params": param_groups["cross_attn"],
            "lr": train_cfg.get("cross_attn_lr", 2e-5),
            "name": "cross_attn",
        })

    if len(param_groups["full_extra"]) > 0:
        optim_groups.append({
            "params": param_groups["full_extra"],
            "lr": train_cfg.get("full_extra_lr", 2e-5),
            "name": "full_extra",
        })

    optimizer = torch.optim.AdamW(
        optim_groups,
        weight_decay=train_cfg.get("weight_decay", 0.0),
        betas=tuple(train_cfg.get("betas", [0.9, 0.999])),
        eps=train_cfg.get("eps", 1e-8),
    )

    params_for_clip = []
    for group in optim_groups:
        params_for_clip.extend(group["params"])

    return optimizer, params_for_clip


# -------------------------
# DiffSynth pipeline
# -------------------------

def build_pipe(cfg: Dict):
    model_cfg = cfg["model"]

    model_id = model_cfg.get("model_id", "Wan-AI/Wan2.1-I2V-14B-480P")
    tokenizer_path = model_cfg["tokenizer_path"]

    dtype = torch.bfloat16
    device = model_cfg.get("device", "cuda")

    vram_config = {
        "offload_dtype": dtype,
        "offload_device": model_cfg.get("offload_device", "cpu"),
        "onload_dtype": dtype,
        "onload_device": model_cfg.get("onload_device", "cpu"),
        "computation_dtype": dtype,
        "computation_device": model_cfg.get("computation_device", "cuda"),
    }

    vram_limit = model_cfg.get("vram_limit_gb", None)
    if vram_limit is None and device == "cuda":
        try:
            vram_limit = torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2
        except Exception:
            vram_limit = None

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=dtype,
        device=device,
        model_configs=[
            ModelConfig(
                model_id=model_id,
                origin_file_pattern="diffusion_pytorch_model*.safetensors",
                **vram_config,
            ),
            ModelConfig(
                model_id=model_id,
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                **vram_config,
            ),
            ModelConfig(
                model_id=model_id,
                origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                **vram_config,
            ),
            ModelConfig(
                model_id=model_id,
                origin_file_pattern="Wan2.1_VAE.pth",
                **vram_config,
            ),
        ],
        tokenizer_config=ModelConfig(path=tokenizer_path),
        redirect_common_files=False,
        vram_limit=vram_limit,
    )

    return pipe


@torch.no_grad()
def prepare_conditioning(pipe: WanVideoPipeline, sample: Dict, infer_cfg: Dict):
    """
    复用 keyframe.py 里的 units，得到：
      context: [1, F, 512, text_dim]
      clip_feature
      y
      fuse_vae_embedding_in_latents
      first_frame_latents
    """
    prompt = sample["prompt"]
    frame_prompts = sample["frame_prompts"]
    input_image = Image.open(sample["image"]).convert("RGB")

    height = infer_cfg.get("height", 480)
    width = infer_cfg.get("width", 832)
    tiled = infer_cfg.get("tiled", True)
    tile_size = tuple(infer_cfg.get("tile_size", [30, 52]))
    tile_stride = tuple(infer_cfg.get("tile_stride", [15, 26]))

    num_slots = len(frame_prompts)
    num_frames = 1 + (num_slots - 1) * 4

    pipe.scheduler.set_timesteps(
        infer_cfg.get("num_inference_steps", 10),
        denoising_strength=1.0,
        shift=infer_cfg.get("sigma_shift", 5.0),
    )

    inputs_posi = {
        "prompt": prompt,
        "frame_prompts": frame_prompts,
        "num_slots": num_slots,

        "vap_prompt": " ",
        "tea_cache_l1_thresh": None,
        "tea_cache_model_id": "",
        "num_inference_steps": infer_cfg.get("num_inference_steps", 10),
    }

    inputs_nega = {
        "negative_prompt": sample.get("negative_prompt", ""),
        "frame_prompts": None,
        "num_slots": num_slots,

        "negative_vap_prompt": " ",
        "tea_cache_l1_thresh": None,
        "tea_cache_model_id": "",
        "num_inference_steps": infer_cfg.get("num_inference_steps", 10),
    }

    inputs_shared = {
        "input_image": input_image,
        "end_image": None,
        "input_video": None,
        "denoising_strength": 1.0,

        "control_video": None,
        "reference_image": None,

        "camera_control_direction": None,
        "camera_control_speed": 1 / 54,
        "camera_control_origin": (
            0, 0.532139961, 0.946026558,
            0.5, 0.5,
            0, 0, 1,
            0, 0, 0,
            0, 1, 0,
            0, 0, 0,
            0, 1, 0,
        ),

        "vace_video": None,
        "vace_video_mask": None,
        "vace_reference_image": None,
        "vace_scale": 1.0,

        "seed": infer_cfg.get("seed", 42),
        "rand_device": infer_cfg.get("rand_device", "cpu"),

        "height": height,
        "width": width,
        "num_frames": num_frames,

        "cfg_scale": infer_cfg.get("cfg_scale", 5.0),
        "cfg_merge": False,
        "sigma_shift": infer_cfg.get("sigma_shift", 5.0),

        "motion_bucket_id": None,
        "longcat_video": None,

        "tiled": tiled,
        "tile_size": tile_size,
        "tile_stride": tile_stride,

        "sliding_window_size": None,
        "sliding_window_stride": None,

        "input_audio": None,
        "audio_sample_rate": 16000,
        "s2v_pose_video": None,
        "audio_embeds": None,
        "s2v_pose_latents": None,
        "motion_video": None,

        "animate_pose_video": None,
        "animate_face_video": None,
        "animate_inpaint_video": None,
        "animate_mask_video": None,

        "vap_video": None,

        "wantodance_music_path": None,
        "wantodance_reference_image": None,
        "wantodance_fps": 30,
        "wantodance_keyframes": None,
        "wantodance_keyframes_mask": None,

        "framewise_decoding": False,
    }

    for unit in pipe.units:
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(
            unit, pipe, inputs_shared, inputs_posi, inputs_nega
        )

    cond = {
        "context": inputs_posi["context"].detach(),
        "clip_feature": inputs_shared.get("clip_feature", None),
        "y": inputs_shared.get("y", None),
        "fuse_vae_embedding_in_latents": inputs_shared.get("fuse_vae_embedding_in_latents", False),
        "first_frame_latents": inputs_shared.get("first_frame_latents", None),
        "num_slots": num_slots,
        "num_frames": num_frames,
    }

    for k in ["clip_feature", "y", "first_frame_latents"]:
        if cond[k] is not None:
            cond[k] = cond[k].detach()

    print("[conditioning]")
    print("  context:", tuple(cond["context"].shape))
    if cond["clip_feature"] is not None:
        print("  clip_feature:", tuple(cond["clip_feature"].shape))
    if cond["y"] is not None:
        print("  y:", tuple(cond["y"].shape))
    if cond["first_frame_latents"] is not None:
        print("  first_frame_latents:", tuple(cond["first_frame_latents"].shape))
    print("  num_slots:", num_slots, "num_frames:", num_frames)

    return cond


@torch.no_grad()
def encode_target_latents(pipe: WanVideoPipeline, sample: Dict, infer_cfg: Dict, num_frames: int):
    height = infer_cfg.get("height", 480)
    width = infer_cfg.get("width", 832)
    tiled = infer_cfg.get("tiled", True)
    tile_size = tuple(infer_cfg.get("tile_size", [30, 52]))
    tile_stride = tuple(infer_cfg.get("tile_stride", [15, 26]))

    if "target_keyframes" in sample:
        keyframe_paths = sample["target_keyframes"]
    else:
        keyframe_paths = list_images_in_dir(sample["target_keyframes_dir"])

    target_keyframes = [
        load_image(p, width=width, height=height)
        for p in keyframe_paths
    ]

    expected_f = len(sample["frame_prompts"])
    if len(target_keyframes) != expected_f:
        raise ValueError(
            f"target keyframes number mismatch: got {len(target_keyframes)}, expected {expected_f}"
        )

    video_frames = build_video_from_keyframes(target_keyframes, num_frames)

    pipe.load_models_to_device(["vae"])
    video_tensor = pipe.preprocess_video(video_frames)

    target_latents = pipe.vae.encode(
        video_tensor,
        device=pipe.device,
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )
    target_latents = target_latents.to(dtype=pipe.torch_dtype, device=pipe.device)

    print("[target]")
    print("  target_keyframes:", len(target_keyframes))
    print("  video_frames:", len(video_frames))
    print("  target_latents:", tuple(target_latents.shape))

    return target_latents.detach()


# -------------------------
# checkpoint
# -------------------------

def collect_trainable_state_dict(pipe: WanVideoPipeline):
    state = {}
    for name, p in pipe.dit.named_parameters():
        if p.requires_grad:
            state[name] = p.detach().cpu()
    return state


def save_checkpoint(
    pipe: WanVideoPipeline,
    optimizer: torch.optim.Optimizer,
    path: str,
    step: int,
    loss: float,
    cfg: Dict,
):
    ensure_dir(os.path.dirname(path))
    torch.save(
        {
            "step": step,
            "loss": float(loss),
            "trainable_state_dict": collect_trainable_state_dict(pipe),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
        },
        path,
    )


# -------------------------
# training
# -------------------------

def train(cfg: Dict):
    train_cfg = cfg["train"]
    infer_cfg = cfg["infer"]

    out_dir = train_cfg["out_dir"]
    log_dir = os.path.join(out_dir, "tb")
    ckpt_dir = os.path.join(out_dir, "checkpoints")

    ensure_dir(out_dir)
    ensure_dir(log_dir)
    ensure_dir(ckpt_dir)

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    samples = load_train_samples(cfg)
    print(f"[dataset] num samples = {len(samples)}")
    for s in samples:
        print(f"  - {s.get('id', '<no_id>')}: {s['sample_dir'] if 'sample_dir' in s else s.get('image', '')}")

    pipe = build_pipe(cfg)

    # 先预处理 5 条样本：condition + target latents。
    # 5 条 overfit 存在内存里更方便，也避免每 step 重复跑 text/image/VAE encoder。
    train_items = []
    for sample in samples:
        print(f"\n[prepare sample] {sample.get('id', '<no_id>')}")
        cond = prepare_conditioning(pipe, sample, infer_cfg)
        target_latents = encode_target_latents(
            pipe,
            sample,
            infer_cfg,
            num_frames=cond["num_frames"],
        )
        train_items.append({
            "sample": sample,
            "cond": cond,
            "target_latents": target_latents,
        })

    pipe.load_models_to_device(["dit"])
    pipe.dit.train()

    param_groups = set_trainable_params(pipe, train_cfg)
    optimizer, params_for_clip = build_optimizer(param_groups, train_cfg)

    writer = SummaryWriter(log_dir=log_dir)

    steps = int(train_cfg.get("steps", 1000))
    log_every = int(train_cfg.get("log_every", 1))
    save_every = int(train_cfg.get("save_every", 100))
    grad_clip = train_cfg.get("grad_clip", 1.0)
    use_grad_ckpt = bool(train_cfg.get("use_gradient_checkpointing", True))

    min_t = float(train_cfg.get("min_timestep", 0.0))
    max_t = float(train_cfg.get("max_timestep", 1000.0))

    print("\n[start training]")
    print("  steps:", steps)
    print("  num_train_items:", len(train_items))
    print("  min_timestep:", min_t)
    print("  max_timestep:", max_t)

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)

        # 5 条数据轮流训练，最简单稳定。
        item = train_items[(step - 1) % len(train_items)]
        sample = item["sample"]
        sample_id = sample.get("id", f"sample_{(step - 1) % len(train_items)}")
        cond = item["cond"]
        target_latents = item["target_latents"]

        b = target_latents.shape[0]

        # Wan / Flow Matching:
        # x_t = (1 - sigma) * x0 + sigma * noise
        # target velocity = noise - x0
        timestep = torch.empty((b,), device=pipe.device, dtype=torch.float32).uniform_(min_t, max_t)
        sigma = (timestep / 1000.0).view(b, 1, 1, 1, 1).to(dtype=target_latents.dtype)

        noise = torch.randn_like(target_latents)
        noisy_latents = (1.0 - sigma) * target_latents + sigma * noise
        target_velocity = noise - target_latents

        loss_mask = torch.ones_like(target_velocity)

        # 如果是 TI2V fused first-frame latent，第一帧是条件，不训预测它。
        if cond["fuse_vae_embedding_in_latents"] and cond["first_frame_latents"] is not None:
            noisy_latents[:, :, 0:1] = cond["first_frame_latents"]
            loss_mask[:, :, 0:1] = 0

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = pipe.model_fn(
                dit=pipe.dit,
                latents=noisy_latents,
                timestep=timestep,
                context=cond["context"],
                clip_feature=cond["clip_feature"],
                y=cond["y"],
                fuse_vae_embedding_in_latents=cond["fuse_vae_embedding_in_latents"],
                use_gradient_checkpointing=use_grad_ckpt,
                use_gradient_checkpointing_offload=False,
            )

            loss = ((pred.float() - target_velocity.float()) ** 2 * loss_mask.float()).mean()

        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(params_for_clip, float(grad_clip))
        else:
            grad_norm = torch.tensor(0.0)

        optimizer.step()

        if step % log_every == 0:
            print(
                f"[step {step:06d}] sample={sample_id} "
                f"loss={loss.item():.6f} grad_norm={float(grad_norm):.4f}"
            )
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/grad_norm", float(grad_norm), step)
            writer.add_scalar(f"sample_loss/{sample_id}", loss.item(), step)

            for group in optimizer.param_groups:
                name = group.get("name", "group")
                writer.add_scalar(f"train/lr/{name}", group["lr"], step)

        if step % save_every == 0 or step == steps:
            ckpt_path = os.path.join(ckpt_dir, f"step_{step:06d}.pt")
            save_checkpoint(
                pipe=pipe,
                optimizer=optimizer,
                path=ckpt_path,
                step=step,
                loss=loss.item(),
                cfg=cfg,
            )
            print("saved checkpoint:", ckpt_path)

    writer.close()
    print("done:", out_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_json(args.config)
    train(cfg)


if __name__ == "__main__":
    main()