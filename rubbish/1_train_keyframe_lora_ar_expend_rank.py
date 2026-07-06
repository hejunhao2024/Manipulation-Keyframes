import os
import sys
sys.stdout.reconfigure(line_buffering=True)
import json
import math
import random
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

try:
    import deepspeed
except ImportError:
    deepspeed = None


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


def read_manifest(path: str) -> List[str]:
    """Read one sample directory per line."""
    paths = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            paths.append(line)
    return paths


def load_sample_entries(cfg: Dict, split: str = "train") -> List[Dict]:
    """
    大规模训练入口。

    返回轻量 entry，不在启动阶段读取/编码所有样本：
      - manifest: {"kind": "dir", "path": ".../sample_xxxxxx"}
      - sample_dirs: 同上
      - sample: {"kind": "sample", "sample": {...}}

    这样训练循环可以每一步再 load_sample_from_dir(entry["path"])，
    避免像 overfit 版本那样一次性 prepare 全部视频。
    """
    data_cfg = cfg.get("data", {})
    manifest_key = f"{split}_manifest"

    if manifest_key in data_cfg:
        entries = [
            {"kind": "dir", "path": p}
            for p in read_manifest(data_cfg[manifest_key])
        ]

    elif split == "train" and "sample_dirs" in data_cfg:
        entries = [
            {"kind": "dir", "path": p}
            for p in data_cfg["sample_dirs"]
        ]

    elif split == "train" and "sample" in cfg:
        entries = [{"kind": "sample", "sample": cfg["sample"]}]

    else:
        raise KeyError(
            f'Config must contain "data.{manifest_key}", '
            f'or "data.sample_dirs", or "sample".'
        )

    max_samples_key = f"max_{split}_samples"
    if max_samples_key in data_cfg and data_cfg[max_samples_key] is not None:
        entries = entries[: int(data_cfg[max_samples_key])]

    if len(entries) == 0:
        raise ValueError(f"No {split} samples found.")

    return entries


def materialize_sample(entry: Dict) -> Dict:
    """Turn a lightweight entry into the real sample dict used by the original code."""
    if entry["kind"] == "dir":
        return load_sample_from_dir(entry["path"])
    if entry["kind"] == "sample":
        return entry["sample"]
    raise ValueError(f"Unknown sample entry kind: {entry.get('kind')}")


def entry_name(entry: Dict) -> str:
    if entry["kind"] == "dir":
        return Path(entry["path"]).name
    sample = entry.get("sample", {})
    return sample.get("id", sample.get("image", "sample"))


def make_epoch_order(num_samples: int, epoch: int, seed: int, shuffle: bool = True) -> List[int]:
    order = list(range(num_samples))
    if shuffle:
        random.Random(seed + epoch).shuffle(order)
    return order


# Backward-compatible helper for old overfit usage.
# Not used by the new large-scale train() below, because that function uses entries lazily.
def load_train_samples(cfg: Dict) -> List[Dict]:
    entries = load_sample_entries(cfg, split="train")
    return [materialize_sample(e) for e in entries]


def init_distributed():
    """Initialize torch.distributed when launched by torchrun.

    Important: bind each process to its local GPU before any NCCL collective.
    Otherwise torch may run an early barrier on an unknown device and hang.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if not torch.distributed.is_initialized():
            from datetime import timedelta
            torch.distributed.init_process_group(
                backend="nccl",
                timeout=timedelta(hours=2),
            )
        return True, rank, world_size, local_rank

    return False, 0, 1, local_rank


def is_rank0(rank: int) -> bool:
    return rank == 0


def barrier(distributed: bool, local_rank: int = 0):
    if distributed and torch.distributed.is_initialized():
        # Explicit device_ids avoids NCCL warning/hang when device mapping is unknown.
        if torch.cuda.is_available():
            torch.distributed.barrier(device_ids=[local_rank])
        else:
            torch.distributed.barrier()


def rank_print(rank: int, *args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)


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

        # Low-memory LoRA:
        # Keep LoRA weights on the same device and dtype as the wrapped Linear.
        # The previous fp32 LoRA forward created very large fp32 activations
        # during gradient-checkpoint recomputation and caused OOM.
        device = base.weight.device
        dtype = base.weight.dtype
        self.lora_A = nn.Parameter(
            torch.empty(rank, base.in_features, dtype=dtype, device=device)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_features, rank, dtype=dtype, device=device)
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
        foreach=False,
        fused=False,
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
def prepare_conditioning(pipe: WanVideoPipeline, sample: Dict, infer_cfg: Dict, verbose: bool = True):
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

    if verbose:
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
def encode_target_latents(pipe: WanVideoPipeline, sample: Dict, infer_cfg: Dict, num_frames: int, verbose: bool = True):
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

    if verbose:
        print("[target]")
        print("  target_keyframes:", len(target_keyframes))
        print("  video_frames:", len(video_frames))
        print("  target_latents:", tuple(target_latents.shape))

    return target_latents.detach()




# -------------------------
# cached conditioning / latents
# -------------------------

def sample_id_from_entry(entry: Dict) -> str:
    """Get the sample id used by encode_text.py / encode_vae.py cache files."""
    if entry["kind"] == "dir":
        sample_dir = Path(entry["path"])
        ann_path = sample_dir / "annotation.json"
        if ann_path.exists():
            try:
                ann = load_json(str(ann_path))
                return ann.get("id", sample_dir.name)
            except Exception:
                return sample_dir.name
        return sample_dir.name
    sample = entry.get("sample", {})
    return sample.get("id", entry_name(entry))


def as_batched_5d(x: torch.Tensor, name: str) -> torch.Tensor:
    """Accept [C,T,H,W] or [1,C,T,H,W], return [1,C,T,H,W]."""
    if x is None:
        return None
    if x.ndim == 4:
        return x.unsqueeze(0)
    if x.ndim == 5:
        return x
    raise ValueError(f"{name} must be 4D or 5D, got shape={tuple(x.shape)}")


def as_batched_context(x: torch.Tensor) -> torch.Tensor:
    """Accept [F,512,D] or [1,F,512,D], return [1,F,512,D]."""
    if x.ndim == 3:
        return x.unsqueeze(0)
    if x.ndim == 4:
        return x
    if x.ndim == 5 and x.shape[1] == 1:
        return x[:, 0]
    raise ValueError(f"context must be 3D/4D, got shape={tuple(x.shape)}")


def move_tensor(x, device, dtype=None):
    if x is None:
        return None
    if not torch.is_tensor(x):
        return x
    if dtype is None:
        return x.to(device=device, non_blocking=True)
    return x.to(device=device, dtype=dtype, non_blocking=True)


def load_cached_training_item(entry: Dict, data_cfg: Dict, device, dtype=torch.bfloat16) -> Dict:
    """
    Load one sample from split cache:
      text_cache_dir/sample_xxxxxx.pt: context, clip_feature
      vae_cache_dir/sample_xxxxxx.pt: target_latents, y, first_frame_latents
    """
    text_cache_dir = data_cfg["text_cache_dir"]
    vae_cache_dir = data_cfg["vae_cache_dir"]
    sample_id = sample_id_from_entry(entry)

    text_path = os.path.join(text_cache_dir, f"{sample_id}.pt")
    vae_path = os.path.join(vae_cache_dir, f"{sample_id}.pt")

    if not os.path.exists(text_path):
        raise FileNotFoundError(f"text cache not found: {text_path}")
    if not os.path.exists(vae_path):
        raise FileNotFoundError(f"vae cache not found: {vae_path}")

    text_obj = torch.load(text_path, map_location="cpu")
    vae_obj = torch.load(vae_path, map_location="cpu")

    context = as_batched_context(text_obj["context"])
    clip_feature = text_obj.get("clip_feature", None)

    target_latents = as_batched_5d(vae_obj["target_latents"], "target_latents")
    y = vae_obj.get("y", None)
    if y is None:
        raise KeyError(f"vae cache has no y: {vae_path}")
    y = as_batched_5d(y, "y")

    first_frame_latents = vae_obj.get("first_frame_latents", None)
    if first_frame_latents is not None:
        first_frame_latents = as_batched_5d(first_frame_latents, "first_frame_latents")

    num_slots = int(vae_obj.get("num_slots", text_obj.get("num_slots", 0)))
    num_frames = int(vae_obj.get("num_frames", text_obj.get("num_frames", 0)))

    if int(text_obj.get("num_slots", num_slots)) != num_slots:
        raise ValueError(f"num_slots mismatch for {sample_id}")
    if int(text_obj.get("num_frames", num_frames)) != num_frames:
        raise ValueError(f"num_frames mismatch for {sample_id}")

    cond = {
        "context": move_tensor(context, device=device, dtype=dtype),
        "clip_feature": move_tensor(clip_feature, device=device, dtype=dtype),
        "y": move_tensor(y, device=device, dtype=dtype),
        "fuse_vae_embedding_in_latents": bool(vae_obj.get("fuse_vae_embedding_in_latents", False)),
        "first_frame_latents": move_tensor(first_frame_latents, device=device, dtype=dtype),
        "num_slots": num_slots,
        "num_frames": num_frames,
    }

    target_latents = move_tensor(target_latents, device=device, dtype=dtype)

    return {
        "sample_id": sample_id,
        "cond": cond,
        "target_latents": target_latents,
        "text_path": text_path,
        "vae_path": vae_path,
    }


def load_trainable_checkpoint_into_dit(
    pipe: WanVideoPipeline,
    path: str,
    rank: int = 0,
    strict_resume: bool = True,
    allow_lora_rank_expansion: bool = False,
    expected_source_lora_rank: Optional[int] = None,
    source_lora_alpha: Optional[float] = None,
    target_lora_alpha: Optional[float] = None,
):
    """Load an old trainable checkpoint, optionally expanding LoRA rank losslessly.

    For a source rank ``r_old`` and target rank ``r_new >= r_old``:
      A_new[:r_old] = A_old
      B_new[:, :r_old] = B_old
      B_new[:, r_old:] remains zero

    Therefore the initial LoRA update is unchanged when alpha/r is also unchanged.
    Non-LoRA shape mismatches always remain errors in strict mode.
    """
    if path is None or str(path).strip() == "":
        return

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("trainable_state_dict", ckpt)
    current = pipe.dit.state_dict()

    compatible = {}
    mismatches = []
    expanded = []

    if allow_lora_rank_expansion and source_lora_alpha is not None and target_lora_alpha is not None:
        old_rank = float(expected_source_lora_rank) if expected_source_lora_rank else None
        if old_rank is not None:
            old_scale = float(source_lora_alpha) / old_rank
            target_rank = None
            for key, value in current.items():
                if key.endswith("lora_A"):
                    target_rank = float(value.shape[0])
                    break
            if target_rank is not None:
                new_scale = float(target_lora_alpha) / target_rank
                if abs(old_scale - new_scale) > 1e-12:
                    raise ValueError(
                        f"Lossless LoRA expansion requires equal alpha/r scaling, "
                        f"but source={old_scale} and target={new_scale}."
                    )

    for key, old_value in state.items():
        if key not in current:
            mismatches.append((key, "missing_in_model", tuple(old_value.shape)))
            continue

        new_value = current[key]
        if tuple(old_value.shape) == tuple(new_value.shape):
            compatible[key] = old_value.to(dtype=new_value.dtype)
            continue

        expanded_value = None
        if allow_lora_rank_expansion and key.endswith("lora_A"):
            # A: [rank, in_features]
            if (
                old_value.ndim == 2
                and new_value.ndim == 2
                and old_value.shape[1] == new_value.shape[1]
                and old_value.shape[0] <= new_value.shape[0]
            ):
                if expected_source_lora_rank is not None and old_value.shape[0] != int(expected_source_lora_rank):
                    mismatches.append((key, "unexpected_source_rank", tuple(old_value.shape), int(expected_source_lora_rank)))
                    continue
                expanded_value = new_value.detach().clone()
                expanded_value[: old_value.shape[0], :] = old_value.to(dtype=new_value.dtype)

        elif allow_lora_rank_expansion and key.endswith("lora_B"):
            # B: [out_features, rank]
            if (
                old_value.ndim == 2
                and new_value.ndim == 2
                and old_value.shape[0] == new_value.shape[0]
                and old_value.shape[1] <= new_value.shape[1]
            ):
                if expected_source_lora_rank is not None and old_value.shape[1] != int(expected_source_lora_rank):
                    mismatches.append((key, "unexpected_source_rank", tuple(old_value.shape), int(expected_source_lora_rank)))
                    continue
                # LoRALinear initializes all new B columns to zero. Preserve that property.
                expanded_value = torch.zeros_like(new_value)
                expanded_value[:, : old_value.shape[1]] = old_value.to(dtype=new_value.dtype)

        if expanded_value is not None:
            compatible[key] = expanded_value
            expanded.append((key, tuple(old_value.shape), tuple(new_value.shape)))
        else:
            mismatches.append((key, tuple(old_value.shape), tuple(new_value.shape)))

    msg = pipe.dit.load_state_dict(compatible, strict=False)

    if rank == 0:
        print(f"[resume_trainable] path={path}")
        print(
            f"[resume_trainable] exact={len(compatible) - len(expanded)} "
            f"expanded={len(expanded)} mismatched={len(mismatches)}"
        )
        for item in expanded[:20]:
            print("[resume_trainable expanded]", item)
        if len(expanded) > 20:
            print(f"[resume_trainable expanded] ... and {len(expanded) - 20} more")
        for item in mismatches[:20]:
            print("[resume_trainable mismatch]", item)
        print(
            f"[resume_trainable] missing_after_load={len(msg.missing_keys)} "
            f"unexpected_after_load={len(msg.unexpected_keys)}"
        )

    if strict_resume and mismatches:
        preview = "\n".join(str(x) for x in mismatches[:20])
        raise RuntimeError(
            f"Checkpoint is not structurally compatible except for permitted LoRA expansion. "
            f"Found {len(mismatches)} unhandled mismatches. First mismatches:\n{preview}"
        )




# -------------------------
# SVI-style autoregressive windows and error recycling
# -------------------------

class TimestepErrorReplayBuffer:
    """Per-rank CPU fp16 replay memory organized by scheduler timestep grids."""
    def __init__(self, num_grids=50, capacity_per_grid=32, replacement_strategy="l2_batch"):
        self.num_grids = int(num_grids)
        self.capacity_per_grid = int(capacity_per_grid)
        self.replacement_strategy = replacement_strategy
        self.buffers = {i: [] for i in range(self.num_grids)}
        self.fifo_pos = {i: 0 for i in range(self.num_grids)}

    def __len__(self):
        return sum(len(v) for v in self.buffers.values())

    def has_grid(self, grid_idx):
        return len(self.buffers[int(grid_idx)]) > 0

    def has_any(self):
        return any(self.buffers[i] for i in range(self.num_grids))

    @torch.no_grad()
    def add(self, error, grid_idx):
        grid_idx = int(grid_idx)
        x = error.detach().to(device="cpu", dtype=torch.float16).contiguous()
        buf = self.buffers[grid_idx]
        if len(buf) < self.capacity_per_grid:
            buf.append(x)
            return
        if self.replacement_strategy == "random":
            buf[random.randrange(len(buf))] = x
        elif self.replacement_strategy == "fifo":
            pos = self.fifo_pos[grid_idx] % len(buf)
            buf[pos] = x
            self.fifo_pos[grid_idx] = pos + 1
        elif self.replacement_strategy in ("l2_batch", "l2_similarity"):
            # Replace the most similar sample, preserving diversity.
            x_flat = x.float().flatten()
            if self.replacement_strategy == "l2_batch":
                stacked = torch.stack(buf).float().flatten(start_dim=1)
                idx = torch.argmin(torch.norm(stacked - x_flat.unsqueeze(0), dim=1)).item()
            else:
                idx = min(range(len(buf)), key=lambda i: torch.norm(buf[i].float().flatten() - x_flat).item())
            buf[idx] = x
        else:
            raise ValueError(f"Unknown replacement_strategy={self.replacement_strategy}")

    @torch.no_grad()
    def sample(self, reference, grid_idx=None, from_all_grids=False, modulate_factor=0.0):
        candidates = []
        if from_all_grids:
            for buf in self.buffers.values():
                candidates.extend(buf)
        elif grid_idx is not None:
            candidates = self.buffers[int(grid_idx)]
        if not candidates:
            return torch.zeros_like(reference)
        x = random.choice(candidates).to(device=reference.device, dtype=reference.dtype)
        if modulate_factor > 0:
            x = x * random.uniform(1.0 - modulate_factor, 1.0 + modulate_factor)
        return x

    def state_dict(self):
        return {
            "num_grids": self.num_grids,
            "capacity_per_grid": self.capacity_per_grid,
            "replacement_strategy": self.replacement_strategy,
            "buffers": self.buffers,
            "fifo_pos": self.fifo_pos,
        }

    def load_state_dict(self, state):
        self.num_grids = int(state["num_grids"])
        self.capacity_per_grid = int(state["capacity_per_grid"])
        self.replacement_strategy = state["replacement_strategy"]
        self.buffers = state["buffers"]
        self.fifo_pos = state.get("fifo_pos", {i: 0 for i in range(self.num_grids)})


def build_grid_sigmas(num_grids: int, sigma_shift: float, device):
    """Representative sigma values matching shifted FlowMatch inference spacing."""
    # Uniform raw timesteps followed by Wan's common shift transform.
    raw = torch.linspace(1.0, 0.0, steps=num_grids, device=device, dtype=torch.float32)
    shift = float(sigma_shift)
    if shift != 1.0:
        raw = shift * raw / (1.0 + (shift - 1.0) * raw)
    return raw


def timestep_to_grid(timestep, grid_sigmas):
    sigma = timestep.detach().float().flatten()[0] / 1000.0
    return int(torch.argmin(torch.abs(grid_sigmas - sigma)).item())


def project_to_clean_endpoint(sample, velocity, sigma):
    return sample - sigma * velocity


def project_to_noise_endpoint(sample, velocity, sigma):
    return sample + (1.0 - sigma) * velocity


def choose_window_start(num_slots: int, window_size: int, rng=random):
    if num_slots <= window_size:
        return 0
    return rng.randint(0, num_slots - window_size)


def slice_context_window(context, start, length):
    if context.shape[1] < start + length:
        raise ValueError(f"context has {context.shape[1]} slots, cannot slice [{start}:{start+length}]")
    return context[:, start:start + length]


def build_history_y(template_y, history_latent, window_slots):
    """Create one-history-slot Wan image condition y=[4 mask channels, VAE channels]."""
    if template_y is None:
        return None
    if template_y.shape[1] < 4 + history_latent.shape[1]:
        raise ValueError(f"Unexpected y channels={template_y.shape[1]}, history channels={history_latent.shape[1]}")
    y = torch.zeros(
        (history_latent.shape[0], template_y.shape[1], window_slots, history_latent.shape[-2], history_latent.shape[-1]),
        device=history_latent.device, dtype=history_latent.dtype,
    )
    # Reuse the exact first-slot mask encoding produced by the existing pipeline/cache.
    y[:, :4, 0:1] = template_y[:, :4, 0:1].to(y.dtype)
    y[:, 4:4 + history_latent.shape[1], 0:1] = history_latent
    return y


def prepare_ar_window_from_cached(cached, window_size=21):
    cond = cached["cond"]
    full_latents = cached["target_latents"]
    total_slots = full_latents.shape[2]
    if total_slots < window_size:
        raise ValueError(f"Need at least {window_size} slots, got {total_slots}")
    start = choose_window_start(total_slots, window_size)
    end = start + window_size
    target = full_latents[:, :, start:end].contiguous()
    context = slice_context_window(cond["context"], start, window_size).contiguous()
    history = target[:, :, 0:1].detach()

    # For arbitrary cached windows, CLIP must be cached per slot. Current 21-slot data always starts at 0.
    clip_feature = cond.get("clip_feature")
    if start > 0:
        per_slot = cached.get("clip_features_per_slot")
        if per_slot is None:
            raise ValueError(
                "Random cached window start > 0 requires clip_features_per_slot in text cache. "
                "Current 21-slot data naturally uses start=0; for future long caches, update encode_text.py."
            )
        clip_feature = per_slot[:, start]

    ar_cond = dict(cond)
    ar_cond["context"] = context
    ar_cond["clip_feature"] = clip_feature
    ar_cond["y"] = build_history_y(cond["y"], history, window_size)
    ar_cond["first_frame_latents"] = history
    ar_cond["num_slots"] = window_size
    ar_cond["num_frames"] = 1 + (window_size - 1) * 4
    return ar_cond, target, start


def all_gather_tensor(x, distributed, world_size):
    if not distributed:
        return [x.detach()]
    gathered = [torch.empty_like(x) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, x.detach())
    return gathered


def save_error_replay_state(path, step, noise_buffer, data_buffer, iteration_count):
    ensure_dir(os.path.dirname(path))
    torch.save({
        "step": int(step),
        "iteration_count": int(iteration_count),
        "noise_error_buffer": noise_buffer.state_dict(),
        "data_error_buffer": data_buffer.state_dict(),
    }, path)


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
# preview inference
# -------------------------

@torch.no_grad()
def save_preview(pipe: WanVideoPipeline, sample: Dict, infer_cfg: Dict, out_root: str, step: int):
    """Run a short keyframe inference on rank0 and save png previews."""
    preview_dir = Path(out_root) / "previews" / f"step_{step:06d}" / sample.get("id", "sample")
    ensure_dir(str(preview_dir))

    input_image = Image.open(sample["image"]).convert("RGB")
    pipe.dit.eval()

    frames = pipe(
        prompt=sample["prompt"],
        negative_prompt=sample.get("negative_prompt", ""),
        input_image=input_image,
        frame_prompts=sample["frame_prompts"],
        height=infer_cfg.get("height", 480),
        width=infer_cfg.get("width", 832),
        num_inference_steps=infer_cfg.get("preview_num_inference_steps", infer_cfg.get("num_inference_steps", 10)),
        cfg_scale=infer_cfg.get("cfg_scale", 5.0),
        cfg_merge=False,
        sigma_shift=infer_cfg.get("sigma_shift", 5.0),
        seed=infer_cfg.get("seed", 42),
        rand_device=infer_cfg.get("rand_device", "cpu"),
        tiled=infer_cfg.get("tiled", True),
        tile_size=tuple(infer_cfg.get("tile_size", [30, 52])),
        tile_stride=tuple(infer_cfg.get("tile_stride", [15, 26])),
    )

    for i, frame in enumerate(frames):
        frame.save(preview_dir / f"{i:02d}.png")

    pipe.dit.train()
    return str(preview_dir)


# -------------------------
# training
# -------------------------

class StepProfiler:
    """Lightweight first-N-step profiler with one timing all-reduce per step.

    GPU work is synchronized at section boundaries only while profiling. This adds
    overhead to the first few steps intentionally, but gives trustworthy wall times.
    Reported ``max`` is the slowest rank and therefore the distributed critical path.
    """
    def __init__(self, enabled, first_n_steps, rank, distributed, device, writer=None):
        self.enabled = bool(enabled)
        self.first_n_steps = int(first_n_steps)
        self.rank = int(rank)
        self.distributed = bool(distributed)
        self.device = device
        self.writer = writer
        self.stage_names = [
            "data_load",
            "window_prepare",
            "error_prepare",
            "forward",
            "backward",
            "optimizer_step",
            "endpoint_error",
            "replay_sync_update",
            "loss_reduce_log",
            "checkpoint",
            "total",
        ]
        self.accumulated_max = {name: 0.0 for name in self.stage_names}
        self.accumulated_avg = {name: 0.0 for name in self.stage_names}
        self.profiled_steps = 0

    def active(self, step):
        return self.enabled and int(step) <= self.first_n_steps

    def sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def now(self):
        return time.perf_counter()

    def reduce_and_report(self, step, global_step, local_times):
        if not self.active(step):
            return
        values = torch.tensor(
            [float(local_times.get(name, 0.0)) for name in self.stage_names],
            device=self.device, dtype=torch.float64,
        )
        max_values = values.clone()
        sum_values = values.clone()
        if self.distributed:
            torch.distributed.all_reduce(max_values, op=torch.distributed.ReduceOp.MAX)
            torch.distributed.all_reduce(sum_values, op=torch.distributed.ReduceOp.SUM)
            sum_values /= torch.distributed.get_world_size()
        max_cpu = max_values.cpu().tolist()
        avg_cpu = sum_values.cpu().tolist()
        self.profiled_steps += 1
        for name, vmax, vavg in zip(self.stage_names, max_cpu, avg_cpu):
            self.accumulated_max[name] += vmax
            self.accumulated_avg[name] += vavg

        if self.rank == 0:
            total = max(max_cpu[self.stage_names.index("total")], 1e-12)
            parts = []
            for name, value in zip(self.stage_names, max_cpu):
                if name == "total":
                    continue
                parts.append(f"{name}={value:.2f}s({100.0*value/total:.1f}%)")
            print(f"[profile step {global_step:06d} max-rank] total={total:.2f}s | " + " | ".join(parts))
            if self.writer is not None:
                for name, vmax, vavg in zip(self.stage_names, max_cpu, avg_cpu):
                    self.writer.add_scalar(f"profile/max_rank/{name}_sec", vmax, global_step)
                    self.writer.add_scalar(f"profile/avg_rank/{name}_sec", vavg, global_step)

            if int(step) == self.first_n_steps:
                n = max(self.profiled_steps, 1)
                mean_total = self.accumulated_max["total"] / n
                print(f"\n[profile summary first {n} steps: mean slowest-rank time]")
                rows = []
                for name in self.stage_names:
                    mean_max = self.accumulated_max[name] / n
                    mean_avg = self.accumulated_avg[name] / n
                    pct = 100.0 * mean_max / max(mean_total, 1e-12)
                    rows.append((mean_max, name, mean_avg, pct))
                for mean_max, name, mean_avg, pct in sorted(rows, reverse=True):
                    print(f"  {name:20s} max-rank={mean_max:8.3f}s  avg-rank={mean_avg:8.3f}s  share={pct:6.2f}%")
                print("[profile note] Profiling synchronizes CUDA at section boundaries, so these first-N steps are intentionally slower than normal training.\n")


def train(cfg: Dict):
    if deepspeed is None:
        raise ImportError("deepspeed is not installed")

    distributed, rank, world_size, local_rank = init_distributed()
    train_cfg = cfg["train"]
    infer_cfg = cfg["infer"]
    data_cfg = cfg.get("data", {})
    svi_cfg = cfg.get("svi", {})

    out_dir = train_cfg["out_dir"]
    log_dir = os.path.join(out_dir, "tb")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    replay_dir = os.path.join(out_dir, "error_replay")
    if is_rank0(rank):
        for d in (out_dir, log_dir, ckpt_dir, replay_dir):
            ensure_dir(d)
        with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    barrier(distributed, local_rank)

    train_entries = load_sample_entries(cfg, split="train")
    val_entries = load_sample_entries(cfg, split="val") if "val_manifest" in data_cfg else []
    pipe = build_pipe(cfg)
    pipe.load_models_to_device(["dit"])
    pipe.dit.train()

    param_groups = set_trainable_params(pipe, train_cfg)
    optimizer, params_for_clip = build_optimizer(param_groups, train_cfg)
    resume_trainable_path = train_cfg.get("resume_trainable_path")
    if resume_trainable_path:
        lora_cfg = train_cfg.get("lora", {})
        load_trainable_checkpoint_into_dit(
            pipe,
            resume_trainable_path,
            rank=rank,
            strict_resume=bool(train_cfg.get("strict_resume", True)),
            allow_lora_rank_expansion=bool(train_cfg.get("allow_lora_rank_expansion", False)),
            expected_source_lora_rank=train_cfg.get("source_lora_rank"),
            source_lora_alpha=train_cfg.get("source_lora_alpha"),
            target_lora_alpha=lora_cfg.get("alpha"),
        )

    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))
    ds_config = cfg.get("deepspeed") or {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": grad_accum,
        "train_batch_size": max(1, world_size) * grad_accum,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 2, "contiguous_gradients": True, "overlap_comm": True,
            "reduce_scatter": True, "allgather_partitions": True,
            "reduce_bucket_size": 5e8, "allgather_bucket_size": 5e8,
        },
        "gradient_clipping": float(train_cfg.get("grad_clip", 1.0)),
        "steps_per_print": int(train_cfg.get("ds_steps_per_print", 1000000)),
    }
    engine, optimizer, _, _ = deepspeed.initialize(
        model=pipe.dit, optimizer=optimizer, model_parameters=params_for_clip, config=ds_config
    )
    pipe.dit = engine.module
    pipe.dit.train()
    writer = SummaryWriter(log_dir=log_dir) if is_rank0(rank) else None

    # SVI-Shot defaults, adapted to one semantic keyframe history slot.
    window_size = int(svi_cfg.get("window_size", 21))
    num_grids = int(svi_cfg.get("num_grids", 50))
    buffer_k = int(svi_cfg.get("error_buffer_k", 32))
    replacement = svi_cfg.get("buffer_replacement_strategy", "l2_batch")
    warmup_iter = int(svi_cfg.get("buffer_warmup_iter", 50))
    clean_prob = float(svi_cfg.get("clean_prob", 0.2))
    y_prob = float(svi_cfg.get("y_prob", 0.9))
    latent_prob = float(svi_cfg.get("latent_prob", 0.9))
    noise_prob = float(svi_cfg.get("noise_prob", 0.01))
    clean_buffer_update_prob = float(svi_cfg.get("clean_buffer_update_prob", 0.1))
    y_from_all = bool(svi_cfg.get("y_error_sample_from_all_grids", True))
    modulate = float(svi_cfg.get("error_modulate_factor", 0.0))
    use_error_recycling = bool(svi_cfg.get("use_error_recycling", True))
    mask_history_loss = bool(svi_cfg.get("mask_history_loss", True))

    noise_error_buffer = TimestepErrorReplayBuffer(num_grids, buffer_k, replacement)
    data_error_buffer = TimestepErrorReplayBuffer(num_grids, buffer_k, replacement)
    iteration_count = 0
    replay_resume = svi_cfg.get("resume_error_replay_path")
    if replay_resume:
        replay_resume = str(replay_resume).format(rank=rank, local_rank=local_rank)
        state = torch.load(replay_resume, map_location="cpu")
        noise_error_buffer.load_state_dict(state["noise_error_buffer"])
        data_error_buffer.load_state_dict(state["data_error_buffer"])
        iteration_count = int(state.get("iteration_count", 0))

    grid_sigmas = build_grid_sigmas(num_grids, infer_cfg.get("sigma_shift", 5.0), pipe.device)

    steps = int(train_cfg.get("steps", 1000))
    step_offset = int(train_cfg.get("step_offset", 0))
    log_every = int(train_cfg.get("log_every", 10))
    save_every = int(train_cfg.get("save_every", 1000))
    preview_every = int(train_cfg.get("preview_every", 0))
    use_grad_ckpt = bool(train_cfg.get("use_gradient_checkpointing", True))
    min_t = float(train_cfg.get("min_timestep", 0.0))
    max_t = float(train_cfg.get("max_timestep", 1000.0))
    shuffle = bool(data_cfg.get("shuffle", True))
    shuffle_seed = int(data_cfg.get("shuffle_seed", 42))
    empty_cache_every = int(train_cfg.get("empty_cache_every", 20))
    cache_mode = bool(data_cfg.get("cache_mode", False) or ("text_cache_dir" in data_cfg and "vae_cache_dir" in data_cfg))

    profile_enabled = bool(train_cfg.get("profile_enabled", True))
    profile_first_n_steps = int(train_cfg.get("profile_first_n_steps", 10))
    profiler = StepProfiler(
        enabled=profile_enabled,
        first_n_steps=profile_first_n_steps,
        rank=rank,
        distributed=distributed,
        device=pipe.device,
        writer=writer,
    )
    if is_rank0(rank):
        print(f"[profile] enabled={profile_enabled}, first_n_steps={profile_first_n_steps}")

    if not cache_mode:
        raise NotImplementedError(
            "This SVI training script currently expects cached context/y/target latents. "
            "That is your established high-throughput path; long raw sequences can be supported after cache generation."
        )

    current_epoch, order = -1, []
    rank_print(rank, f"[SVI] window={window_size}, grids={num_grids}, k={buffer_k}, world={world_size}")
    rank_print(rank, f"[SVI probabilities] clean={clean_prob}, y={y_prob}, latent={latent_prob}, noise={noise_prob}")

    for step in range(1, steps + 1):
        profile_active = profiler.active(step)
        stage_times = {name: 0.0 for name in profiler.stage_names}
        if profile_active:
            profiler.sync()
            step_t0 = profiler.now()

        engine.zero_grad()
        global_step = step_offset + step
        global_i = (global_step - 1) * world_size + rank
        epoch = global_i // len(train_entries)
        pos = global_i % len(train_entries)
        if epoch != current_epoch:
            order = make_epoch_order(len(train_entries), epoch, shuffle_seed, shuffle)
            current_epoch = epoch
        entry = train_entries[order[pos]]

        if profile_active:
            profiler.sync()
            t0 = profiler.now()
        cached = load_cached_training_item(entry, data_cfg, pipe.device, pipe.torch_dtype)
        # Optional future cache field for arbitrary-window CLIP conditioning.
        try:
            text_obj = torch.load(cached["text_path"], map_location="cpu")
            if "clip_features_per_slot" in text_obj:
                cached["clip_features_per_slot"] = move_tensor(
                    text_obj["clip_features_per_slot"], pipe.device, pipe.torch_dtype
                )
        except Exception:
            pass
        if profile_active:
            profiler.sync()
            stage_times["data_load"] = profiler.now() - t0
            t0 = profiler.now()
        cond, clean_latents, window_start = prepare_ar_window_from_cached(cached, window_size)
        pipe.load_models_to_device(["dit"])
        pipe.dit.train()
        if profile_active:
            profiler.sync()
            stage_times["window_prepare"] = profiler.now() - t0
            t0 = profiler.now()

        b = clean_latents.shape[0]
        timestep = torch.empty((b,), device=pipe.device, dtype=torch.float32).uniform_(min_t, max_t)
        sigma = (timestep / 1000.0).view(b, 1, 1, 1, 1).to(clean_latents.dtype)
        grid_idx = timestep_to_grid(timestep, grid_sigmas)

        base_noise = torch.randn_like(clean_latents)
        noise_w_error = base_noise
        latents_w_error = clean_latents
        y_w_error = cond["y"].clone() if cond["y"] is not None else None

        use_clean = random.random() < clean_prob
        add_noise_error = (not use_clean) and (random.random() < noise_prob)
        add_y_error = (not use_clean) and (random.random() < y_prob)
        add_latent_error = (not use_clean) and (random.random() < latent_prob)

        if use_error_recycling:
            if add_noise_error and noise_error_buffer.has_grid(grid_idx):
                noise_w_error = base_noise + noise_error_buffer.sample(
                    base_noise, grid_idx=grid_idx, from_all_grids=False, modulate_factor=modulate
                )
            if add_latent_error and data_error_buffer.has_grid(grid_idx):
                latents_w_error = clean_latents + data_error_buffer.sample(
                    clean_latents, grid_idx=grid_idx, from_all_grids=False, modulate_factor=modulate
                )
            if add_y_error and y_w_error is not None and data_error_buffer.has_any():
                sampled = data_error_buffer.sample(
                    clean_latents, grid_idx=grid_idx, from_all_grids=y_from_all, modulate_factor=modulate
                )
                # One semantic history keyframe: inject exactly one temporal latent slot into y's VAE channels.
                src_idx = random.randint(0, sampled.shape[2] - 1)
                y_w_error[:, 4:, 0:1] = y_w_error[:, 4:, 0:1] + sampled[:, :, src_idx:src_idx + 1]

        noisy_latents = (1.0 - sigma) * latents_w_error + sigma * noise_w_error
        target_velocity = noise_w_error - clean_latents  # self-correct toward clean data endpoint
        loss_mask = torch.ones_like(target_velocity)
        if mask_history_loss:
            loss_mask[:, :, 0:1] = 0
        if cond["fuse_vae_embedding_in_latents"] and cond["first_frame_latents"] is not None:
            noisy_latents[:, :, 0:1] = cond["first_frame_latents"]
            loss_mask[:, :, 0:1] = 0

        if profile_active:
            profiler.sync()
            stage_times["error_prepare"] = profiler.now() - t0
            t0 = profiler.now()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = pipe.model_fn(
                dit=engine.module, latents=noisy_latents, timestep=timestep,
                context=cond["context"], clip_feature=cond["clip_feature"], y=y_w_error,
                fuse_vae_embedding_in_latents=cond["fuse_vae_embedding_in_latents"],
                use_gradient_checkpointing=use_grad_ckpt,
                use_gradient_checkpointing_offload=False,
            )
            sq = (pred.float() - target_velocity.float()) ** 2 * loss_mask.float()
            loss = sq.sum() / loss_mask.float().sum().clamp_min(1.0)

        if profile_active:
            profiler.sync()
            stage_times["forward"] = profiler.now() - t0
            t0 = profiler.now()
        engine.backward(loss)
        if profile_active:
            profiler.sync()
            stage_times["backward"] = profiler.now() - t0
            t0 = profiler.now()
        engine.step()
        if profile_active:
            profiler.sync()
            stage_times["optimizer_step"] = profiler.now() - t0
            t0 = profiler.now()
        iteration_count += 1

        # Online one-step bidirectional endpoint errors, generated after the same corrupted input.
        if use_error_recycling:
            with torch.no_grad():
                pred_clean = project_to_clean_endpoint(noisy_latents, pred, sigma)
                gt_clean = project_to_clean_endpoint(noisy_latents, target_velocity, sigma)
                data_error = pred_clean - gt_clean
                pred_noise = project_to_noise_endpoint(noisy_latents, pred, sigma)
                gt_noise = project_to_noise_endpoint(noisy_latents, target_velocity, sigma)
                noise_error = pred_noise - gt_noise
                if profile_active:
                    profiler.sync()
                    stage_times["endpoint_error"] = profiler.now() - t0
                    t0 = profiler.now()

                should_update = (not use_clean) or (random.random() < clean_buffer_update_prob)
                if iteration_count <= warmup_iter:
                    # Every rank must enter every collective. Gather the local decision too;
                    # conditionally entering all_gather would deadlock when ranks sample differently.
                    update_flag = torch.tensor(
                        [int(should_update)], device=data_error.device, dtype=torch.int32
                    )
                    data_list = all_gather_tensor(data_error, distributed, world_size)
                    noise_list = all_gather_tensor(noise_error, distributed, world_size)
                    t_list = all_gather_tensor(timestep, distributed, world_size)
                    flag_list = all_gather_tensor(update_flag, distributed, world_size)
                    for de, ne, tt, flag in zip(data_list, noise_list, t_list, flag_list):
                        if int(flag.reshape(-1)[0].item()) == 0:
                            continue
                        gi = timestep_to_grid(tt, grid_sigmas)
                        data_error_buffer.add(de, gi)
                        noise_error_buffer.add(ne, gi)
                elif should_update:
                    data_error_buffer.add(data_error, grid_idx)
                    noise_error_buffer.add(noise_error, grid_idx)

        if profile_active:
            profiler.sync()
            stage_times["replay_sync_update"] = profiler.now() - t0
            t0 = profiler.now()

        loss_detached = loss.detach()
        if distributed:
            loss_for_log = loss_detached.clone()
            torch.distributed.all_reduce(loss_for_log, op=torch.distributed.ReduceOp.AVG)
        else:
            loss_for_log = loss_detached

        if is_rank0(rank) and step % log_every == 0:
            print(
                f"[step {global_step:06d}] loss={loss_for_log.item():.6f} "
                f"start={window_start} clean={int(use_clean)} y={int(add_y_error)} "
                f"latent={int(add_latent_error)} noise={int(add_noise_error)} "
                f"grid={grid_idx} buffers=({len(data_error_buffer)},{len(noise_error_buffer)})"
            )
            writer.add_scalar("train/loss", loss_for_log.item(), global_step)
            writer.add_scalar("svi/data_buffer_size", len(data_error_buffer), global_step)
            writer.add_scalar("svi/noise_buffer_size", len(noise_error_buffer), global_step)
            writer.add_scalar("svi/grid", grid_idx, global_step)

        if profile_active:
            profiler.sync()
            stage_times["loss_reduce_log"] = profiler.now() - t0
            t0 = profiler.now()

        if step % save_every == 0 or step == steps:
            tag = f"step_{global_step:06d}"
            engine.save_checkpoint(os.path.join(ckpt_dir, "deepspeed"), tag=tag)
            # Every rank saves its local replay memory because buffers diverge after warmup.
            save_error_replay_state(
                os.path.join(replay_dir, f"replay_{tag}_rank{rank:02d}.pt"),
                global_step, noise_error_buffer, data_error_buffer, iteration_count,
            )
            if is_rank0(rank):
                ckpt_path = os.path.join(ckpt_dir, f"trainable_{tag}.pt")
                save_checkpoint(pipe, optimizer, ckpt_path, global_step, loss_for_log.item(), cfg)
                print("saved checkpoint:", ckpt_path)

        if profile_active:
            profiler.sync()
            stage_times["checkpoint"] = profiler.now() - t0

        if preview_every > 0 and (step % preview_every == 0 or step == steps):
            barrier(distributed, local_rank)
            if is_rank0(rank):
                try:
                    preview_entry = val_entries[0] if val_entries else train_entries[0]
                    preview_dir = save_preview(pipe, materialize_sample(preview_entry), infer_cfg, out_dir, global_step)
                    print("saved preview:", preview_dir)
                except Exception as e:
                    print("[preview failed]", repr(e))
            barrier(distributed, local_rank)
            pipe.dit.train()

        if profile_active:
            profiler.sync()
            stage_times["total"] = profiler.now() - step_t0
            profiler.reduce_and_report(step, global_step, stage_times)

        del cached, cond, clean_latents, base_noise, noise_w_error, latents_w_error, noisy_latents
        del target_velocity, loss_mask, pred, loss, loss_detached
        if y_w_error is not None:
            del y_w_error
        if torch.cuda.is_available() and empty_cache_every > 0 and step % empty_cache_every == 0:
            torch.cuda.empty_cache()

    if writer is not None:
        writer.close()
    barrier(distributed, local_rank)
    rank_print(rank, "done:", out_dir)
    if distributed and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_json(args.config)
    train(cfg)


if __name__ == "__main__":
    main()