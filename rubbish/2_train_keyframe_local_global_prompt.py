"""
Pure dual-context keyframe training.

- No SVI / error recycling.
- No previous trainable checkpoint is loaded.
- Rank-128 LoRA is attached only to self-attention, global cross-attention,
  and local cross-attention Linear layers.
- The per-layer global/local bounded gates are trained as scalar parameters.
"""

import os
import sys
sys.stdout.reconfigure(line_buffering=True)
import json
import math
import random
import argparse
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

from diffsynth.pipelines.key_frame_dual_context import WanVideoPipeline
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
        rank: int = 128,
        alpha: int = 128,
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
        skip_keywords = []

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
    ##### DUAL-CONTEXT TRAINING
    Train only:
      - LoRA on self_attn
      - LoRA on global cross_attn
      - LoRA on local_cross_attn
      - global_gate / local_gate as full scalar parameters

    All pretrained base weights remain frozen.
    """
    dit = pipe.dit

    for p in dit.parameters():
        p.requires_grad_(False)

    lora_cfg = train_cfg.get("lora", {})
    inject_lora_to_dit(
        dit,
        target_keywords=lora_cfg.get(
            "target_keywords",
            ["self_attn", "cross_attn", "local_cross_attn"],
        ),
        rank=int(lora_cfg.get("rank", 128)),
        alpha=int(lora_cfg.get("alpha", 128)),
        dropout=float(lora_cfg.get("dropout", 0.05)),
        skip_keywords=lora_cfg.get("skip_keywords", []),
    )

    for name, p in dit.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad_(True)
        elif name.endswith("global_gate") or name.endswith("local_gate"):
            p.requires_grad_(True)

    groups = {
        "lora": [],
        "context_gates": [],
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
        elif name.endswith("global_gate") or name.endswith("local_gate"):
            groups["context_gates"].append(p)
            tag = "context_gate"
        else:
            raise RuntimeError(f"Unexpected trainable parameter: {name}")

        print(f"  [{tag}] {name}: {n / 1e6:.6f}M")

    print(f"Total trainable params: {total / 1e6:.3f}M\n")
    return groups

def build_optimizer(param_groups: Dict[str, List[nn.Parameter]], train_cfg: Dict):
    optim_groups = []

    if len(param_groups["lora"]) > 0:
        optim_groups.append({
            "params": param_groups["lora"],
            "lr": train_cfg.get("lora_lr", train_cfg.get("lr", 2e-5)),
            "name": "lora",
        })

    if len(param_groups["context_gates"]) > 0:
        optim_groups.append({
            "params": param_groups["context_gates"],
            "lr": train_cfg.get("gate_lr", 1e-4),
            "name": "context_gates",
            "weight_decay": 0.0,
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


def get_context_scales(dit: nn.Module):
    """Return [(layer_id, global_scale, local_scale), ...] as Python floats."""
    model = dit.module if hasattr(dit, "module") else dit
    values = []
    for layer_id, block in enumerate(model.blocks):
        if not hasattr(block, "context_scales"):
            raise AttributeError(f"block {layer_id} has no context_scales()")
        global_scale, local_scale = block.context_scales()
        values.append((
            layer_id,
            float(global_scale.detach().float().cpu().item()),
            float(local_scale.detach().float().cpu().item()),
        ))
    return values


def log_context_scales(dit: nn.Module, writer, global_step: int):
    """Print and TensorBoard-log every layer's global/local scale every step."""
    values = get_context_scales(dit)
    compact = " ".join(
        f"L{layer_id:02d}:g={global_scale:.4f},l={local_scale:.4f}"
        for layer_id, global_scale, local_scale in values
    )
    print(f"[context_scales step={global_step:06d}] {compact}")

    if writer is not None:
        for layer_id, global_scale, local_scale in values:
            writer.add_scalar(
                f"context_scale/layer_{layer_id:02d}/global",
                global_scale,
                global_step,
            )
            writer.add_scalar(
                f"context_scale/layer_{layer_id:02d}/local",
                local_scale,
                global_step,
            )



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
    复用 key_frame_dual_context.py 里的 units，得到：
      global_context: [1, 512, text_dim]
      local_context: [1, F, 512, text_dim]
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
        "global_context": inputs_posi["global_context"].detach(),
        "local_context": inputs_posi["local_context"].detach(),
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
        print("  global_context:", tuple(cond["global_context"].shape))
        print("  local_context:", tuple(cond["local_context"].shape))
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


def as_batched_global_context(x: torch.Tensor) -> torch.Tensor:
    """Accept [L,D] or [1,L,D], return [1,L,D]."""
    if x.ndim == 2:
        return x.unsqueeze(0)
    if x.ndim == 3:
        return x
    raise ValueError(
        f"global_context must be 2D/3D, got shape={tuple(x.shape)}"
    )


def as_batched_local_context(x: torch.Tensor) -> torch.Tensor:
    """Accept [F,L,D] or [1,F,L,D], return [1,F,L,D]."""
    if x.ndim == 3:
        return x.unsqueeze(0)
    if x.ndim == 4:
        return x
    raise ValueError(
        f"local_context must be 3D/4D, got shape={tuple(x.shape)}"
    )


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
      text_cache_dir/sample_xxxxxx.pt: global_context, local_context, clip_feature
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

    if "global_context" not in text_obj or "local_context" not in text_obj:
        raise KeyError(
            f"dual-context cache requires global_context/local_context: {text_path}"
        )
    global_context = as_batched_global_context(text_obj["global_context"])
    local_context = as_batched_local_context(text_obj["local_context"])
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
        "global_context": move_tensor(global_context, device=device, dtype=dtype),
        "local_context": move_tensor(local_context, device=device, dtype=dtype),
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


def load_trainable_checkpoint_into_dit(pipe: WanVideoPipeline, path: str, rank: int = 0):
    """
    Load the trainable_state_dict saved by save_checkpoint().
    This loads weights only. Optimizer/DeepSpeed states are intentionally not restored here.
    Set train.resume_trainable_path to null/empty to train from scratch.
    """
    if path is None or str(path).strip() == "":
        return

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("trainable_state_dict", ckpt)

    missing_shape = []
    compatible = {}
    current = pipe.dit.state_dict()

    for k, v in state.items():
        if k not in current:
            missing_shape.append((k, "missing_in_model", tuple(v.shape)))
            continue
        if tuple(current[k].shape) != tuple(v.shape):
            missing_shape.append((k, tuple(v.shape), tuple(current[k].shape)))
            continue
        compatible[k] = v.to(dtype=current[k].dtype)

    msg = pipe.dit.load_state_dict(compatible, strict=False)

    if rank == 0:
        print(f"[resume_trainable] path={path}")
        print(f"[resume_trainable] loaded={len(compatible)} skipped={len(missing_shape)}")
        print(f"[resume_trainable] missing_after_load={len(msg.missing_keys)} unexpected_after_load={len(msg.unexpected_keys)}")
        for item in missing_shape[:20]:
            print("[resume_trainable skipped]", item)
        if len(missing_shape) > 20:
            print(f"[resume_trainable skipped] ... and {len(missing_shape) - 20} more")


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

def train(cfg: Dict):
    if deepspeed is None:
        raise ImportError(
            "deepspeed is not installed. Run: pip install deepspeed -i https://pypi.tuna.tsinghua.edu.cn/simple"
        )

    distributed, rank, world_size, local_rank = init_distributed()

    train_cfg = cfg["train"]
    infer_cfg = cfg["infer"]
    data_cfg = cfg.get("data", {})

    out_dir = train_cfg["out_dir"]
    log_dir = os.path.join(out_dir, "tb")
    ckpt_dir = os.path.join(out_dir, "checkpoints")

    if is_rank0(rank):
        ensure_dir(out_dir)
        ensure_dir(log_dir)
        ensure_dir(ckpt_dir)
        with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    barrier(distributed, local_rank)

    train_entries = load_sample_entries(cfg, split="train")
    val_entries = load_sample_entries(cfg, split="val") if "val_manifest" in data_cfg else []

    rank_print(rank, f"[distributed] enabled={distributed} rank={rank}/{world_size} local_rank={local_rank}")
    rank_print(rank, f"[dataset] train samples = {len(train_entries)}")
    rank_print(rank, f"[dataset] val samples = {len(val_entries)}")

    print_first_n = int(data_cfg.get("print_first_n", 20))
    for e in train_entries[:print_first_n]:
        rank_print(rank, f"  - {entry_name(e)}: {e.get('path', '<inline sample>')}")
    if len(train_entries) > print_first_n:
        rank_print(rank, f"  ... and {len(train_entries) - print_first_n} more")

    # 保留原来的模型构建逻辑，不改 build_pipe / ModelConfig / tokenizer 对齐方式。
    pipe = build_pipe(cfg)

    pipe.load_models_to_device(["dit"])
    pipe.dit.train()

    param_groups = set_trainable_params(pipe, train_cfg)
    optimizer, params_for_clip = build_optimizer(param_groups, train_cfg)

    ##### TRAIN FROM SCRATCH: do not load any previously trained LoRA/checkpoint.
    resume_trainable_path = train_cfg.get("resume_trainable_path", None)
    if resume_trainable_path:
        raise ValueError(
            "This script is configured for scratch dual-context training; "
            "set train.resume_trainable_path to null."
        )

    # DeepSpeed ZeRO-2 shards optimizer states and gradients across GPUs.
    # DDP alone does NOT solve this OOM, because DDP replicates optimizer states on every GPU.
    ds_config = cfg.get("deepspeed", {})
    if not ds_config:
        grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))
        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": grad_accum,
            "train_batch_size": max(1, world_size) * grad_accum,
            "bf16": {"enabled": True},
            "zero_optimization": {
                "stage": 2,
                "contiguous_gradients": True,
                "overlap_comm": True,
                "reduce_scatter": True,
                "allgather_partitions": True,
                "reduce_bucket_size": 5e8,
                "allgather_bucket_size": 5e8
            },
            "gradient_clipping": float(train_cfg.get("grad_clip", 1.0)),
            "steps_per_print": int(train_cfg.get("ds_steps_per_print", 1000000)),
            "wall_clock_breakdown": False
        }

    engine, optimizer, _, _ = deepspeed.initialize(
        model=pipe.dit,
        optimizer=optimizer,
        model_parameters=params_for_clip,
        config=ds_config,
    )
    pipe.dit = engine.module
    pipe.dit.train()

    writer = SummaryWriter(log_dir=log_dir) if is_rank0(rank) else None

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
    verbose_prepare_every = int(train_cfg.get("verbose_prepare_every", 100))
    empty_cache_every = int(train_cfg.get("empty_cache_every", 20))

    cache_mode = bool(data_cfg.get("cache_mode", False) or ("text_cache_dir" in data_cfg and "vae_cache_dir" in data_cfg))
    verbose_cache_every = int(train_cfg.get("verbose_cache_every", 20))

    current_epoch = -1
    order = []

    steps_per_epoch = math.ceil(len(train_entries) / max(1, world_size))

    rank_print(rank, "\n[start training]")
    rank_print(rank, "  steps:", steps)
    rank_print(rank, "  step_offset:", step_offset)
    rank_print(rank, "  train entries:", len(train_entries))
    rank_print(rank, "  world_size:", world_size)
    rank_print(rank, "  approx global batch:", world_size)
    rank_print(rank, "  approx steps_per_epoch:", steps_per_epoch)
    rank_print(rank, "  min_timestep:", min_t)
    rank_print(rank, "  max_timestep:", max_t)
    rank_print(rank, "  cache_mode:", cache_mode)
    if cache_mode:
        rank_print(rank, "  text_cache_dir:", data_cfg.get("text_cache_dir"))
        rank_print(rank, "  vae_cache_dir:", data_cfg.get("vae_cache_dir"))

    for step in range(1, steps + 1):
        engine.zero_grad()

        # 每个 rank 在同一个 global step 取不同样本。
        # 不再提前构造 train_items，因此不会一次性 encode 几万条视频。
        global_step = step_offset + step
        global_i = (global_step - 1) * world_size + rank
        epoch = global_i // len(train_entries)
        pos = global_i % len(train_entries)

        if epoch != current_epoch:
            order = make_epoch_order(
                num_samples=len(train_entries),
                epoch=epoch,
                seed=shuffle_seed,
                shuffle=shuffle,
            )
            current_epoch = epoch
            if is_rank0(rank):
                print(f"[epoch {epoch}] reshuffle={shuffle}")

        entry_idx = order[pos]
        entry = train_entries[entry_idx]
        sample = None

        verbose_prepare = is_rank0(rank) and (
            step == 1 or (verbose_prepare_every > 0 and step % verbose_prepare_every == 0)
        )

        if cache_mode:
            cached = load_cached_training_item(
                entry=entry,
                data_cfg=data_cfg,
                device=pipe.device,
                dtype=pipe.torch_dtype,
            )
            sample_id = cached["sample_id"]
            cond = cached["cond"]
            target_latents = cached["target_latents"]

            if is_rank0(rank) and (step == 1 or (verbose_cache_every > 0 and step % verbose_cache_every == 0)):
                print("[cache item]")
                print("  sample_id:", sample_id)
                print("  text:", cached["text_path"])
                print("  vae:", cached["vae_path"])
                print("  global_context:", tuple(cond["global_context"].shape))
                print("  local_context:", tuple(cond["local_context"].shape))
                print("  clip_feature:", None if cond["clip_feature"] is None else tuple(cond["clip_feature"].shape))
                print("  y:", tuple(cond["y"].shape))
                print("  target_latents:", tuple(target_latents.shape))
                print("  num_slots:", cond["num_slots"], "num_frames:", cond["num_frames"])
        else:
            sample = materialize_sample(entry)
            sample_id = sample.get("id", f"sample_{entry_idx}")

            # 复用原来的 conditioning / VAE target latent 逻辑；
            # 只是从“启动前处理所有样本”改为“每步处理当前样本”。
            cond = prepare_conditioning(
                pipe,
                sample,
                infer_cfg,
                verbose=verbose_prepare,
            )
            target_latents = encode_target_latents(
                pipe,
                sample,
                infer_cfg,
                num_frames=cond["num_frames"],
                verbose=verbose_prepare,
            )

        # cache 模式下不会触发 VAE/text/image encoder；lazy 模式下前面可能切走了模型。
        pipe.load_models_to_device(["dit"])
        pipe.dit.train()

        b = target_latents.shape[0]

        timestep = torch.empty((b,), device=pipe.device, dtype=torch.float32).uniform_(min_t, max_t)
        sigma = (timestep / 1000.0).view(b, 1, 1, 1, 1).to(dtype=target_latents.dtype)

        noise = torch.randn_like(target_latents)
        noisy_latents = (1.0 - sigma) * target_latents + sigma * noise
        target_velocity = noise - target_latents

        loss_mask = torch.ones_like(target_velocity)

        if cond["fuse_vae_embedding_in_latents"] and cond["first_frame_latents"] is not None:
            noisy_latents[:, :, 0:1] = cond["first_frame_latents"]
            loss_mask[:, :, 0:1] = 0

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = pipe.model_fn(
                dit=engine.module,
                latents=noisy_latents,
                timestep=timestep,
                global_context=cond["global_context"],
                local_context=cond["local_context"],
                clip_feature=cond["clip_feature"],
                y=cond["y"],
                fuse_vae_embedding_in_latents=cond["fuse_vae_embedding_in_latents"],
                use_gradient_checkpointing=use_grad_ckpt,
                use_gradient_checkpointing_offload=False,
            )
            loss = ((pred.float() - target_velocity.float()) ** 2 * loss_mask.float()).mean()

        engine.backward(loss)
        engine.step()

        loss_detached = loss.detach()
        if distributed:
            loss_for_log = loss_detached.clone()
            torch.distributed.all_reduce(loss_for_log, op=torch.distributed.ReduceOp.AVG)
        else:
            loss_for_log = loss_detached

        if is_rank0(rank) and step % log_every == 0:
            print(
                f"[step {global_step:06d}] "
                f"epoch={epoch} "
                f"loss={loss_for_log.item():.6f} "
                f"sample_rank0={sample_id}"
            )
            writer.add_scalar("train/loss", loss_for_log.item(), global_step)
            writer.add_scalar(f"sample_loss_rank0/{sample_id}", loss_detached.item(), global_step)
            for group in optimizer.param_groups:
                name = group.get("name", "group")
                writer.add_scalar(f"train/lr/{name}", group["lr"], global_step)

        # ##### DUAL-CONTEXT: every optimization step, log all layers' two scales.
        if is_rank0(rank):
            log_context_scales(engine.module, writer, global_step)

        if step % save_every == 0 or step == steps:
            tag = f"step_{global_step:06d}"
            ds_save_dir = os.path.join(ckpt_dir, "deepspeed")
            engine.save_checkpoint(ds_save_dir, tag=tag)

            if is_rank0(rank):
                ckpt_path = os.path.join(ckpt_dir, f"trainable_{tag}.pt")
                save_checkpoint(
                    pipe=pipe,
                    optimizer=optimizer,
                    path=ckpt_path,
                    step=global_step,
                    loss=loss_for_log.item(),
                    cfg=cfg,
                )
                print("saved checkpoint:", ckpt_path)

        if preview_every > 0 and (step % preview_every == 0 or step == steps):
            barrier(distributed, local_rank)
            if is_rank0(rank):
                try:
                    preview_entry = val_entries[0] if len(val_entries) > 0 else train_entries[0]
                    preview_sample = materialize_sample(preview_entry)
                    preview_dir = save_preview(pipe, preview_sample, infer_cfg, out_dir, global_step)
                    print("saved preview:", preview_dir)
                except Exception as e:
                    print("[preview failed]", repr(e))
            barrier(distributed, local_rank)
            pipe.dit.train()

        # 主动释放这一步产生的临时大张量，避免长跑时显存碎片累积。
        del sample, cond, target_latents, noise, noisy_latents, target_velocity, loss_mask, loss, loss_detached
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