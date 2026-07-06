#!/usr/bin/env python3
"""Evaluate one AgiBot keyframe experiment checkpoint.

The script shards samples across torchrun ranks, writes 21 predicted keyframes
per sample, and saves a 1 FPS side-by-side GT/prediction video for quick review.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffsynth.core import ModelConfig
from diffsynth.utils.data import save_video


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_step_from_checkpoint(path: Path) -> Optional[int]:
    stem = path.stem
    for prefix in ("trainable_step_", "step_"):
        if stem.startswith(prefix):
            try:
                return int(stem[len(prefix):])
            except ValueError:
                return None
    return None


def resolve_checkpoint(cfg: Dict[str, Any]) -> Tuple[Path, str]:
    checkpoint_cfg = cfg["checkpoint"]
    explicit_path = str(checkpoint_cfg.get("path", "")).strip()
    if explicit_path:
        path = Path(explicit_path)
        step = parse_step_from_checkpoint(path)
        return path, f"step_{step:06d}" if step is not None else path.stem

    experiment_dir = Path(cfg["experiment"]["dir"])
    checkpoint_dir = Path(
        checkpoint_cfg.get("dir", experiment_dir / "checkpoints")
    )
    selector = str(checkpoint_cfg.get("selector", "latest"))
    candidates = sorted(checkpoint_dir.glob("trainable_step_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No trainable_step_*.pt in {checkpoint_dir}")

    if selector == "latest":
        stepped = [
            (parse_step_from_checkpoint(path), path)
            for path in candidates
        ]
        stepped = [(step, path) for step, path in stepped if step is not None]
        if not stepped:
            raise FileNotFoundError(f"No stepped checkpoints in {checkpoint_dir}")
        step, path = max(stepped, key=lambda item: item[0])
        return path, f"step_{step:06d}"

    if selector.startswith("step_"):
        step = int(selector[len("step_"):])
    else:
        step = int(selector)
    path = checkpoint_dir / f"trainable_step_{step:06d}.pt"
    return path, f"step_{step:06d}"


def resolve_output_dir(cfg: Dict[str, Any], checkpoint_label: str) -> Path:
    output_dir = str(cfg["inference"].get("output_dir", "")).strip()
    if output_dir:
        return Path(output_dir)
    return Path(cfg["experiment"]["dir"]) / "eval" / checkpoint_label


def read_manifest(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.lstrip().startswith("#")
        ]


def resolve_path(base_dir: Path, value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else base_dir / path)


def load_sample(sample_dir: str, expected_num_slots: int) -> Dict[str, Any]:
    sample_root = Path(sample_dir)
    ann_path = sample_root / "annotation.json"
    with open(ann_path, "r", encoding="utf-8") as f:
        ann = json.load(f)

    frame_prompts = list(ann["frame_prompts"])
    if len(frame_prompts) != expected_num_slots:
        raise ValueError(
            f"{sample_root}: expected {expected_num_slots} frame prompts, "
            f"got {len(frame_prompts)}"
        )

    keyframes = [
        resolve_path(sample_root, value)
        for value in ann.get("keyframes", [])
    ]
    if keyframes and len(keyframes) != expected_num_slots:
        raise ValueError(
            f"{sample_root}: expected {expected_num_slots} keyframes, "
            f"got {len(keyframes)}"
        )

    return {
        "id": str(ann.get("id", sample_root.name)),
        "sample_dir": str(sample_root),
        "image": resolve_path(sample_root, ann["image"]),
        "prompt": str(ann.get("prompt", "")),
        "negative_prompt": str(ann.get("negative_prompt", "")),
        "frame_prompts": frame_prompts,
        "target_keyframes": keyframes,
    }


def init_distributed() -> Tuple[bool, int, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

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


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: int,
        dropout: float,
    ):
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.alpha = int(alpha)
        self.scale = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

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
        base_output = self.base(x)
        value = self.dropout(x).to(self.lora_A)
        value = value @ self.lora_A.t()
        value = value @ self.lora_B.t()
        return base_output + value.to(base_output) * self.scale


def get_parent_module(root: nn.Module, module_name: str):
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_lora(
    model: nn.Module,
    adapter_cfg: Dict[str, Any],
) -> List[str]:
    target_keywords = list(adapter_cfg["target_keywords"])
    skip_keywords = list(adapter_cfg.get("skip_keywords", []))
    rank = int(adapter_cfg["rank"])
    alpha = int(adapter_cfg["alpha"])
    dropout = float(adapter_cfg.get("dropout", 0.0))

    replaced: List[str] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(keyword in name for keyword in target_keywords):
            continue
        if any(keyword in name for keyword in skip_keywords):
            continue

        parent, child_name = get_parent_module(model, name)
        setattr(
            parent,
            child_name,
            LoRALinear(module, rank, alpha, dropout),
        )
        replaced.append(name)

    if not replaced:
        raise RuntimeError(
            "No LoRA modules were injected. Check adapter.target_keywords."
        )
    return replaced


def import_pipeline(conditioning_mode: str):
    if conditioning_mode == "local_only":
        from diffsynth.pipelines.keyframe_local_context import WanVideoPipeline
        return WanVideoPipeline
    if conditioning_mode == "dual_context":
        from diffsynth.pipelines.keyframe_dual_context import WanVideoPipeline
        return WanVideoPipeline
    raise ValueError(
        "checkpoint.conditioning_mode must be local_only or dual_context, "
        f"got {conditioning_mode}"
    )


def build_pipe(
    cfg: Dict[str, Any],
    conditioning_mode: str,
    local_rank: int,
):
    WanVideoPipeline = import_pipeline(conditioning_mode)
    model_cfg = cfg["model"]
    model_id = model_cfg["model_id"]
    tokenizer_path = model_cfg["tokenizer_path"]
    local_model_path = model_cfg.get("local_model_path")
    skip_download = bool(model_cfg.get("skip_download", False))
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
                local_model_path=local_model_path,
                skip_download=skip_download,
                **vram_config,
            ),
            ModelConfig(
                model_id=model_id,
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                local_model_path=local_model_path,
                skip_download=skip_download,
                **vram_config,
            ),
            ModelConfig(
                model_id=model_id,
                origin_file_pattern=(
                    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
                ),
                local_model_path=local_model_path,
                skip_download=skip_download,
                **vram_config,
            ),
            ModelConfig(
                model_id=model_id,
                origin_file_pattern="Wan2.1_VAE.pth",
                local_model_path=local_model_path,
                skip_download=skip_download,
                **vram_config,
            ),
        ],
        tokenizer_config=ModelConfig(path=tokenizer_path, skip_download=True),
        redirect_common_files=False,
        vram_limit=model_cfg.get("vram_limit_gb"),
        **(
            {
                "dual_context_global_scale_init": float(
                    cfg.get("dual_context", {}).get("global_scale_init", 1.0)
                ),
                "dual_context_local_scale_init": float(
                    cfg.get("dual_context", {}).get("local_scale_init", 1.0)
                ),
                "dual_context_max_context_scale": float(
                    cfg.get("dual_context", {}).get("max_context_scale", 2.0)
                ),
            }
            if conditioning_mode == "dual_context"
            else {}
        ),
    )
    return pipe


def load_trainable_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
) -> Dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state = checkpoint.get("trainable_state_dict", checkpoint)
    current = model.state_dict()

    compatible = {}
    skipped = []
    for name, value in state.items():
        if name not in current:
            skipped.append((name, "missing"))
            continue
        if tuple(value.shape) != tuple(current[name].shape):
            skipped.append(
                (name, tuple(value.shape), tuple(current[name].shape))
            )
            continue
        compatible[name] = value.to(dtype=current[name].dtype)

    message = model.load_state_dict(compatible, strict=False)
    if not compatible:
        raise RuntimeError(
            f"No checkpoint tensors matched model structure: {checkpoint_path}"
        )

    return {
        "loaded": len(compatible),
        "skipped": skipped,
        "missing_keys": message.missing_keys,
        "unexpected_keys": message.unexpected_keys,
        "global_step": int(
            checkpoint.get("global_step", checkpoint.get("step", 0))
        ),
    }


def save_contact_sheet(
    frames: List[Image.Image],
    output_path: Path,
    prefix: str,
    thumb_width: int,
) -> None:
    if not frames:
        return

    thumbs: List[Image.Image] = []
    label_height = 24
    for frame in frames:
        frame = frame.convert("RGB")
        ratio = thumb_width / frame.width
        thumb_height = max(1, round(frame.height * ratio))
        thumbs.append(frame.resize((thumb_width, thumb_height)))

    columns = min(7, len(thumbs))
    rows = (len(thumbs) + columns - 1) // columns
    tile_width = thumbs[0].width
    tile_height = thumbs[0].height + label_height
    canvas = Image.new(
        "RGB",
        (columns * tile_width, rows * tile_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)

    for index, thumb in enumerate(thumbs):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        canvas.paste(thumb, (x, y + label_height))
        draw.text(
            (x + 4, y + 4),
            f"{prefix} {index:02d}",
            fill=(0, 0, 0),
        )

    canvas.save(output_path)


def outputs_complete(sample_output: Path, expected: int) -> bool:
    prediction_dir = sample_output / "pred"
    files = sorted(prediction_dir.glob("[0-9][0-9].png"))
    return (
        len(files) >= expected
        and (sample_output / "compare_1fps.mp4").exists()
        and (sample_output / "meta.json").exists()
    )


def save_ground_truth(
    sample: Dict[str, Any],
    sample_output: Path,
    thumb_width: int,
) -> None:
    paths = sample["target_keyframes"]
    if not paths:
        return

    ground_truth_dir = sample_output / "gt"
    ensure_dir(ground_truth_dir)
    frames = []
    for index, path in enumerate(paths):
        frame = Image.open(path).convert("RGB")
        frame.save(ground_truth_dir / f"{index:02d}.png")
        frames.append(frame)

    save_contact_sheet(
        frames,
        sample_output / "gt_contact.jpg",
        prefix="gt",
        thumb_width=thumb_width,
    )


def resize_square(image: Image.Image, size: int) -> Image.Image:
    return image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)


def save_compare_video(
    gt_frames: List[Image.Image],
    pred_frames: List[Image.Image],
    sample_output: Path,
    square_size: int,
    fps: int,
) -> None:
    if len(gt_frames) != len(pred_frames):
        raise ValueError(
            f"GT/pred frame count mismatch: {len(gt_frames)} vs {len(pred_frames)}"
        )

    compare_dir = sample_output / "compare_frames"
    ensure_dir(compare_dir)
    combined_frames: List[Image.Image] = []
    for index, (gt, pred) in enumerate(zip(gt_frames, pred_frames)):
        gt_sq = resize_square(gt, square_size)
        pred_sq = resize_square(pred, square_size)
        canvas = Image.new("RGB", (square_size * 2, square_size), "black")
        canvas.paste(gt_sq, (0, 0))
        canvas.paste(pred_sq, (square_size, 0))
        canvas.save(compare_dir / f"{index:02d}.png")
        combined_frames.append(canvas)
    save_video(
        combined_frames,
        str(sample_output / "compare_1fps.mp4"),
        fps=fps,
        quality=5,
    )


def save_metadata(
    sample: Dict[str, Any],
    sample_output: Path,
    cfg: Dict[str, Any],
    checkpoint_info: Dict[str, Any],
) -> None:
    metadata = {
        "sample_id": sample["id"],
        "sample_dir": sample["sample_dir"],
        "input_image": sample["image"],
        "prompt": sample["prompt"],
        "negative_prompt": sample["negative_prompt"],
        "frame_prompts": sample["frame_prompts"],
        "checkpoint": cfg["checkpoint"]["path"],
        "checkpoint_global_step": checkpoint_info["global_step"],
        "conditioning_mode": cfg["checkpoint"]["conditioning_mode"],
        "inference": cfg["inference"],
    }
    with open(sample_output / "meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


@torch.no_grad()
def infer_sample(
    pipe,
    sample: Dict[str, Any],
    cfg: Dict[str, Any],
    checkpoint_info: Dict[str, Any],
    sample_index: int,
) -> str:
    inference_cfg = cfg["inference"]
    output_root = Path(inference_cfg["output_dir"])
    sample_output = output_root / sample["id"]
    expected_slots = int(cfg["data"]["expected_num_slots"])

    if (
        not bool(inference_cfg["overwrite"])
        and outputs_complete(sample_output, expected_slots)
    ):
        return "skipped"

    ensure_dir(sample_output)
    prediction_dir = sample_output / "pred"
    ensure_dir(prediction_dir)

    input_image = Image.open(sample["image"]).convert("RGB")
    input_image.save(sample_output / "input.png")

    seed = int(inference_cfg["seed"])
    if bool(inference_cfg.get("seed_per_sample", True)):
        seed += sample_index

    frames = pipe(
        prompt=sample["prompt"],
        negative_prompt=sample["negative_prompt"],
        input_image=input_image,
        frame_prompts=sample["frame_prompts"],
        height=int(inference_cfg["height"]),
        width=int(inference_cfg["width"]),
        num_inference_steps=int(inference_cfg["num_inference_steps"]),
        cfg_scale=float(inference_cfg["cfg_scale"]),
        cfg_merge=bool(inference_cfg["cfg_merge"]),
        sigma_shift=float(inference_cfg["sigma_shift"]),
        seed=seed,
        rand_device=str(inference_cfg["rand_device"]),
        tiled=bool(inference_cfg["tiled"]),
        tile_size=tuple(inference_cfg["tile_size"]),
        tile_stride=tuple(inference_cfg["tile_stride"]),
        tea_cache_l1_thresh=inference_cfg.get("tea_cache_l1_thresh"),
        tea_cache_model_id=str(
            inference_cfg.get("tea_cache_model_id", "")
        ),
        framewise_decoding=bool(
            inference_cfg.get("framewise_decoding", False)
        ),
        output_type=str(inference_cfg.get("output_type", "quantized")),
    )

    if len(frames) != expected_slots:
        raise RuntimeError(
            f"{sample['id']}: expected {expected_slots} output frames, "
            f"got {len(frames)}"
        )

    for index, frame in enumerate(frames):
        frame.save(prediction_dir / f"{index:02d}.png")

    thumb_width = int(inference_cfg["contact_sheet_thumb_width"])
    save_contact_sheet(
        frames,
        sample_output / "pred_contact.jpg",
        prefix="pred",
        thumb_width=thumb_width,
    )
    if bool(inference_cfg["save_ground_truth"]):
        save_ground_truth(sample, sample_output, thumb_width)
    gt_frames = [
        Image.open(path).convert("RGB")
        for path in sample["target_keyframes"]
    ]
    save_compare_video(
        gt_frames=gt_frames,
        pred_frames=frames,
        sample_output=sample_output,
        square_size=int(inference_cfg.get("compare_square_size", 512)),
        fps=int(inference_cfg.get("compare_fps", 1)),
    )
    save_metadata(sample, sample_output, cfg, checkpoint_info)
    return "saved"


def validate_config(cfg: Dict[str, Any]) -> None:
    for section in ("experiment", "model", "checkpoint", "adapter", "data", "inference"):
        if section not in cfg:
            raise KeyError(f"Missing config section: {section}")

    mode = cfg["checkpoint"]["conditioning_mode"]
    if mode not in {"local_only", "dual_context"}:
        raise ValueError(f"Invalid conditioning mode: {mode}")

    if int(cfg["data"]["expected_num_slots"]) != 21:
        raise ValueError(
            "AgiBot non-AR inference currently expects exactly 21 slots."
        )

    checkpoint_path = Path(cfg["checkpoint"]["path"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    manifest_path = Path(cfg["data"]["val_manifest"])
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--step",
        default=None,
        help="Checkpoint step to evaluate, e.g. 1000 or step_001000. Use latest for newest.",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--local-rank", type=int, default=0)
    args = parser.parse_args()

    cfg = load_json(args.config)
    if args.step is not None:
        cfg["checkpoint"]["path"] = ""
        cfg["checkpoint"]["selector"] = str(args.step)
    checkpoint_path, checkpoint_label = resolve_checkpoint(cfg)
    cfg["checkpoint"]["path"] = str(checkpoint_path)
    cfg["inference"]["output_dir"] = str(resolve_output_dir(cfg, checkpoint_label))
    validate_config(cfg)

    distributed, rank, world_size, local_rank = init_distributed()
    inference_cfg = cfg["inference"]

    all_sample_dirs = read_manifest(cfg["data"]["val_manifest"])
    start = int(cfg["data"].get("start_index", 0))
    count = cfg["data"].get("num_samples")
    selected = all_sample_dirs[start:]
    if count is not None:
        selected = selected[: int(count)]
    if not selected:
        raise ValueError("No validation samples selected.")

    # Deterministic rank striding prevents duplicate work under torchrun.
    rank_items = [
        (global_index, sample_dir)
        for global_index, sample_dir in enumerate(selected, start=start)
        if (global_index - start) % world_size == rank
    ]

    output_root = Path(inference_cfg["output_dir"])
    if rank == 0:
        ensure_dir(output_root)
        with open(
            output_root / "inference_config.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    barrier(distributed, local_rank)

    print(
        f"[rank {rank}/{world_size}] device=cuda:{local_rank} "
        f"assigned_samples={len(rank_items)}",
        flush=True,
    )

    pipe = build_pipe(
        cfg,
        cfg["checkpoint"]["conditioning_mode"],
        local_rank,
    )
    pipe.load_models_to_device(["dit"])

    replaced = inject_lora(pipe.dit, cfg["adapter"])
    print(
        f"[rank {rank}] injected LoRA into {len(replaced)} linear layers",
        flush=True,
    )

    checkpoint_info = load_trainable_checkpoint(
        pipe.dit,
        cfg["checkpoint"]["path"],
    )
    print(
        f"[rank {rank}] checkpoint loaded={checkpoint_info['loaded']} "
        f"skipped={len(checkpoint_info['skipped'])} "
        f"global_step={checkpoint_info['global_step']}",
        flush=True,
    )

    for parameter in pipe.dit.parameters():
        parameter.requires_grad_(False)
    pipe.dit.eval()

    saved = 0
    skipped = 0
    failed = 0

    for local_position, (global_index, sample_dir) in enumerate(rank_items):
        try:
            sample = load_sample(
                sample_dir,
                expected_num_slots=int(cfg["data"]["expected_num_slots"]),
            )
            status = infer_sample(
                pipe,
                sample,
                cfg,
                checkpoint_info,
                sample_index=global_index,
            )
            if status == "saved":
                saved += 1
            else:
                skipped += 1
            print(
                f"[rank {rank}] [{local_position + 1}/{len(rank_items)}] "
                f"{status}: {sample['id']}",
                flush=True,
            )
        except Exception as exc:
            failed += 1
            print(
                f"[rank {rank}] [failed] sample_dir={sample_dir} "
                f"error={repr(exc)}",
                flush=True,
            )
            if bool(inference_cfg.get("fail_fast", False)):
                raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    totals = torch.tensor(
        [saved, skipped, failed],
        device=f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu",
        dtype=torch.long,
    )
    if distributed:
        torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)

    if rank == 0:
        print(
            f"[done] saved={int(totals[0])} "
            f"skipped={int(totals[1])} failed={int(totals[2])} "
            f"output={output_root}",
            flush=True,
        )

    barrier(distributed, local_rank)
    if distributed and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
