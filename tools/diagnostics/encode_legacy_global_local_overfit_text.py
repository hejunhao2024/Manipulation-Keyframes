#!/usr/bin/env python3
"""Build a one-off legacy-style local text cache for overfit diagnostics.

This does not modify dataset annotations. It writes a normal local-only cache
whose per-slot prompt is:

    Current keyframe: <local prompt>
    Global task and scene: <summary of all local prompts>
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffsynth.core import ModelConfig
from diffsynth.pipelines.keyframe_dual_context import WanVideoPipeline


def read_manifest(path: Path):
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def resolve_item(item: str, data_root: Path) -> Path:
    p = Path(item)
    return p if p.is_absolute() else data_root / p


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sample_cache_id(sample_dir: Path, ann: dict, data_root: Path) -> str:
    for key in ("id", "sample_id", "sample_short"):
        value = ann.get(key)
        if value:
            return str(value).strip("/")
    try:
        return sample_dir.relative_to(data_root).as_posix()
    except ValueError:
        return sample_dir.name


def resolve_path(base_dir: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else base_dir / p


def load_sample(sample_dir: Path, data_root: Path):
    ann = load_json(sample_dir / "annotation.json")
    frames = ann["frames"]
    frame_prompts = [
        frame.get("generated_prompt")
        or frame.get("frame_prompt_en_compiled")
        or frame.get("frame_prompt_template_en")
        for frame in frames
    ]
    if any(not prompt for prompt in frame_prompts):
        raise RuntimeError(f"{sample_dir}: missing frame prompt")
    return {
        "id": sample_cache_id(sample_dir, ann, data_root),
        "sample_dir": str(sample_dir),
        "image": resolve_path(sample_dir, frames[0]["image"]),
        "frame_prompts": frame_prompts,
        "negative_prompt": ann.get("negative_prompt", ""),
    }


def legacy_summary(frame_prompts):
    compact = " ".join(prompt.strip() for prompt in frame_prompts if prompt.strip())
    return (
        "A robot manipulation keyframe sequence in a grocery shelf and shopping "
        f"cart scene. The visible sequence is: {compact}"
    )


@torch.no_grad()
def encode_prompts(pipe, prompts):
    pipe.load_models_to_device(["text_encoder"])
    ids, mask = pipe.tokenizer(prompts, return_mask=True, add_special_tokens=True)
    ids = ids.to(pipe.device)
    mask = mask.to(pipe.device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    emb = pipe.text_encoder(ids, mask)
    for i, length in enumerate(seq_lens):
        emb[i, length:] = 0
    return emb.detach().cpu(), mask.detach().to(dtype=torch.bool).cpu()


@torch.no_grad()
def encode_clip(pipe, image_path: Path, width: int, height: int):
    if pipe.image_encoder is None:
        return None
    pipe.load_models_to_device(["image_encoder"])
    image = Image.open(image_path).convert("RGB").resize((width, height))
    image = pipe.preprocess_image(image).to(pipe.device)
    clip = pipe.image_encoder.encode_image([image])
    return clip.to(dtype=pipe.torch_dtype).detach().cpu()


def build_pipe(model_path: Path, device: str, vram_limit_gb: float):
    dtype = torch.bfloat16
    vram = {
        "offload_dtype": dtype,
        "offload_device": "cpu",
        "onload_dtype": dtype,
        "onload_device": "cpu",
        "computation_dtype": dtype,
        "computation_device": device,
    }
    return WanVideoPipeline.from_pretrained(
        torch_dtype=dtype,
        device=device,
        model_configs=[
            ModelConfig(
                path=str(model_path / "models_t5_umt5-xxl-enc-bf16.pth"),
                **vram,
            ),
            ModelConfig(
                path=str(model_path / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
                **vram,
            ),
        ],
        tokenizer_config=ModelConfig(path=str(model_path / "google/umt5-xxl")),
        redirect_common_files=False,
        vram_limit=vram_limit_gb,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", default="/media/datasets/yumi/hjh")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--vram-limit-gb", type=float, default=76)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = [
        load_sample(resolve_item(item, data_root), data_root)
        for item in read_manifest(Path(args.manifest))
    ]
    pipe = build_pipe(Path(args.model_path), args.device, args.vram_limit_gb)

    for sample in samples:
        out_path = out_dir / f"{sample['id']}.pt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {sample['id']} {out_path}")
            continue

        summary = legacy_summary(sample["frame_prompts"])
        slot_prompts = [
            f"Current keyframe: {prompt}\nGlobal task and scene: {summary}"
            for prompt in sample["frame_prompts"]
        ]
        context, local_mask = encode_prompts(pipe, slot_prompts)
        clip_feature = encode_clip(pipe, sample["image"], args.width, args.height)

        obj = {
            "cache_version": "legacy_global_local_local_only_v1",
            "sample_id": sample["id"],
            "sample_dir": sample["sample_dir"],
            "image": str(sample["image"]),
            "prompt": summary,
            "negative_prompt": sample["negative_prompt"],
            "frame_prompts": slot_prompts,
            "original_frame_prompts": sample["frame_prompts"],
            "num_slots": len(slot_prompts),
            "num_frames": 1 + (len(slot_prompts) - 1) * 4,
            "context": context.unsqueeze(0).contiguous(),
            "local_attention_mask": local_mask.unsqueeze(0).contiguous(),
            "clip_feature": None if clip_feature is None else clip_feature.contiguous(),
        }
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        torch.save(obj, tmp)
        tmp.replace(out_path)
        print(
            f"[ok] {sample['id']} context={tuple(obj['context'].shape)} "
            f"clip={None if clip_feature is None else tuple(clip_feature.shape)}"
        )
        print("[summary]", summary[:500])


if __name__ == "__main__":
    main()
