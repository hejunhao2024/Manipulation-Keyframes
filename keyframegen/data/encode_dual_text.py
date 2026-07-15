#!/usr/bin/env python3
"""Encode global and frame-local text into two independent context streams.

Saved per sample:
  global_context:        [1, 512, D]
  local_context:         [1, T, 512, D]
  global_attention_mask: [1, 512]
  local_attention_mask:  [1, T, 512]
  clip_feature:          optional first-frame CLIP feature

Supports torchrun and manual rank/world-size sharding.
"""
import os
import sys
sys.stdout.reconfigure(line_buffering=True)
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
import torch

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from diffsynth.pipelines.keyframe_dual_context import WanVideoPipeline
from diffsynth.core import ModelConfig


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_manifest(path: str) -> List[str]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            items.append(line)
    return items


def resolve_path(base_dir: Path, p: str) -> str:
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    return str(base_dir / pp)


def resolve_manifest_item(item: str, data_root: Optional[str] = None) -> str:
    path = Path(item)
    if path.is_absolute():
        return str(path)
    if data_root:
        return str(Path(data_root) / path)
    return str(path)


def sample_cache_id(sample_dir: Path, ann: Dict, data_root: Optional[str] = None) -> str:
    for key in ("id", "sample_id", "sample_short"):
        value = ann.get(key)
        if value:
            return str(value).strip("/")
    if data_root:
        try:
            return sample_dir.relative_to(Path(data_root)).as_posix()
        except ValueError:
            pass
    return sample_dir.name


def load_sample_from_dir(
    sample_dir: str,
    global_prompt_field: str = "global_prompt_long",
    data_root: Optional[str] = None,
    require_global_prompt: bool = True,
) -> Dict:
    sample_dir = Path(sample_dir)
    ann_path = sample_dir / "annotation.json"
    ann = load_json(str(ann_path))

    if "frames" in ann:
        frames = ann["frames"]
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"{sample_dir}: missing non-empty frames list")
        image = resolve_path(sample_dir, frames[0]["image"])
        frame_prompts = [
            frame.get("generated_prompt") or frame.get("frame_prompt_en_compiled") or frame.get("frame_prompt_template_en")
            for frame in frames
        ]
        if any(not prompt for prompt in frame_prompts):
            raise KeyError(f"{sample_dir}: every frame needs generated_prompt or frame_prompt_en_compiled")
    else:
        image = resolve_path(sample_dir, ann["image"])
        frame_prompts = ann["frame_prompts"]

    global_prompt = ann.get(global_prompt_field) or ann.get("system_prompt") or ann.get("global_prompt") or ann.get("prompt")
    if require_global_prompt and not global_prompt:
        raise KeyError(f"{sample_dir}: missing global prompt field {global_prompt_field!r} and fallback 'prompt'")

    return {
        "id": sample_cache_id(sample_dir, ann, data_root=data_root),
        "sample_dir": str(sample_dir),
        "image": image,
        "prompt": global_prompt or "",
        "global_prompt_field": global_prompt_field,
        "negative_prompt": ann.get("negative_prompt", ""),
        "frame_prompts": frame_prompts,
        "num_slots": len(frame_prompts),
    }


def maybe_slice_for_shard(items: List[str], rank: int, world_size: int) -> List[str]:
    if world_size <= 1:
        return items
    return items[rank::world_size]


def bucket_by_num_slots(samples: List[Dict]) -> Dict[int, List[Dict]]:
    buckets = {}
    for s in samples:
        buckets.setdefault(s["num_slots"], []).append(s)
    return buckets


def cuda_mem(tag: str, device: str):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)
        alloc = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        peak = torch.cuda.max_memory_allocated(device) / 1024**3
        print(f"[mem] {tag}: alloc={alloc:.2f}GB reserved={reserved:.2f}GB peak={peak:.2f}GB")


def build_text_clip_pipe(cfg: Dict, device_override: str = None, load_clip: bool = True):
    model_cfg = cfg["model"]

    model_id = model_cfg.get("model_id", "Wan-AI/Wan2.1-I2V-14B-480P")
    model_path = model_cfg.get("model_path")
    tokenizer_path = model_cfg["tokenizer_path"]
    device = device_override or model_cfg.get("device", "cuda")
    dtype = torch.bfloat16

    vram_limit = model_cfg.get("vram_limit_gb", None)
    if vram_limit is None and str(device).startswith("cuda"):
        try:
            vram_limit = torch.cuda.mem_get_info(device)[1] / (1024 ** 3) - 2
        except Exception:
            vram_limit = None

    vram_config = {
        "offload_dtype": dtype,
        "offload_device": model_cfg.get("offload_device", "cpu"),
        "onload_dtype": dtype,
        "onload_device": model_cfg.get("onload_device", "cpu"),
        "computation_dtype": dtype,
        "computation_device": device,
    }

    if model_path:
        text_model_config = ModelConfig(
            path=str(Path(model_path) / "models_t5_umt5-xxl-enc-bf16.pth"),
            **vram_config,
        )
    else:
        text_model_config = ModelConfig(
            model_id=model_id,
            origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
            **vram_config,
        )

    model_configs = [text_model_config]
    if load_clip:
        if model_path:
            image_model_config = ModelConfig(
                path=str(Path(model_path) / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
                **vram_config,
            )
        else:
            image_model_config = ModelConfig(
                model_id=model_id,
                origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                **vram_config,
            )
        model_configs.append(image_model_config)

    # 只加载 text encoder + image encoder，不加载 DiT，不加载 VAE
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=dtype,
        device=device,
        model_configs=model_configs,
        tokenizer_config=ModelConfig(path=tokenizer_path),
        redirect_common_files=False,
        vram_limit=vram_limit,
    )

    return pipe


def make_global_prompts(samples: List[Dict]) -> List[str]:
    """One global task/scene prompt per sample."""
    return [s["prompt"] for s in samples]


def make_local_prompts(samples: List[Dict]) -> List[str]:
    """Flatten frame-wise local prompts in sample-major order."""
    flat = []
    for s in samples:
        flat.extend(s["frame_prompts"])
    return flat


@torch.no_grad()
def encode_prompts_batched(
    pipe: WanVideoPipeline,
    prompts_all: List[str],
    text_batch_size: int,
):
    """
    Returns:
      embeddings: [M, 512, D] on CPU
      masks:      [M, 512] on CPU
    Padding embeddings are explicitly zeroed, matching the previous encoder.
    """
    all_embs = []
    all_masks = []

    pipe.load_models_to_device(["text_encoder"])

    for start in range(0, len(prompts_all), text_batch_size):
        prompts = prompts_all[start:start + text_batch_size]

        ids, mask = pipe.tokenizer(
            prompts,
            return_mask=True,
            add_special_tokens=True,
        )
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)

        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = pipe.text_encoder(ids, mask)

        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0

        all_embs.append(prompt_emb.detach().cpu())
        all_masks.append(mask.detach().to(dtype=torch.bool).cpu())

        del ids, mask, prompt_emb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.cat(all_embs, dim=0), torch.cat(all_masks, dim=0)


@torch.no_grad()
def encode_clip_batched(pipe: WanVideoPipeline, samples: List[Dict], infer_cfg: Dict, clip_batch_size: int):
    if pipe.image_encoder is None:
        return [None for _ in samples]

    height = infer_cfg.get("height", 480)
    width = infer_cfg.get("width", 832)

    pipe.load_models_to_device(["image_encoder"])

    outputs = []

    for start in range(0, len(samples), clip_batch_size):
        batch = samples[start:start + clip_batch_size]

        images = []
        for s in batch:
            img = Image.open(s["image"]).convert("RGB").resize((width, height))
            img = pipe.preprocess_image(img).to(pipe.device)
            images.append(img)

        clip_context = pipe.image_encoder.encode_image(images)
        clip_context = clip_context.to(dtype=pipe.torch_dtype).detach().cpu()

        for i in range(len(batch)):
            outputs.append(clip_context[i:i + 1].clone())

        del images, clip_context
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return outputs


def save_one_text_pt(
    sample: Dict,
    out_dir: str,
    global_context: torch.Tensor,
    local_context: torch.Tensor,
    global_attention_mask: torch.Tensor,
    local_attention_mask: torch.Tensor,
    clip_feature,
):
    """
    Saved shapes per sample:
      global_context:         [1, 512, D]
      local_context:          [1, T, 512, D]
      global_attention_mask:  [1, 512]
      local_attention_mask:   [1, T, 512]
      clip_feature:           unchanged image feature
    """
    sample_id = sample["id"]
    out_path = os.path.join(out_dir, f"{sample_id}.pt")
    tmp_path = out_path + ".tmp"
    ensure_dir(os.path.dirname(out_path))

    num_slots = sample["num_slots"]
    num_frames = 1 + (num_slots - 1) * 4

    obj = {
        "cache_version": "global_local_v1",
        "sample_id": sample_id,
        "sample_dir": sample["sample_dir"],
        "image": sample["image"],
        "prompt": sample["prompt"],
        "negative_prompt": sample["negative_prompt"],
        "frame_prompts": sample["frame_prompts"],
        "num_slots": num_slots,
        "num_frames": num_frames,
        "global_context": global_context.contiguous().clone(),
        "local_context": local_context.contiguous().clone(),
        "global_attention_mask": global_attention_mask.contiguous().clone(),
        "local_attention_mask": local_attention_mask.contiguous().clone(),
        "clip_feature": None if clip_feature is None else clip_feature.contiguous().clone(),
    }

    torch.save(obj, tmp_path)
    os.replace(tmp_path, out_path)


def save_one_local_text_pt(
    sample: Dict,
    out_dir: str,
    context: torch.Tensor,
    local_attention_mask: torch.Tensor,
    clip_feature,
):
    sample_id = sample["id"]
    out_path = os.path.join(out_dir, f"{sample_id}.pt")
    tmp_path = out_path + ".tmp"
    ensure_dir(os.path.dirname(out_path))

    num_slots = sample["num_slots"]
    num_frames = 1 + (num_slots - 1) * 4

    obj = {
        "cache_version": "local_only_v2_from_global_local_encoder",
        "sample_id": sample_id,
        "sample_dir": sample["sample_dir"],
        "image": sample["image"],
        "prompt": sample["prompt"],
        "global_prompt_field": sample.get("global_prompt_field", "global_prompt_long"),
        "negative_prompt": sample["negative_prompt"],
        "frame_prompts": sample["frame_prompts"],
        "num_slots": num_slots,
        "num_frames": num_frames,
        "context": context.contiguous().clone(),
        "local_attention_mask": local_attention_mask.contiguous().clone(),
        "clip_feature": None if clip_feature is None else clip_feature.contiguous().clone(),
    }

    torch.save(obj, tmp_path)
    os.replace(tmp_path, out_path)


@torch.no_grad()
def process_bucket(
    pipe: WanVideoPipeline,
    bucket: List[Dict],
    local_out_dir: str,
    dual_out_dir: str,
    infer_cfg: Dict,
    batch_size: int,
    text_batch_size: int,
    clip_batch_size: int,
    encode_clip: bool,
    encode_global: bool,
    overwrite: bool,
    device: str,
):
    work = []
    skip = 0

    for s in bucket:
        local_path = os.path.join(local_out_dir, f"{s['id']}.pt")
        dual_path = os.path.join(dual_out_dir, f"{s['id']}.pt") if encode_global and dual_out_dir else None
        local_done = os.path.exists(local_path)
        dual_done = True if dual_path is None else os.path.exists(dual_path)
        if local_done and dual_done and not overwrite:
            skip += 1
        else:
            work.append(s)

    if len(work) == 0:
        return 0, skip, 0

    ok = 0
    fail = 0

    for start in range(0, len(work), batch_size):
        batch = work[start:start + batch_size]
        t0 = time.time()

        try:
            num_slots = batch[0]["num_slots"]
            for s in batch:
                assert s["num_slots"] == num_slots

            if str(device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(device)

            # Encode local frame text, and optionally the legacy global stream.
            # local_context_flat:  [B*T, 512, D]
            local_context_flat, local_mask_flat = encode_prompts_batched(
                pipe=pipe,
                prompts_all=make_local_prompts(batch),
                text_batch_size=text_batch_size,
            )
            if encode_global:
                global_context_all, global_mask_all = encode_prompts_batched(
                    pipe=pipe,
                    prompts_all=make_global_prompts(batch),
                    text_batch_size=text_batch_size,
                )
            else:
                global_context_all = None
                global_mask_all = None

            # [B, T, 512, D], [B, T, 512]
            local_context_all = local_context_flat.view(
                len(batch), num_slots, *local_context_flat.shape[1:]
            )
            local_mask_all = local_mask_flat.view(
                len(batch), num_slots, *local_mask_flat.shape[1:]
            )

            cuda_mem("after text", device)

            if encode_clip:
                clip_features = encode_clip_batched(
                    pipe=pipe,
                    samples=batch,
                    infer_cfg=infer_cfg,
                    clip_batch_size=clip_batch_size,
                )
            else:
                clip_features = [None for _ in batch]

            cuda_mem("after clip", device)

            for j, s in enumerate(batch):
                if encode_global:
                    save_one_text_pt(
                        sample=s,
                        out_dir=dual_out_dir,
                        global_context=global_context_all[j:j + 1],
                        local_context=local_context_all[j:j + 1],
                        global_attention_mask=global_mask_all[j:j + 1],
                        local_attention_mask=local_mask_all[j:j + 1],
                        clip_feature=clip_features[j],
                    )
                save_one_local_text_pt(
                    sample=s,
                    out_dir=local_out_dir,
                    context=local_context_all[j:j + 1],
                    local_attention_mask=local_mask_all[j:j + 1],
                    clip_feature=clip_features[j],
                )
                ok += 1

            dt = time.time() - t0
            speed = len(batch) / max(dt, 1e-8)
            print(
                f"[ok] slots={num_slots} batch={len(batch)} "
                f"saved={start + len(batch)}/{len(work)} "
                f"dt={dt:.2f}s speed={speed:.3f} samples/s total_ok={ok}"
            )

            del (
                local_context_flat,
                local_mask_flat,
                local_context_all,
                local_mask_all,
                clip_features,
            )
            if global_context_all is not None:
                del global_context_all, global_mask_all
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            fail += len(batch)
            print(f"[fail] start={start} bs={len(batch)} err={repr(e)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return ok, skip, fail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None, help="Backward-compatible alias for --dual_out_dir.")
    parser.add_argument("--local_out_dir", type=str, default=None)
    parser.add_argument("--dual_out_dir", type=str, default=None)

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--text_batch_size", type=int, default=64)
    parser.add_argument("--clip_batch_size", type=int, default=16)

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_clip", action="store_true")

    args = parser.parse_args()

    cfg = load_json(args.config)
    infer_cfg = cfg.get("infer", cfg.get("inference", {}))
    data_cfg = cfg.get("data", {})
    manifest = args.manifest or data_cfg.get("train_manifest")
    local_out_dir = args.local_out_dir or data_cfg.get("local_text_cache_dir")
    dual_out_dir = args.dual_out_dir or args.out_dir or data_cfg.get("dual_text_cache_dir") or data_cfg.get("text_cache_dir")
    global_prompt_field = data_cfg.get("global_prompt_field", "global_prompt_long")
    data_root = data_cfg.get("data_root")
    encode_global = bool(data_cfg.get("encode_global", False))
    if not manifest:
        raise ValueError("Missing manifest. Pass --manifest or set data.train_manifest in config.")
    if not local_out_dir:
        raise ValueError("Missing local output dir. Pass --local_out_dir or set data.local_text_cache_dir in config.")
    if encode_global and not dual_out_dir:
        raise ValueError("Missing dual output dir. Pass --dual_out_dir/--out_dir or set data.dual_text_cache_dir in config.")

    # Supports both:
    #   1) torchrun --nproc_per_node=6 ...
    #   2) manual --rank / --world_size launches
    env_local_rank = int(os.environ.get("LOCAL_RANK", args.rank))
    env_rank = int(os.environ.get("RANK", args.rank))
    env_world_size = int(os.environ.get("WORLD_SIZE", args.world_size))

    if args.device is not None:
        device = args.device
    elif torch.cuda.is_available():
        device = f"cuda:{env_local_rank}"
    else:
        device = "cpu"

    if str(device).startswith("cuda"):
        torch.cuda.set_device(device)

    rank = env_rank
    world_size = env_world_size

    ensure_dir(local_out_dir)
    ensure_dir(dual_out_dir)

    all_dirs = [resolve_manifest_item(item, data_root=data_root) for item in read_manifest(manifest)]
    shard_dirs = maybe_slice_for_shard(all_dirs, rank, world_size)
    samples = [
        load_sample_from_dir(
            d,
            global_prompt_field=global_prompt_field,
            data_root=data_root,
            require_global_prompt=encode_global,
        )
        for d in shard_dirs
    ]

    buckets = bucket_by_num_slots(samples)

    print(f"[mode] {'global_local_text_to_local_and_dual_caches' if encode_global else 'local_text_cache'}")
    print(f"[cfg] device={device}")
    print(f"[cfg] batch_size={args.batch_size} text_batch_size={args.text_batch_size} clip_batch_size={args.clip_batch_size}")
    print(f"[cfg] encode_clip={not args.no_clip}")
    print(f"[cfg] encode_global={encode_global}")
    print(f"[cfg] global_prompt_field={global_prompt_field}")
    print(f"[data] total={len(all_dirs)} shard={len(shard_dirs)} rank={rank}/{world_size}")
    print(f"[data] manifest={manifest}")
    print(f"[data] local_out_dir={local_out_dir}")
    if encode_global:
        print(f"[data] dual_out_dir={dual_out_dir}")

    pipe = build_text_clip_pipe(cfg, device_override=device, load_clip=not args.no_clip)

    total_ok = 0
    total_skip = 0
    total_fail = 0

    for num_slots in sorted(buckets.keys()):
        bucket = buckets[num_slots]
        num_frames = 1 + (num_slots - 1) * 4
        print(f"\n[bucket] num_slots={num_slots}, num_frames={num_frames}, samples={len(bucket)}")

        ok, skip, fail = process_bucket(
            pipe=pipe,
            bucket=bucket,
            local_out_dir=local_out_dir,
            dual_out_dir=dual_out_dir,
            infer_cfg=infer_cfg,
            batch_size=args.batch_size,
            text_batch_size=args.text_batch_size,
            clip_batch_size=args.clip_batch_size,
            encode_clip=not args.no_clip,
            encode_global=encode_global,
            overwrite=args.overwrite,
            device=device,
        )

        total_ok += ok
        total_skip += skip
        total_fail += fail

    print(f"\n[done] ok={total_ok} skip={total_skip} fail={total_fail}")


if __name__ == "__main__":
    main()
