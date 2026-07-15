#!/usr/bin/env python3
"""Offline Wan VAE cache builder for keyframe sequences.

For each sample this stores target_latents, first_frame_latents, and the Wan I2V
conditioning tensor y. Full keyframe sequences are cached once; AR/non-AR
training can slice the latent sequence later without rerunning the VAE.

Supports torchrun and manual rank/world-size sharding.
"""
import os
import sys
sys.stdout.reconfigure(line_buffering=True)
import json
import math
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def load_sample_from_dir(sample_dir: str, data_root: Optional[str] = None) -> Dict:
    sample_dir = Path(sample_dir)
    ann_path = sample_dir / "annotation.json"
    ann = load_json(str(ann_path))

    if "frames" in ann:
        frames = ann["frames"]
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"{sample_dir}: missing non-empty frames list")
        keyframes = [resolve_path(sample_dir, frame["image"]) for frame in frames]
        frame_prompts = [
            frame.get("generated_prompt") or frame.get("frame_prompt_en_compiled") or frame.get("frame_prompt_template_en")
            for frame in frames
        ]
        image = keyframes[0]
    else:
        image = resolve_path(sample_dir, ann["image"])
        keyframes = [resolve_path(sample_dir, p) for p in ann["keyframes"]]
        frame_prompts = ann["frame_prompts"]

    if len(keyframes) != len(frame_prompts):
        raise ValueError(
            f"{sample_dir}: len(keyframes)={len(keyframes)} != len(frame_prompts)={len(frame_prompts)}"
        )

    return {
        "id": sample_cache_id(sample_dir, ann, data_root=data_root),
        "sample_dir": str(sample_dir),
        "image": image,
        "target_keyframes": keyframes,
        "frame_prompts": frame_prompts,
        "num_slots": len(frame_prompts),
    }


def maybe_slice_for_shard(items: List[str], rank: int, world_size: int) -> List[str]:
    if world_size <= 1:
        return items
    return items[rank::world_size]


def build_video_from_keyframes(keyframes: List[Image.Image], num_frames: int) -> List[Image.Image]:
    frames = []
    f = len(keyframes)
    for t in range(num_frames):
        idx = min(t // 4, f - 1)
        frames.append(keyframes[idx])
    return frames


def load_image(path: str, width: int, height: int) -> Image.Image:
    return Image.open(path).convert("RGB").resize((width, height))


def build_vae_pipe(cfg: Dict, device_override: str = None):
    model_cfg = cfg["model"]

    model_id = model_cfg.get("model_id", "Wan-AI/Wan2.1-I2V-14B-480P")
    model_path = model_cfg.get("model_path")
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
        vae_model_config = ModelConfig(
            path=str(Path(model_path) / "Wan2.1_VAE.pth"),
            **vram_config,
        )
    else:
        vae_model_config = ModelConfig(
            model_id=model_id,
            origin_file_pattern="Wan2.1_VAE.pth",
            **vram_config,
        )

    # 只加载 VAE
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=dtype,
        device=device,
        model_configs=[vae_model_config],
        tokenizer_config=None,
        redirect_common_files=False,
        vram_limit=vram_limit,
    )
    return pipe


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


@torch.no_grad()
def encode_batch_vae_only(
    pipe: WanVideoPipeline,
    batch_samples: List[Dict],
    infer_cfg: Dict,
    tiled: bool,
    device: str,
):
    height = infer_cfg.get("height", 480)
    width = infer_cfg.get("width", 832)
    tile_size = tuple(infer_cfg.get("tile_size", [30, 52]))
    tile_stride = tuple(infer_cfg.get("tile_stride", [15, 26]))

    num_slots = batch_samples[0]["num_slots"]
    num_frames = 1 + (num_slots - 1) * 4

    # ---------- first frame ----------
    first_frame_tensors = []
    for sample in batch_samples:
        first_img = load_image(sample["image"], width=width, height=height)
        first_video_1f = pipe.preprocess_video([first_img])   # [1, 3, 1, H, W]
        first_frame_tensors.append(first_video_1f)

    first_frame_tensor = torch.cat(first_frame_tensors, dim=0)

    # ---------- target video ----------
    target_video_tensors = []
    for sample in batch_samples:
        keyframes = [load_image(p, width=width, height=height) for p in sample["target_keyframes"]]
        frames = build_video_from_keyframes(keyframes, num_frames)
        video_tensor = pipe.preprocess_video(frames)          # [1, 3, F, H, W]
        target_video_tensors.append(video_tensor)

    target_video_tensor = torch.cat(target_video_tensors, dim=0)

    # ---------- I2V condition y input ----------
    # Match WanVideoUnit_ImageEmbedderVAE:
    # first frame is known, future frames are zeros.
    y_input_tensors = []
    for sample in batch_samples:
        first_img = load_image(sample["image"], width=width, height=height)
        image = pipe.preprocess_image(first_img)  # [1, 3, H, W]
        vae_input = torch.concat(
            [
                image.transpose(0, 1),  # [3, 1, H, W]
                torch.zeros(
                    3,
                    num_frames - 1,
                    height,
                    width,
                    dtype=image.dtype,
                    device=image.device,
                ),
            ],
            dim=1,
        )  # [3, F, H, W]
        y_input_tensors.append(vae_input)

    y_input_tensor = torch.stack(y_input_tensors, dim=0)  # [B, 3, F, H, W]

    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    # 编 first_frame_latents
    first_frame_latents = pipe.vae.encode(
        first_frame_tensor,
        device=pipe.device,
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )
    first_frame_latents = first_frame_latents.to(dtype=pipe.torch_dtype, device="cpu").detach()

    cuda_mem("after first_frame_latents", device)

    # 编 target_latents
    target_latents = pipe.vae.encode(
        target_video_tensor,
        device=pipe.device,
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )
    target_latents = target_latents.to(dtype=pipe.torch_dtype, device="cpu").detach()

    cuda_mem("after target_latents", device)

    # ---------- encode y ----------
    y_latents = pipe.vae.encode(
        y_input_tensor,
        device=pipe.device,
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )
    y_latents = y_latents.to(dtype=pipe.torch_dtype, device=pipe.device)

    # Match WanVideoUnit_ImageEmbedderVAE mask construction.
    bsz = len(batch_samples)
    h8, w8 = height // 8, width // 8
    msk = torch.ones(
        bsz,
        num_frames,
        h8,
        w8,
        dtype=pipe.torch_dtype,
        device=pipe.device,
    )
    msk[:, 1:] = 0
    msk = torch.cat(
        [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]],
        dim=1,
    )
    msk = msk.view(bsz, msk.shape[1] // 4, 4, h8, w8)
    msk = msk.transpose(1, 2)  # [B, 4, T, h, w]

    y = torch.cat([msk, y_latents], dim=1)
    y = y.to(dtype=pipe.torch_dtype, device="cpu").detach()

    cuda_mem("after y", device)

    del first_frame_tensor, target_video_tensor, y_input_tensor, y_latents, msk
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    return {
        "num_slots": num_slots,
        "num_frames": num_frames,
        "first_frame_latents": first_frame_latents,   # [B, C, 1, h, w]
        "target_latents": target_latents,             # [B, C, F', h, w]
        "y": y,                                       # [B, C_y, F', h, w]
        "fuse_vae_embedding_in_latents": False,
    }


def save_one_pt(sample: Dict, out_dir: str, encoded: Dict, idx_in_batch: int):
    sample_id = sample["id"]
    out_path = os.path.join(out_dir, f"{sample_id}.pt")
    tmp_path = out_path + ".tmp"
    ensure_dir(os.path.dirname(out_path))

    obj = {
        "cache_version": "wan_i2v_vae_v1",
        "sample_id": sample_id,
        "sample_dir": sample["sample_dir"],
        "image": sample["image"],
        "target_keyframes": sample["target_keyframes"],
        "num_slots": encoded["num_slots"],
        "num_frames": encoded["num_frames"],
        "first_frame_latents": encoded["first_frame_latents"][idx_in_batch:idx_in_batch + 1].clone(),
        "target_latents": encoded["target_latents"][idx_in_batch:idx_in_batch + 1].clone(),
        "y": encoded["y"][idx_in_batch:idx_in_batch + 1].clone(),
        "fuse_vae_embedding_in_latents": bool(encoded.get("fuse_vae_embedding_in_latents", False)),
    }

    torch.save(obj, tmp_path)
    os.replace(tmp_path, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--no-tiled", action="store_true")

    parser.add_argument("--print_every", type=int, default=1)
    args = parser.parse_args()

    cfg = load_json(args.config)
    infer_cfg = cfg.get("infer", cfg.get("inference", {}))
    data_cfg = cfg.get("data", {})
    manifest = args.manifest or data_cfg.get("train_manifest")
    out_dir = args.out_dir or data_cfg.get("vae_cache_dir")
    data_root = data_cfg.get("data_root")
    if not manifest:
        raise ValueError("Missing manifest. Pass --manifest or set data.train_manifest in config.")
    if not out_dir:
        raise ValueError("Missing output dir. Pass --out_dir or set data.vae_cache_dir in config.")

    local_rank = int(os.environ.get("LOCAL_RANK", args.rank))
    rank = int(os.environ.get("RANK", args.rank))
    world_size = int(os.environ.get("WORLD_SIZE", args.world_size))

    if args.device is not None:
        device = args.device
    elif torch.cuda.is_available():
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"

    if str(device).startswith("cuda"):
        torch.cuda.set_device(device)

    tiled_default = bool(infer_cfg.get("tiled", True))
    if args.tiled:
        tiled = True
    elif args.no_tiled:
        tiled = False
    else:
        tiled = tiled_default

    ensure_dir(out_dir)

    all_dirs = [resolve_manifest_item(item, data_root=data_root) for item in read_manifest(manifest)]
    shard_dirs = maybe_slice_for_shard(all_dirs, rank, world_size)
    samples = [load_sample_from_dir(d, data_root=data_root) for d in shard_dirs]
    buckets = bucket_by_num_slots(samples)

    print(f"[mode] vae_latent_cache")
    print(f"[cfg] device={device} batch_size={args.batch_size} tiled={tiled}")
    print(f"[data] total={len(all_dirs)} shard={len(shard_dirs)} rank={rank}/{world_size}")
    print(f"[data] manifest={manifest}")
    print(f"[data] out_dir={out_dir}")

    pipe = build_vae_pipe(cfg, device_override=device)
    pipe.load_models_to_device(["vae"])

    total_ok = 0
    total_skip = 0
    total_fail = 0

    for num_slots in sorted(buckets.keys()):
        bucket = buckets[num_slots]
        num_frames = 1 + (num_slots - 1) * 4
        print(f"\n[bucket] num_slots={num_slots}, num_frames={num_frames}, samples={len(bucket)}")

        # 过滤已存在
        work_bucket = []
        for s in bucket:
            out_path = os.path.join(out_dir, f"{s['id']}.pt")
            if os.path.exists(out_path) and not args.overwrite:
                total_skip += 1
            else:
                work_bucket.append(s)

        if len(work_bucket) == 0:
            print("[bucket] all skipped")
            continue

        for start in range(0, len(work_bucket), args.batch_size):
            batch = work_bucket[start:start + args.batch_size]
            t0 = time.time()

            try:
                encoded = encode_batch_vae_only(
                    pipe=pipe,
                    batch_samples=batch,
                    infer_cfg=infer_cfg,
                    tiled=tiled,
                    device=device,
                )

                for j, sample in enumerate(batch):
                    save_one_pt(sample, out_dir, encoded, j)
                    total_ok += 1

                dt = time.time() - t0
                speed = len(batch) / max(dt, 1e-8)

                if ((start // args.batch_size) % args.print_every) == 0:
                    print(
                        f"[ok] bucket={num_slots} batch={len(batch)} "
                        f"saved={start + len(batch)}/{len(work_bucket)} "
                        f"dt={dt:.2f}s total_ok={total_ok} speed={speed:.3f} samples/s"
                    )

                del encoded
                if str(device).startswith("cuda"):
                    torch.cuda.empty_cache()

            except Exception as e:
                total_fail += len(batch)
                print(f"[fail] bucket={num_slots} start={start} bs={len(batch)} err={repr(e)}")

                # 如果 batch>1 爆了，提示缩小 batch
                if len(batch) > 1:
                    print("[hint] try smaller --batch_size or use --tiled")
                if str(device).startswith("cuda"):
                    torch.cuda.empty_cache()

    print(f"\n[done] ok={total_ok} skip={total_skip} fail={total_fail}")


if __name__ == "__main__":
    main()
