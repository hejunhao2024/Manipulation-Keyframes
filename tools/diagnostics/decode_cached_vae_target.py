#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffsynth.core import ModelConfig
from diffsynth.pipelines.keyframe_local_context import WanVideoPipeline


def save_contact(frames, path: Path, thumb_width: int = 192):
    label_h = 24
    thumbs = []
    for frame in frames:
        frame = frame.convert("RGB")
        h = round(frame.height * thumb_width / frame.width)
        thumbs.append(frame.resize((thumb_width, h)))
    cols = min(7, len(thumbs))
    rows = (len(thumbs) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * thumb_width, rows * (thumbs[0].height + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, thumb in enumerate(thumbs):
        x = (i % cols) * thumb_width
        y = (i // cols) * (thumb.height + label_h)
        draw.text((x + 4, y + 4), f"dec {i:02d}", fill=(0, 0, 0))
        canvas.paste(thumb, (x, y + label_h))
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-cache", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tiled", action="store_true")
    args = parser.parse_args()

    dtype = torch.bfloat16
    vram = {
        "offload_dtype": dtype,
        "offload_device": "cpu",
        "onload_dtype": dtype,
        "onload_device": "cpu",
        "computation_dtype": dtype,
        "computation_device": args.device,
    }
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=dtype,
        device=args.device,
        model_configs=[
            ModelConfig(path=str(Path(args.model_path) / "Wan2.1_VAE.pth"), **vram)
        ],
        tokenizer_config=None,
        redirect_common_files=False,
        vram_limit=76,
    )
    pipe.load_models_to_device(["vae"])
    obj = torch.load(args.vae_cache, map_location="cpu", weights_only=False)
    latents = obj["target_latents"].to(device=pipe.device, dtype=pipe.torch_dtype)
    video = pipe.vae.decode(
        latents,
        device=pipe.device,
        tiled=args.tiled,
        tile_size=(30, 40),
        tile_stride=(15, 20),
    )
    frames = pipe.vae_output_to_video(video)
    keyframes = frames[::4]

    out = Path(args.output_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "keyframes").mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(out / "frames" / f"{i:02d}.png")
    for i, frame in enumerate(keyframes):
        frame.save(out / "keyframes" / f"{i:02d}.png")
    save_contact(keyframes, out / "decoded_keyframes_contact.jpg")
    print("decoded_frames", len(frames))
    print("decoded_keyframes", len(keyframes))
    print("output", out)


if __name__ == "__main__":
    main()
