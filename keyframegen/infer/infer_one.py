import os
import sys
import json
import argparse
from pathlib import Path
from PIL import Image
import torch

ROOT = "/home/hejunhao-20251119/mnt/work/Manipulation-Keyframes"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from diffsynth.pipelines.keyframe import WanVideoPipeline
from diffsynth.core import ModelConfig


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def build_pipe(cfg):
    model_cfg = cfg["model"]

    model_id = model_cfg["model_id"]
    tokenizer_path = model_cfg["tokenizer_path"]

    dtype = torch.bfloat16
    device = model_cfg.get("device", "cuda")

    # 尽量沿用你已经跑通的低显存配置
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


def run_one(pipe, cfg):
    input_cfg = cfg["input"]
    output_cfg = cfg["output"]
    infer_cfg = cfg["infer"]

    image_path = input_cfg["image"]
    prompt = input_cfg["prompt"]
    negative_prompt = input_cfg.get("negative_prompt", "")
    frame_prompts = input_cfg["frame_prompts"]

    out_dir = output_cfg["out_dir"]
    ensure_dir(out_dir)

    input_image = Image.open(image_path).convert("RGB")

    frames = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        input_image=input_image,
        frame_prompts=frame_prompts,
        height=infer_cfg.get("height", 480),
        width=infer_cfg.get("width", 832),
        num_inference_steps=infer_cfg.get("num_inference_steps", 10),
        cfg_scale=infer_cfg.get("cfg_scale", 5.0),
        cfg_merge=infer_cfg.get("cfg_merge", False),
        sigma_shift=infer_cfg.get("sigma_shift", 5.0),
        seed=infer_cfg.get("seed", None),
        tiled=infer_cfg.get("tiled", True),
        tile_size=tuple(infer_cfg.get("tile_size", [30, 52])),
        tile_stride=tuple(infer_cfg.get("tile_stride", [15, 26])),
    )

    print("num output keyframes =", len(frames))
    print("expected =", len(frame_prompts))

    if len(frames) != len(frame_prompts):
        raise ValueError(
            f"output frame number mismatch: got {len(frames)}, expected {len(frame_prompts)}"
        )

    for i, frame in enumerate(frames):
        frame.save(os.path.join(out_dir, f"{i:02d}.png"))

    # 顺便把配置存一份，便于追踪
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print("saved to", out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="path to one-sample json config")
    args = parser.parse_args()

    cfg = load_json(args.config)
    pipe = build_pipe(cfg)
    run_one(pipe, cfg)


if __name__ == "__main__":
    main()