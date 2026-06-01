import os
import sys
import json
from PIL import Image
import torch

ROOT = "/home/hejunhao-20251119/mnt/work/Manipulation-Keyframes"
sys.path.insert(0, ROOT)

from diffsynth.pipelines.keyframe import WanVideoPipeline
from diffsynth.core import ModelConfig


JSON_PATH = "/home/hejunhao-20251119/mnt/work/Manipulation-Keyframes/test/key_frame.json"
OUT_DIR = "/home/hejunhao-20251119/mnt/work/Manipulation-Keyframes/debug_outputs/test_one_keyframe"

os.makedirs(OUT_DIR, exist_ok=True)

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

input_image = Image.open(data["image"]).convert("RGB")

# 14B 建议先开低显存管理
vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(
            model_id="Wan-AI/Wan2.1-I2V-14B-480P",
            origin_file_pattern="diffusion_pytorch_model*.safetensors",
            **vram_config,
        ),
        ModelConfig(
            model_id="Wan-AI/Wan2.1-I2V-14B-480P",
            origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
            **vram_config,
        ),
        ModelConfig(
            model_id="Wan-AI/Wan2.1-I2V-14B-480P",
            origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            **vram_config,
        ),
        ModelConfig(
            model_id="Wan-AI/Wan2.1-I2V-14B-480P",
            origin_file_pattern="Wan2.1_VAE.pth",
            **vram_config,
        ),
    ],
    tokenizer_config=ModelConfig(
        model_id="Wan-AI/Wan2.1-T2V-1.3B",
        origin_file_pattern="google/umt5-xxl/",
    ),
    redirect_common_files=False,
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
)

frames = pipe(
    prompt=data["prompt"],
    negative_prompt="",
    input_image=input_image,
    frame_prompts=data["frame_prompts"],

    height=480,
    width=832,

    num_inference_steps=10,
    cfg_scale=5.0,
    cfg_merge=False,
    sigma_shift=5.0,

    tiled=True,
    tile_size=(30, 52),
    tile_stride=(15, 26),
)

print("num output keyframes =", len(frames))
print("expected =", len(data["frame_prompts"]))

assert len(frames) == len(data["frame_prompts"]), (
    len(frames),
    len(data["frame_prompts"]),
)

for i, frame in enumerate(frames):
    frame.save(os.path.join(OUT_DIR, f"{i:02d}.png"))

print("saved to", OUT_DIR)