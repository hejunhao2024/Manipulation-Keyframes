import os
import sys
import torch
from PIL import Image
from diffsynth.pipelines.dreamshot import DreamShotPipeline, ModelConfig
import transformers
import math
from PIL import Image
import json
import dataclasses
from accelerate import Accelerator
import random
from utils import safe_request_image, group_and_split, resize_and_pad, concat_images_grid
import glob
import time


@dataclasses.dataclass
class InferenceConfig:
    sub_idx: int = 0
    total_num: int = 1

    resolution: int = 720
    seed: int = 42

    img_cfg_scale: float = 1.0
    cfg_scale: float = 1.0
    sigma_shift: float = 1.0

    num_inference_steps: int = 4

    model_name: str = "Wan2.1-T2V-A14B" # "Wan2.2-T2V-A14B" 
    model_path: str = "./checkpoints/sft_lora_model.safetensors"
    model_path2: str = ""

    rl_model_path: str = "./checkpoints/rl_lora_model.safetensors"

    data_root: str = "./checkpoints/VistoryBench"
    output_dir: str = "./output/vistory_bench" 

    shot_mode: str = "ref2shot"

    vistory_json_path: str = "./checkpoints/vistory_dreamshot_ch_v1.json"

    lightx2v_model_path: str = "./checkpoints/wan2.1_t2v_14b_lora_rank64_lightx2v_4step.safetensors"

    sample_id: str = ""
    use_last_ref: bool = False
    use_phase_offset: str = "None"

    target_frame: int = 6
    context_frame: int = 0

    # negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，杂乱的背景，三条腿"
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"


def main():
    parser = transformers.HfArgumentParser(InferenceConfig)
    config: InferenceConfig = parser.parse_args_into_dataclasses()[0]

    if config.resolution == 720:
        height = 720
        width = 1280
    elif config.resolution == 480:
        height = 480
        width = 832
    elif config.resolution == 360:
        height = 352
        width = 640

    with open(config.vistory_json_path, "r") as f:
        vistory_json = json.load(f)

    shot_mode = config.shot_mode
    
    vis_dir = os.path.join(os.path.dirname(config.output_dir), "dreamshot_vis")
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    to_process = sorted(list(vistory_json.keys()))

    to_process = [process for process in to_process if not os.path.exists(os.path.join(vis_dir, process+".jpg"))]
    selected = group_and_split(to_process, total_num=config.total_num, sub_idx=config.sub_idx)
    vistory_json = {k: vistory_json[k] for k in selected}

    if config.model_name == "Wan2.1-T2V-A14B":
        pipe = DreamShotPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[
                ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
                ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
                ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="Wan2.1_VAE.pth"),
            ],
            tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
        )
        pipe.dit.require_vae_embedding = True
        pipe.load_lora(pipe.dit, config.model_path, alpha=1)
        if config.rl_model_path != "None":
            pipe.load_lora(pipe.dit, config.rl_model_path, alpha=1)
        if config.lightx2v_model_path != "None":
            config.cfg_scale = 1
            config.img_cfg_scale = 1.5
            config.num_inference_steps = 4
            pipe.load_lora(pipe.dit, config.lightx2v_model_path, alpha=1)
    elif config.model_name == "Wan2.2-TI2V-5B":
        pipe = DreamShotPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[
                ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
                ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
                ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth"),
            ],
        )
        pipe.dit.require_vae_embedding = True
        pipe.load_lora(pipe.dit, config.model_path, alpha=1)

    for key, value in vistory_json.items():
        shot_key, shot_id = key, 0
        shot_id = int(shot_id)
        tmp_valid_prompts = value["prompt_cn"]
        role_indices = value["role_indices"]
        ref_prompts = value["ref_prompt_cn_format"]
        ref_images_path = value['fg_url_list']
        ref_images = [Image.open(os.path.join(config.data_root, v)) for v in ref_images_path]

        save_path = os.path.join(config.output_dir, shot_key)
        os.makedirs(save_path, exist_ok=True)

        total_frames = len(tmp_valid_prompts)
        print(f"--- Starting inference: total frames {total_frames}, step size {config.target_frame} ---")

        generated_frames = []
        save_idx = 0
        existing_files = glob.glob(os.path.join(save_path, "*.png"))
        existing_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))


        if len(existing_files) > 0:
            print(f"[Resume] Detected {len(existing_files)} generated images, restoring context...")
            
            for file_path in existing_files:
                img = Image.open(file_path).convert("RGB")
                generated_frames.append(img)
            save_idx = len(generated_frames)

        resized_ref_images = [resize_and_pad(img, (width, height)) for img in ref_images]

        total_time = 0.0
        for start_idx in range(0, total_frames, config.target_frame):
            end_idx = min(start_idx + config.target_frame, total_frames)
            current_batch_indices = list(range(start_idx, end_idx))

            if end_idx <= len(generated_frames):
                print(f"[Skip] Skipping already generated batch: {current_batch_indices}")
                continue
            print(f"\n[Step] Generating shots: {current_batch_indices}")

            context_images = []
            context_prompts = []
            if start_idx == 0:
                print("  -> Context: none (first generation round)")
            else:
                avail_context_len = len(generated_frames)
                ctx_start = max(0, avail_context_len - config.context_frame)
                context_images = generated_frames[ctx_start:]
                context_prompts = tmp_valid_prompts[ctx_start:avail_context_len]
                print(f"  -> Context: using generated frames {list(range(ctx_start, avail_context_len))} as context")

            current_roles = set()
            no_role_indices = []
            for i, global_idx in enumerate(current_batch_indices):
                role_key = str(global_idx)
                roles = role_indices.get(role_key, [])
                if len(roles) == 0:
                    no_role_indices.append(i)
                for r in roles:
                    current_roles.add(r)
            sorted_roles = sorted(list(current_roles))

            print(f"  -> No-character frame indices: {no_role_indices}")
            
            batch_ref_images = [resized_ref_images[i] for i in sorted_roles]
            print(f"  -> Reference characters: {sorted_roles}")

            batch_prompts = tmp_valid_prompts[start_idx-len(context_images):end_idx]
            batch_ref_prompts = [ref_prompts[i] for i in sorted_roles] + context_prompts

            start_time = time.time()
            outputs = pipe(
                prompt=batch_prompts,
                negative_prompt=config.negative_prompt,
                seed=config.seed, tiled=True,
                height=height, width=width,
                num_frames=1+(len(batch_prompts) -1)*4,
                cfg_scale=config.cfg_scale,
                sigma_shift=config.sigma_shift,
                reference_images=batch_ref_images if shot_mode == "ref2shot" else None,
                context_images=context_images,
                ref_prompts=batch_ref_prompts if shot_mode == "ref2shot" else None,
                no_role_indices=no_role_indices,
                img_cfg_scale=config.img_cfg_scale if len(batch_ref_images) > 0 else 1.0,
                num_inference_steps=config.num_inference_steps,
            )
            total_time += time.time() - start_time


            grid_img = outputs[::4]

            grid_img_target = grid_img[len(context_images):]
            
            for img in grid_img_target:
                img.save(f"{save_path}/{save_idx+1}.png")
                generated_frames.append(img)
                save_idx += 1
        
        if shot_mode == "ref2shot":
            grid_img = resized_ref_images + generated_frames
        grid_img = concat_images_grid(grid_img)
        grid_img.save(os.path.join(vis_dir, f"{key}.jpg"))


if __name__ == "__main__":
    main()