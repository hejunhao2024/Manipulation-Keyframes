import os
import torch
from PIL import Image
import math
import re
import torch
import torch.nn as nn
import pathlib
import requests
import time
from io import BytesIO
from collections import defaultdict


RETRY = 3
TIMEOUT = 10
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


def group_and_split(to_process, total_num, sub_idx):
    grouped_list = list(to_process)

    n = len(grouped_list)
    base = n // total_num
    extra = n % total_num

    start = sub_idx * base + min(sub_idx, extra)
    end = start + base + (1 if sub_idx < extra else 0)

    selected = grouped_list[start:end]
    return selected


def resize_and_pad(img: Image.Image, target_size, pad_color=(255, 255, 255)):
    target_w, target_h = target_size
    w, h = img.size

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = img.resize((new_w, new_h), Image.LANCZOS)

    new_img = Image.new("RGB", (target_w, target_h), pad_color)

    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    new_img.paste(resized, (paste_x, paste_y))

    return new_img


def concat_images_grid(image_list, images_per_row=4):
    images = [Image.open(img) if isinstance(img, str) else img for img in image_list]
    
    w, h = images[0].size
    rows = math.ceil(len(images) / images_per_row)
    
    grid_img = Image.new('RGB', (w * images_per_row, h * rows), color=(255, 255, 255))

    for idx, img in enumerate(images):
        x = (idx % images_per_row) * w
        y = (idx // images_per_row) * h
        grid_img.paste(img, (x, y))

    return grid_img

def safe_request_image(url: str | None,
                       dst: pathlib.Path | None = None,
                       retries: int = RETRY,
                       timeout: int = TIMEOUT) -> Image.Image | None:

    url = str(url).strip(' "\n') if url else ""
    if (not url) or url.lower() == "nan" or url.startswith(("{", "[")):
        return None

    if dst is not None and dst.exists():
        try:
            return Image.open(dst).convert("RGB")
        except Exception:
            pass

    for i in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()

            img = Image.open(BytesIO(r.content)).convert("RGB")

            if dst is not None:
                dst.parent.mkdir(parents=True, exist_ok=True)
                with dst.open("wb") as f:
                    f.write(r.content)

            return img

        except Exception as e:
            if i < retries - 1:
                time.sleep(1)
            else:
                print(f"[FAIL] {url} -> {e}")

    return None



def shots_roles_indices(shots, role_dict):
    allowed = set()
    for k in role_dict.keys():
        m = re.search(r'角色(\d+)', k)
        if m:
            allowed.add(int(m.group(1)))

    pat = re.compile(r'\<角色(\d+)\>')  
    result = {}

    for i, sent in enumerate(shots):
        ids = {int(x) for x in pat.findall(sent)}    
        ids = sorted((n - 1) for n in ids if n in allowed)
        result[str(i)] = ids

    return result


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0


    def on_step_end(self, accelerator, model,save_steps=None, optimizer=None, scheduler=None):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}", optimizer, scheduler)


    def on_training_end(self, accelerator, model, save_steps=None, optimizer=None, scheduler=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}", optimizer, scheduler)


    def save_model(self, accelerator, model, file_name, optimizer, scheduler):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            # save optimizer and scheduler
            output_dir = os.path.join(self.output_path, file_name)
            os.makedirs(output_dir, exist_ok=True)
            accelerator.save(state_dict, os.path.join(output_dir, "lora_model.safetensors"), safe_serialization=True)
            train_state = {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": self.num_steps
            }
            accelerator.save(train_state, os.path.join(output_dir, "train_state.pth"))

    
    def log_validation(self, accelerator, model, data, vis_step=None, shot_mode=None):
        if vis_step is None or self.num_steps % vis_step != 0:
            return
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            model.eval()
            
            with torch.no_grad():
                pipeline = model.module.pipe if accelerator.num_processes > 1 else model.pipe
                outputs = pipeline(
                    prompt=tmp_valid_prompts,
                    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                    seed=12345, tiled=True,
                    height=352, width=640,
                    num_frames=1+(len(tmp_valid_prompts)-1)*4,
                    reference_image=ref_images if shot_mode == "ref2shot" else None,
                    ref_prompts=ref_prompts if shot_mode == "ref2shot" else None,
                    role_indices=role_indices if shot_mode == "ref2shot" else None,
                    shot_mode=shot_mode,
                )
                valid_save_path = os.path.join(self.output_path, "validation_vis", f"step-{self.num_steps}")
                os.makedirs(valid_save_path, exist_ok=True) 
                grid_img = [outputs[0]]
                if shot_mode == "ref2shot":
                    tmp_ref_images = ref_images.copy()
                    tmp_ref_images = [resize_and_pad(ref_img, grid_img[0].size) for ref_img in tmp_ref_images]
                    grid_img = tmp_ref_images + grid_img
                grid_img.extend(outputs[4::4])
                grid_img = concat_images_grid(grid_img)
                grid_img.save(os.path.join(valid_save_path, f"validation.png"))

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            model.train()
            pipeline.scheduler.set_timesteps(1000, training=True)
        accelerator.wait_for_everyone()