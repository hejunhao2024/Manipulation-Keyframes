#!/usr/bin/env python3
import argparse
import csv
import os
import time
from pathlib import Path
from typing import List, Sequence

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import Dinov2Model


DEFAULT_MODEL_ID = "facebook/dinov2-base"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "facebook" / "dinov2-base"


def list_numeric_images(image_dir: Path) -> List[Path]:
    images = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    numeric = [p for p in images if p.stem.isdigit()]
    if numeric:
        return sorted(numeric, key=lambda p: int(p.stem))
    return sorted(images, key=lambda p: p.name)


def resize_shortest_edge(image: Image.Image, shortest_edge: int) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("image has invalid size")
    if width < height:
        new_width = shortest_edge
        new_height = round(height * shortest_edge / width)
    else:
        new_height = shortest_edge
        new_width = round(width * shortest_edge / height)
    return image.resize((new_width, new_height), Image.Resampling.BICUBIC)


def center_crop(image: Image.Image, crop_size: int) -> Image.Image:
    width, height = image.size
    left = max(0, (width - crop_size) // 2)
    top = max(0, (height - crop_size) // 2)
    return image.crop((left, top, left + crop_size, top + crop_size))


def preprocess_image(path: Path, image_size: int = 224, resize_size: int = 256) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = center_crop(resize_shortest_edge(image, resize_size), image_size)
    data = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (data - mean) / std


def ensure_model(model_id: str, model_dir: Path) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(1, 4):
        try:
            snapshot_download(
                repo_id=model_id,
                local_dir=str(model_dir),
                max_workers=1,
                allow_patterns=[
                    "config.json",
                    "model.safetensors",
                    "preprocessor_config.json",
                ],
            )
            break
        except Exception as exc:
            last_error = exc
            if attempt == 3:
                raise
            print(f"Download attempt {attempt} failed: {exc}. Retrying...")
            time.sleep(5 * attempt)
    return model_dir


@torch.no_grad()
def encode_images(
    model: Dinov2Model,
    image_paths: Sequence[Path],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    features = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        pixel_values = torch.stack([preprocess_image(path) for path in batch_paths]).to(device)
        output = model(pixel_values=pixel_values)
        cls_features = output.last_hidden_state[:, 0]
        features.append(F.normalize(cls_features.float().cpu(), dim=-1))
    return torch.cat(features, dim=0)


def write_adjacent_similarity_csv(image_paths: Sequence[Path], features: torch.Tensor, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index_a", "image_a", "index_b", "image_b", "dino_cosine_similarity"],
        )
        writer.writeheader()
        for idx in range(len(image_paths) - 1):
            score = torch.dot(features[idx], features[idx + 1]).item()
            writer.writerow(
                {
                    "index_a": idx,
                    "image_a": image_paths[idx].name,
                    "index_b": idx + 1,
                    "image_b": image_paths[idx + 1].name,
                    "dino_cosine_similarity": f"{score:.8f}",
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute adjacent-frame DINOv2 cosine similarity for keyframe deduplication.")
    parser.add_argument("--image-dir", type=Path, required=True, help="Directory containing ordered frame images.")
    parser.add_argument("--output-csv", type=Path, required=True, help="CSV path for adjacent-frame similarity rows.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face DINO/DINOv2 model id.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Local model directory under project models.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda", help="Use cuda or cpu. With CUDA_VISIBLE_DEVICES=1, cuda maps to physical GPU 1.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = list_numeric_images(args.image_dir)
    if len(image_paths) < 2:
        raise ValueError(f"Need at least 2 images in {args.image_dir}, found {len(image_paths)}")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model_dir = ensure_model(args.model_id, args.model_dir)
    model = Dinov2Model.from_pretrained(model_dir).to(device).eval()

    features = encode_images(model, image_paths, device=device, batch_size=args.batch_size)
    write_adjacent_similarity_csv(image_paths, features, args.output_csv)

    print(f"model_dir={model_dir}")
    print(f"device={device}")
    print(f"images={len(image_paths)}")
    print(f"rows={len(image_paths) - 1}")
    print(f"output_csv={args.output_csv}")


if __name__ == "__main__":
    main()
