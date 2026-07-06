#!/usr/bin/env python3
import argparse
import csv
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import Dinov2Model

from dino_adjacent_similarity import (
    DEFAULT_MODEL_DIR,
    DEFAULT_MODEL_ID,
    list_numeric_images,
    preprocess_image,
    write_adjacent_similarity_csv,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class FrameDataset(Dataset):
    def __init__(self, records: Sequence[Tuple[int, int, Path]]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        sample_idx, frame_idx, path = self.records[idx]
        return sample_idx, frame_idx, preprocess_image(path)


def collate_frames(batch):
    sample_indices, frame_indices, images = zip(*batch)
    return (
        torch.tensor(sample_indices, dtype=torch.long),
        torch.tensor(frame_indices, dtype=torch.long),
        torch.stack(images, dim=0),
    )


def find_sample_dirs(input_root: Path) -> List[Path]:
    return sorted([p for p in input_root.iterdir() if p.is_dir() and p.name.startswith("sample_")], key=lambda p: p.name)


def output_path_for_sample(output_root: Path, sample_dir: Path, suffix: str) -> Path:
    return output_root / f"{sample_dir.name}{suffix}"


def load_done_samples(done_path: Path) -> set[str]:
    if not done_path.exists():
        return set()
    done = set()
    with done_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("status") == "ok" and obj.get("sample"):
                done.add(str(obj["sample"]))
    return done


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_failure_csv(output_csv: Path, sample_dir: Path, message: str) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample", "error"])
        writer.writeheader()
        writer.writerow({"sample": sample_dir.name, "error": message})


def chunked(items: Sequence[Path], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def build_records(sample_dirs: Sequence[Path]) -> Tuple[List[Tuple[int, int, Path]], Dict[int, List[Path]], Dict[int, Path]]:
    records: List[Tuple[int, int, Path]] = []
    sample_images: Dict[int, List[Path]] = {}
    sample_dirs_by_idx: Dict[int, Path] = {}

    for sample_idx, sample_dir in enumerate(sample_dirs):
        image_dir = sample_dir / "images"
        if not image_dir.is_dir():
            sample_images[sample_idx] = []
            sample_dirs_by_idx[sample_idx] = sample_dir
            continue
        images = list_numeric_images(image_dir)
        sample_images[sample_idx] = images
        sample_dirs_by_idx[sample_idx] = sample_dir
        for frame_idx, path in enumerate(images):
            records.append((sample_idx, frame_idx, path))
    return records, sample_images, sample_dirs_by_idx


@torch.no_grad()
def encode_chunk(
    model: Dinov2Model,
    sample_dirs: Sequence[Path],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, List[Path]], Dict[int, Path]]:
    records, sample_images, sample_dirs_by_idx = build_records(sample_dirs)
    features_by_sample: Dict[int, List[Tuple[int, torch.Tensor]]] = {idx: [] for idx in sample_images}
    if not records:
        return {idx: torch.empty(0) for idx in sample_images}, sample_images, sample_dirs_by_idx

    loader = DataLoader(
        FrameDataset(records),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_frames,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    use_amp = device.type == "cuda"
    for sample_indices, frame_indices, pixel_values in loader:
        pixel_values = pixel_values.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            output = model(pixel_values=pixel_values)
            features = output.last_hidden_state[:, 0]
        features = F.normalize(features.float().cpu(), dim=-1)
        for sample_idx, frame_idx, feature in zip(sample_indices.tolist(), frame_indices.tolist(), features):
            features_by_sample[sample_idx].append((frame_idx, feature))

    stacked: Dict[int, torch.Tensor] = {}
    for sample_idx, pairs in features_by_sample.items():
        pairs.sort(key=lambda x: x[0])
        stacked[sample_idx] = torch.stack([feature for _, feature in pairs], dim=0) if pairs else torch.empty(0)
    return stacked, sample_images, sample_dirs_by_idx


def worker_main(
    rank: int,
    gpu_id: int,
    sample_dirs: Sequence[Path],
    output_root: Path,
    model_dir: Path,
    batch_size: int,
    samples_per_chunk: int,
    num_workers: int,
    suffix: str,
    skip_existing: bool,
) -> None:
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / f"worker_{rank}_gpu{gpu_id}.jsonl"
    done_samples = load_done_samples(log_path)

    model = Dinov2Model.from_pretrained(model_dir).to(device).eval()
    total = len(sample_dirs)
    processed = 0
    started = time.time()

    for chunk in chunked(sample_dirs, samples_per_chunk):
        todo = []
        for sample_dir in chunk:
            output_csv = output_path_for_sample(output_root, sample_dir, suffix)
            if sample_dir.name in done_samples:
                continue
            if skip_existing and output_csv.exists():
                append_jsonl(log_path, {"sample": sample_dir.name, "status": "ok", "output": str(output_csv), "skipped": True})
                continue
            todo.append(sample_dir)
        if not todo:
            continue

        try:
            features_by_sample, sample_images, sample_dirs_by_idx = encode_chunk(
                model=model,
                sample_dirs=todo,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            for sample_idx, sample_dir in sample_dirs_by_idx.items():
                output_csv = output_path_for_sample(output_root, sample_dir, suffix)
                images = sample_images[sample_idx]
                features = features_by_sample[sample_idx]
                if len(images) < 2 or len(features) != len(images):
                    message = f"need >=2 images and matching features, images={len(images)}, features={len(features)}"
                    write_failure_csv(output_csv, sample_dir, message)
                    append_jsonl(log_path, {"sample": sample_dir.name, "status": "failed", "error": message})
                    continue
                write_adjacent_similarity_csv(images, features, output_csv)
                append_jsonl(log_path, {"sample": sample_dir.name, "status": "ok", "output": str(output_csv), "rows": len(images) - 1})
                processed += 1
        except Exception as exc:
            for sample_dir in todo:
                output_csv = output_path_for_sample(output_root, sample_dir, suffix)
                write_failure_csv(output_csv, sample_dir, repr(exc))
                append_jsonl(log_path, {"sample": sample_dir.name, "status": "failed", "error": repr(exc)})

        elapsed = max(1e-6, time.time() - started)
        print(
            f"[worker {rank} gpu {gpu_id}] processed={processed}/{total} "
            f"rate={processed / elapsed:.2f} samples/s",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch DINOv2 adjacent-frame similarity over AgiBot sample directories.")
    parser.add_argument("--input-root", type=Path, required=True, help="Root containing sample_* directories.")
    parser.add_argument("--output-root", type=Path, required=True, help="Directory for one CSV per sample.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Kept for metadata; model is loaded from --model-dir.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Local DINOv2 model directory.")
    parser.add_argument("--gpus", default="0", help="Comma-separated visible GPU indices, e.g. 0,1 after CUDA_VISIBLE_DEVICES=1,2.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Per-GPU DINO inference batch size.")
    parser.add_argument("--samples-per-chunk", type=int, default=512, help="Samples flattened per GPU chunk.")
    parser.add_argument("--num-workers", type=int, default=12, help="DataLoader CPU workers per GPU process.")
    parser.add_argument("--suffix", default="_dinov2_base_adjacent_similarity.csv")
    parser.add_argument("--limit", type=int, default=0, help="Debug: only run first N samples after filtering.")
    parser.add_argument("--no-skip-existing", action="store_true", help="Recompute even if output CSV exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (args.model_dir / "config.json").exists() or not (args.model_dir / "model.safetensors").exists():
        raise FileNotFoundError(f"Missing local model files in {args.model_dir}. Download the model before running.")

    sample_dirs = find_sample_dirs(args.input_root)
    if args.limit > 0:
        sample_dirs = sample_dirs[: args.limit]
    if not sample_dirs:
        raise ValueError(f"No sample_* directories found under {args.input_root}")

    gpu_ids = [int(x) for x in args.gpus.split(",") if x.strip()]
    if not gpu_ids:
        raise ValueError("--gpus cannot be empty")

    shards = [sample_dirs[i:: len(gpu_ids)] for i in range(len(gpu_ids))]
    print(
        f"samples={len(sample_dirs)} gpus={gpu_ids} batch_size={args.batch_size} "
        f"samples_per_chunk={args.samples_per_chunk} num_workers={args.num_workers}",
        flush=True,
    )

    mp.set_start_method("spawn", force=True)
    processes = []
    for rank, (gpu_id, shard) in enumerate(zip(gpu_ids, shards)):
        proc = mp.Process(
            target=worker_main,
            args=(
                rank,
                gpu_id,
                shard,
                args.output_root,
                args.model_dir,
                args.batch_size,
                args.samples_per_chunk,
                args.num_workers,
                args.suffix,
                not args.no_skip_existing,
            ),
        )
        proc.start()
        processes.append(proc)

    failed = False
    for proc in processes:
        proc.join()
        if proc.exitcode != 0:
            failed = True
            print(f"worker pid={proc.pid} failed with exitcode={proc.exitcode}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
