#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--python", default="/media/datasets/yumi/hjh/conda_envs/keyframe/bin/python")
    parser.add_argument("--model-path", default="/media/datasets/yumi/hjh/models/Wan2.1-I2V-14B-480P")
    parser.add_argument("--gpu", default="7")
    parser.add_argument("--start-step", type=int, default=100)
    parser.add_argument("--end-step", type=int, default=1000)
    parser.add_argument("--interval", type=int, default=100)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--nfe", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--adapter-rank", type=int, default=128)
    parser.add_argument("--adapter-alpha", type=int, default=128)
    return parser.parse_args()


def write_infer_config(args, checkpoint: Path, step: int, config_path: Path) -> None:
    step_dir = Path(args.output_dir) / f"step_{step:06d}_cfg{args.cfg_scale:g}"
    cfg = {
        "experiment": {
            "name": "exp8_327_one_sample_overfit_eval",
            "dir": str(Path(args.run_dir)),
        },
        "model": {
            "model_id": "Wan-AI/Wan2.1-I2V-14B-480P",
            "model_path": args.model_path,
            "skip_download": True,
            "tokenizer_path": str(Path(args.model_path) / "google/umt5-xxl/"),
            "device": "cuda",
            "offload_device": "cpu",
            "onload_device": "cpu",
            "computation_device": "cuda",
            "vram_limit_gb": 76,
        },
        "checkpoint": {
            "conditioning_mode": "local_only",
            "selector": f"step_{step:06d}",
            "path": str(checkpoint),
        },
        "adapter": {
            "rank": args.adapter_rank,
            "alpha": args.adapter_alpha,
            "dropout": 0.05,
            "target_keywords": ["self_attn", "cross_attn"],
            "skip_keywords": [],
        },
        "data": {
            "val_manifest": args.manifest,
            "expected_num_slots": 16,
            "start_index": 0,
            "num_samples": 1,
        },
        "inference": {
            "output_dir": str(step_dir),
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.nfe,
            "cfg_scale": args.cfg_scale,
            "cfg_merge": False,
            "sigma_shift": 5.0,
            "seed": args.seed,
            "seed_per_sample": False,
            "rand_device": "cpu",
            "tiled": True,
            "tile_size": [30, 40],
            "tile_stride": [15, 20],
            "tea_cache_l1_thresh": None,
            "tea_cache_model_id": "Wan2.1-I2V-14B-480P",
            "framewise_decoding": False,
            "output_type": "quantized",
            "save_ground_truth": True,
            "contact_sheet_thumb_width": 192,
            "compare_square_size": 512,
            "compare_fps": 1,
            "overwrite": True,
            "fail_fast": True,
        },
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def run_inference(args, config_path: Path) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    cmd = [
        args.python,
        str(ROOT / "keyframegen/infer/infer_exp.py"),
        "--config",
        str(config_path),
    ]
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    steps = list(range(args.start_step, args.end_step + 1, args.interval))
    done = set()
    config_root = Path(args.output_dir) / "generated_infer_configs"
    print(f"[watch] run_dir={run_dir}", flush=True)
    print(f"[watch] steps={steps}", flush=True)
    while len(done) < len(steps):
        progressed = False
        for step in steps:
            if step in done:
                continue
            checkpoint = ckpt_dir / f"trainable_step_{step:06d}.pt"
            if not checkpoint.exists():
                continue
            config_path = config_root / f"infer_step_{step:06d}_cfg{args.cfg_scale:g}.json"
            print(f"[watch] evaluating {checkpoint}", flush=True)
            write_infer_config(args, checkpoint, step, config_path)
            run_inference(args, config_path)
            done.add(step)
            progressed = True
        if len(done) >= len(steps):
            break
        if not progressed:
            missing = [step for step in steps if step not in done]
            print(f"[watch] waiting for next checkpoint; remaining={missing}", flush=True)
        time.sleep(args.poll_seconds)
    print("[watch] complete", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[watch] interrupted", flush=True)
        sys.exit(130)
