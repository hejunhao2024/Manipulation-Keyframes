#!/usr/bin/env python3
import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional


DEFAULT_INPUT_ROOT = Path("/media/datasets/yumi/hjh/datasets/agibot_v2_uncleaned")
DEFAULT_OUTPUT_ROOT = Path("/media/datasets/yumi/hjh/datasets/agibot_v2_task")


def extract_task_id(source_dir: str) -> Optional[str]:
    match = re.search(r"/agibot_v2/([^/]+)/", source_dir)
    if not match:
        return None
    return match.group(1)


def read_task_id(sample_dir: Path) -> Optional[str]:
    debug_path = sample_dir / "debug.json"
    if not debug_path.exists():
        return None
    try:
        obj = json.loads(debug_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    source_dir = obj.get("source_dir")
    if not isinstance(source_dir, str):
        return None
    return extract_task_id(source_dir)


def copy_sample(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            return
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", "-a", "--reflink=auto", str(src), str(dst)], check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify AgiBot sample_* directories by task id parsed from debug.json source_dir.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_dirs = sorted([p for p in args.input_root.iterdir() if p.is_dir() and p.name.startswith("sample_")], key=lambda p: p.name)
    if args.limit > 0:
        sample_dirs = sample_dirs[: args.limit]

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.csv"
    failures_path = args.output_root / "failures.csv"

    ok = skipped = failed = 0
    with manifest_path.open("w", encoding="utf-8") as mf, failures_path.open("w", encoding="utf-8") as ff:
        mf.write("sample,task_id,source_path,target_path\n")
        ff.write("sample,reason,source_path\n")

        for idx, sample_dir in enumerate(sample_dirs, start=1):
            task_id = read_task_id(sample_dir)
            if task_id is None:
                failed += 1
                ff.write(f"{sample_dir.name},missing_or_invalid_debug,{sample_dir}\n")
                continue

            target_dir = args.output_root / task_id / sample_dir.name
            existed = target_dir.exists()
            try:
                copy_sample(sample_dir, target_dir, args.overwrite)
            except Exception as exc:
                failed += 1
                ff.write(f"{sample_dir.name},{type(exc).__name__}:{exc},{sample_dir}\n")
                continue

            if existed and not args.overwrite:
                skipped += 1
            else:
                ok += 1
            mf.write(f"{sample_dir.name},{task_id},{sample_dir},{target_dir}\n")

            if idx % 500 == 0 or idx == len(sample_dirs):
                print(f"processed={idx}/{len(sample_dirs)} copied={ok} skipped={skipped} failed={failed}", flush=True)

    print(f"done copied={ok} skipped={skipped} failed={failed}")
    print(f"output_root={args.output_root}")
    print(f"manifest={manifest_path}")
    print(f"failures={failures_path}")


if __name__ == "__main__":
    main()
