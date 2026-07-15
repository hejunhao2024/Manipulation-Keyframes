#!/usr/bin/env python3
"""Flatten aigbot_final samples and clean annotation metadata."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("/media/datasets/yumi/hjh/datasets/aigbot_final")


def rename_key_recursive(value: Any, old_key: str, new_key: str) -> Any:
    if isinstance(value, dict):
        updated = {}
        for key, item in value.items():
            key = new_key if key == old_key else key
            updated[key] = rename_key_recursive(item, old_key, new_key)
        return updated
    if isinstance(value, list):
        return [rename_key_recursive(item, old_key, new_key) for item in value]
    return value


def clean_annotation(annotation_path: Path, sample_dir: Path, root: Path, dry_run: bool) -> None:
    with annotation_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    data = rename_key_recursive(data, "global_prompt", "system_prompt")

    relative_sample = sample_dir.relative_to(root).as_posix()
    if isinstance(data, dict):
        data["sample_id"] = relative_sample
        data["sample_short"] = relative_sample
        data["image_dir"] = str(sample_dir)
        frames = data.get("frames")
        if isinstance(frames, list):
            for frame in frames:
                if isinstance(frame, dict):
                    frame.pop("prompt_file", None)

    if not dry_run:
        with annotation_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")


def move_file(src: Path, dst: Path, dry_run: bool) -> bool:
    if src == dst:
        return False
    if dst.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {dst}")
    if not dry_run:
        shutil.move(str(src), str(dst))
    return True


def remove_empty_parents(path: Path, stop_at: Path, dry_run: bool) -> int:
    removed = 0
    current = path
    while current != stop_at and stop_at in current.parents:
        try:
            if any(current.iterdir()):
                break
        except FileNotFoundError:
            current = current.parent
            continue
        if not dry_run:
            current.rmdir()
        removed += 1
        current = current.parent
    return removed


def clean_root(root: Path, dry_run: bool) -> dict[str, int]:
    stats = {
        "samples": 0,
        "images_moved": 0,
        "annotations_moved": 0,
        "annotations_cleaned": 0,
        "txt_deleted": 0,
        "dirs_deleted": 0,
    }

    head_dirs = sorted(root.glob("*/*/*/videos/head_color"))
    for head_dir in head_dirs:
        sample_dir = head_dir.parent.parent
        stats["samples"] += 1

        for image_path in sorted(head_dir.glob("*.jpg")):
            if move_file(image_path, sample_dir / image_path.name, dry_run):
                stats["images_moved"] += 1

        annotation_path = head_dir / "annotation.json"
        if annotation_path.exists():
            clean_annotation(annotation_path, sample_dir, root, dry_run)
            stats["annotations_cleaned"] += 1
            if move_file(annotation_path, sample_dir / "annotation.json", dry_run):
                stats["annotations_moved"] += 1

        for txt_path in sorted(head_dir.glob("*.txt")):
            if not dry_run:
                txt_path.unlink()
            stats["txt_deleted"] += 1

        stats["dirs_deleted"] += remove_empty_parents(head_dir, sample_dir, dry_run)

    for txt_path in sorted(root.rglob("*.txt")):
        if not dry_run:
            txt_path.unlink()
        stats["txt_deleted"] += 1

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move images/annotation.json out of videos/head_color, delete txt files, and clean JSON keys."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    stats = clean_root(root, args.dry_run)
    prefix = "DRY RUN " if args.dry_run else ""
    for key, value in stats.items():
        print(f"{prefix}{key}: {value}")


if __name__ == "__main__":
    main()
