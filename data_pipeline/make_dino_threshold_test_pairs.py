#!/usr/bin/env python3
import argparse
import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


DEFAULT_CSV_ROOT = Path("/media/datasets/yumi/hjh/datasets/agibot_v2_uncleaned_dino")
DEFAULT_IMAGE_ROOT = Path("/media/datasets/yumi/hjh/datasets/agibot_v2_uncleaned")
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "dino阈值测试"
DEFAULT_SUFFIX = "_dinov2_base_adjacent_similarity.csv"
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
]


@dataclass(frozen=True)
class PairRecord:
    sample: str
    image_a: str
    image_b: str
    similarity: float


def make_bins(start: float, stop: float, step: float) -> List[Tuple[float, float]]:
    bins = []
    value = start
    while value < stop - 1e-12:
        upper = min(stop, value + step)
        bins.append((round(value, 3), round(upper, 3)))
        value = upper
    return bins


def bin_name(lower: float, upper: float) -> str:
    return f"{lower:.3f}-{upper:.3f}"


def load_font(size: int):
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def sample_name_from_csv(path: Path, suffix: str) -> str:
    name = path.name
    if not name.endswith(suffix):
        raise ValueError(f"Unexpected CSV suffix: {path}")
    return name[: -len(suffix)]


def collect_pairs(csv_root: Path, suffix: str, bins: List[Tuple[float, float]]) -> Dict[str, List[PairRecord]]:
    pairs_by_bin = {bin_name(lower, upper): [] for lower, upper in bins}
    csv_paths = sorted(csv_root.glob(f"*{suffix}"))
    for csv_path in csv_paths:
        sample = sample_name_from_csv(csv_path, suffix)
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    sim = float(row["dino_cosine_similarity"])
                except Exception:
                    continue
                for lower, upper in bins:
                    is_last = upper >= bins[-1][1]
                    if lower <= sim < upper or (is_last and lower <= sim <= upper):
                        pairs_by_bin[bin_name(lower, upper)].append(
                            PairRecord(
                                sample=sample,
                                image_a=row["image_a"],
                                image_b=row["image_b"],
                                similarity=sim,
                            )
                        )
                        break
    return pairs_by_bin


def resize_to_height(image: Image.Image, target_height: int) -> Image.Image:
    width, height = image.size
    scale = target_height / height
    return image.resize((max(1, round(width * scale)), target_height), Image.Resampling.LANCZOS)


def make_pair_image(record: PairRecord, image_root: Path, title_label: str, image_height: int) -> Image.Image:
    path_a = image_root / record.sample / "images" / record.image_a
    path_b = image_root / record.sample / "images" / record.image_b
    img_a = resize_to_height(Image.open(path_a).convert("RGB"), image_height)
    img_b = resize_to_height(Image.open(path_b).convert("RGB"), image_height)

    gap = 8
    title_height = 46
    width = img_a.width + gap + img_b.width
    height = title_height + image_height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    title = f"{title_label}={record.similarity:.6f}"
    font = load_font(24)
    bbox = draw.textbbox((0, 0), title, font=font)
    text_x = max(0, (width - (bbox[2] - bbox[0])) // 2)
    text_y = max(0, (title_height - (bbox[3] - bbox[1])) // 2 - 2)
    draw.text((text_x, text_y), title, fill="black", font=font)

    canvas.paste(img_a, (0, title_height))
    canvas.paste(img_b, (img_a.width + gap, title_height))
    draw.rectangle((img_a.width, title_height, img_a.width + gap - 1, height - 1), fill=(245, 245, 245))
    return canvas


def write_bin(output_root: Path, bin_label: str, records: List[PairRecord], image_root: Path, title_label: str, image_height: int) -> None:
    out_dir = output_root / bin_label
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "sample", "image_a", "image_b", "similarity"])
        writer.writeheader()
        for idx, record in enumerate(records, start=1):
            out_name = f"{idx:03d}_{record.sample}_{Path(record.image_a).stem}_{Path(record.image_b).stem}_sim_{record.similarity:.6f}.jpg"
            out_path = out_dir / out_name
            pair_image = make_pair_image(record, image_root, title_label, image_height)
            pair_image.save(out_path, quality=95)
            writer.writerow(
                {
                    "file": out_name,
                    "sample": record.sample,
                    "image_a": record.image_a,
                    "image_b": record.image_b,
                    "similarity": f"{record.similarity:.8f}",
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create side-by-side DINO threshold test image pairs.")
    parser.add_argument("--csv-root", type=Path, default=DEFAULT_CSV_ROOT)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--suffix", default=DEFAULT_SUFFIX)
    parser.add_argument("--start", type=float, default=0.950)
    parser.add_argument("--stop", type=float, default=1.000)
    parser.add_argument("--step", type=float, default=0.005)
    parser.add_argument("--samples-per-bin", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-height", type=int, default=360)
    parser.add_argument("--title-label", default="clip sim")
    parser.add_argument("--clear-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bins = make_bins(args.start, args.stop, args.step)
    random.seed(args.seed)

    if args.clear_output and args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    pairs_by_bin = collect_pairs(args.csv_root, args.suffix, bins)
    summary_path = args.output_root / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["bin", "available", "sampled"])
        writer.writeheader()
        for label, pairs in pairs_by_bin.items():
            sampled = random.sample(pairs, min(args.samples_per_bin, len(pairs)))
            write_bin(args.output_root, label, sampled, args.image_root, args.title_label, args.image_height)
            writer.writerow({"bin": label, "available": len(pairs), "sampled": len(sampled)})
            print(f"{label}: available={len(pairs)} sampled={len(sampled)}")
    print(f"output_root={args.output_root}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
