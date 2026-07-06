#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUT_ROOT = Path("/media/datasets/yumi/hjh/datasets/agibot_v2_uncleaned_dino")
DEFAULT_OUTPUT_CSV = Path(__file__).resolve().parent / "dino_top5_similarity_scores.csv"
DEFAULT_SORTED_BAR = Path(__file__).resolve().parent / "dino_top5_similarity_scores_sorted_bar.png"
DEFAULT_HISTOGRAM = Path(__file__).resolve().parent / "dino_top5_similarity_scores_histogram.png"
DEFAULT_SUFFIX = "_dinov2_base_adjacent_similarity.csv"


def sample_name_from_csv(path: Path, suffix: str) -> str:
    if not path.name.endswith(suffix):
        raise ValueError(f"Unexpected CSV suffix: {path}")
    return path.name[: -len(suffix)]


def read_scores(path: Path) -> list[float]:
    values = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                values.append(float(row["dino_cosine_similarity"]))
            except Exception:
                continue
    return values


def compute_rows(input_root: Path, suffix: str, top_k: int) -> list[dict]:
    rows = []
    for csv_path in sorted(input_root.glob(f"*{suffix}")):
        sample = sample_name_from_csv(csv_path, suffix)
        values = read_scores(csv_path)
        top_values = sorted(values, reverse=True)[:top_k]
        if not top_values:
            score = ""
        else:
            score = mean(top_values)
        rows.append(
            {
                "sample": sample,
                "score_top5_mean": score,
                "num_pairs": len(values),
                "top_values": ";".join(f"{v:.8f}" for v in top_values),
                "csv_path": str(csv_path),
            }
        )
    rows.sort(key=lambda row: (row["score_top5_mean"] == "", row["score_top5_mean"]), reverse=True)
    return rows


def write_csv(rows: list[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample", "score_top5_mean", "num_pairs", "top_values", "csv_path"])
        writer.writeheader()
        for row in rows:
            out = dict(row)
            if out["score_top5_mean"] != "":
                out["score_top5_mean"] = f"{out['score_top5_mean']:.8f}"
            writer.writerow(out)


def plot_sorted_bar(rows: list[dict], output_png: Path) -> None:
    valid = [row for row in rows if row["score_top5_mean"] != ""]
    scores = [row["score_top5_mean"] for row in valid]
    fig, ax = plt.subplots(figsize=(18, 6), dpi=180)
    ax.bar(range(len(scores)), scores, width=1.0, color="#4C78A8")
    ax.set_title("DINOv2-base Top-5 Adjacent Similarity Mean per Sample")
    ax.set_xlabel("Samples sorted by score, high means lower quality")
    ax.set_ylabel("Top-5 mean similarity")
    ax.set_ylim(max(0.0, min(scores) - 0.02), min(1.0, max(scores) + 0.005))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)


def plot_histogram(rows: list[dict], output_png: Path) -> None:
    scores = [row["score_top5_mean"] for row in rows if row["score_top5_mean"] != ""]
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    ax.hist(scores, bins=60, color="#59A14F", edgecolor="white")
    ax.set_title("Distribution of DINOv2-base Top-5 Similarity Scores")
    ax.set_xlabel("Top-5 mean similarity score")
    ax.set_ylabel("Sample count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score each sample by mean of its top-k adjacent DINO similarities.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--suffix", default=DEFAULT_SUFFIX)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--sorted-bar", type=Path, default=DEFAULT_SORTED_BAR)
    parser.add_argument("--histogram", type=Path, default=DEFAULT_HISTOGRAM)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = compute_rows(args.input_root, args.suffix, args.top_k)
    if not rows:
        raise ValueError(f"No CSV files found in {args.input_root} with suffix {args.suffix}")
    write_csv(rows, args.output_csv)
    plot_sorted_bar(rows, args.sorted_bar)
    plot_histogram(rows, args.histogram)

    valid_scores = [row["score_top5_mean"] for row in rows if row["score_top5_mean"] != ""]
    print(f"samples={len(rows)}")
    print(f"valid_scores={len(valid_scores)}")
    print(f"min={min(valid_scores):.8f}")
    print(f"max={max(valid_scores):.8f}")
    print(f"mean={mean(valid_scores):.8f}")
    print(f"output_csv={args.output_csv}")
    print(f"sorted_bar={args.sorted_bar}")
    print(f"histogram={args.histogram}")


if __name__ == "__main__":
    main()
