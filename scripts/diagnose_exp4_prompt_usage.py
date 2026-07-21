#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffsynth.core import ModelConfig
from diffsynth.pipelines.keyframe_local_context import (
    WanVideoPipeline,
    WanVideoUnit_PromptEmbedder,
)
from keyframegen.train import agibot_non_ar_train as trainer


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_manifest(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.lstrip().startswith("#")
        ]


def resolve_manifest_item(item: str, data_root: str) -> str:
    path = Path(item)
    if path.is_absolute():
        return str(path)
    return str(Path(data_root) / path)


def sample_id(sample_dir: str) -> str:
    ann = load_json(Path(sample_dir) / "annotation.json")
    return str(
        ann.get("id")
        or ann.get("sample_id")
        or ann.get("sample_short")
        or Path(sample_dir).name
    ).strip("/")


def task_name(sample_id_value: str) -> str:
    return sample_id_value.split("/", 1)[0]


def select_samples(cfg: Dict[str, Any], train_count: int, val_count: int) -> List[Tuple[str, str]]:
    data_cfg = cfg["data"]
    train_items = read_manifest(data_cfg["train_manifest"])[:train_count]
    val_manifest = data_cfg.get("val_manifest")
    val_items = read_manifest(val_manifest)[:val_count] if val_manifest else []
    samples = []
    for item in train_items:
        samples.append(("train", resolve_manifest_item(item, data_cfg["data_root"])))
    for item in val_items:
        samples.append(("val", resolve_manifest_item(item, data_cfg["data_root"])))
    return samples


def build_text_pipe(cfg: Dict[str, Any], device: str):
    model_cfg = cfg["model"]
    model_path = Path(model_cfg["model_path"])
    vram_config = {
        "offload_dtype": torch.bfloat16,
        "offload_device": model_cfg.get("offload_device", "cpu"),
        "onload_dtype": torch.bfloat16,
        "onload_device": model_cfg.get("onload_device", "cpu"),
        "computation_dtype": torch.bfloat16,
        "computation_device": device,
    }
    return WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(
                path=str(model_path / "models_t5_umt5-xxl-enc-bf16.pth"),
                **vram_config,
            )
        ],
        tokenizer_config=ModelConfig(
            path=str(model_path / "google" / "umt5-xxl")
        ),
        redirect_common_files=False,
        vram_limit=model_cfg.get("vram_limit_gb"),
    )


def encode_empty_context(cfg: Dict[str, Any], device: str, num_slots: int) -> torch.Tensor:
    pipe = build_text_pipe(cfg, device)
    unit = WanVideoUnit_PromptEmbedder()
    with torch.no_grad():
        out = unit.process(
            pipe=pipe,
            prompt="",
            frame_prompts=[""] * num_slots,
            num_slots=num_slots,
            positive=True,
        )
    context = out["context"].detach().cpu()
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return context


def strict_load_trainable(model: torch.nn.Module, checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["trainable_state_dict"]
    expected = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    state_keys = set(state.keys())
    expected_keys = set(expected.keys())
    missing = sorted(expected_keys - state_keys)
    extra = sorted(state_keys - expected_keys)
    shape_mismatch = sorted(
        key
        for key in expected_keys & state_keys
        if tuple(expected[key].shape) != tuple(state[key].shape)
    )
    if missing or extra or shape_mismatch:
        raise RuntimeError(
            "Checkpoint trainable keys do not exactly match current model. "
            f"missing={len(missing)} extra={len(extra)} shape_mismatch={len(shape_mismatch)} "
            f"missing_first={missing[:5]} extra_first={extra[:5]} shape_first={shape_mismatch[:5]}"
        )
    model.load_state_dict(
        {key: value.to(dtype=expected[key].dtype) for key, value in state.items()},
        strict=False,
    )
    return checkpoint


def lora_delta_stats(model: torch.nn.Module, out_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for name, module in model.named_modules():
        if not isinstance(module, trainer.LoRALinear):
            continue
        base_weight = module.base.weight.detach().float()
        a = module.lora_A.detach().float()
        b = module.lora_B.detach().float()
        gram_a = a @ a.t()
        gram_b = b.t() @ b
        delta_sq = torch.sum(gram_a * gram_b).clamp_min(0.0) * (module.scale ** 2)
        delta_norm = torch.sqrt(delta_sq).item()
        base_norm = base_weight.norm().item()
        eigvals = torch.linalg.eigvals(gram_a @ gram_b).real.clamp_min(0.0)
        singular = torch.sqrt(eigvals)
        rank_threshold = singular.max().item() * 1e-3 if singular.numel() else 0.0
        effective_rank = int((singular > rank_threshold).sum().item())
        parts = name.split(".")
        layer = int(parts[1]) if len(parts) > 2 and parts[0] == "blocks" else -1
        attn_type = "self_attn" if ".self_attn." in name else "cross_attn" if ".cross_attn." in name else "other"
        module_name = parts[-1]
        rows.append(
            {
                "layer": layer,
                "module": name,
                "attn_type": attn_type,
                "proj": module_name,
                "a_norm": a.norm().item(),
                "b_norm": b.norm().item(),
                "delta_norm": delta_norm,
                "base_norm": base_norm,
                "delta_over_base": delta_norm / (base_norm + 1e-12),
                "effective_rank": effective_rank,
            }
        )
    with open(out_dir / "lora_delta_stats.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def heat_color(value: float, vmax: float) -> Tuple[int, int, int]:
    if vmax <= 0:
        t = 0.0
    else:
        t = max(0.0, min(1.0, value / vmax))
    return (int(255 * t), int(255 * (1.0 - abs(t - 0.5) * 2.0)), int(255 * (1.0 - t)))


def draw_lora_heatmap(rows: List[Dict[str, Any]], out_path: Path, attn_type: str, projs: List[str]) -> None:
    layers = sorted({row["layer"] for row in rows if row["attn_type"] == attn_type})
    values = {
        (row["layer"], row["proj"]): row["delta_over_base"]
        for row in rows
        if row["attn_type"] == attn_type
    }
    cell_w, cell_h = 86, 24
    left, top = 90, 36
    vmax = max([values.get((layer, proj), 0.0) for layer in layers for proj in projs] + [1e-12])
    img = Image.new("RGB", (left + cell_w * len(projs) + 20, top + cell_h * len(layers) + 40), "white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 8), f"{attn_type} delta/base heatmap vmax={vmax:.4g}", fill="black")
    for j, proj in enumerate(projs):
        draw.text((left + j * cell_w + 4, top - 22), proj, fill="black")
    for i, layer in enumerate(layers):
        draw.text((10, top + i * cell_h + 4), f"layer {layer:02d}", fill="black")
        for j, proj in enumerate(projs):
            value = values.get((layer, proj), 0.0)
            x0, y0 = left + j * cell_w, top + i * cell_h
            draw.rectangle([x0, y0, x0 + cell_w - 2, y0 + cell_h - 2], fill=heat_color(value, vmax))
            draw.text((x0 + 3, y0 + 4), f"{value:.2e}", fill="black")
    img.save(out_path)


def draw_matrix_heatmap(matrix: torch.Tensor, out_path: Path, title: str) -> None:
    matrix = matrix.float().cpu()
    n, m = matrix.shape
    cell = 34
    left, top = 70, 42
    vmax = float(matrix.max().item()) if matrix.numel() else 1.0
    img = Image.new("RGB", (left + m * cell + 20, top + n * cell + 50), "white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 8), f"{title} vmax={vmax:.4g}", fill="black")
    for j in range(m):
        draw.text((left + j * cell + 8, top - 22), str(j), fill="black")
    for i in range(n):
        draw.text((12, top + i * cell + 9), f"out {i:02d}", fill="black")
        for j in range(m):
            value = float(matrix[i, j].item())
            x0, y0 = left + j * cell, top + i * cell
            draw.rectangle([x0, y0, x0 + cell - 2, y0 + cell - 2], fill=heat_color(value, vmax))
    img.save(out_path)


def context_variants(context: torch.Tensor, wrong_context: torch.Tensor, empty_context: torch.Tensor) -> Dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1234)
    order = torch.randperm(context.shape[1], generator=generator)
    return {
        "correct": context,
        "empty": empty_context.to(context),
        "wrong": wrong_context.to(context),
        "shuffled": context[:, order].contiguous(),
        "reversed": torch.flip(context, dims=[1]).contiguous(),
        "same": context[:, :1].repeat(1, context.shape[1], 1, 1).contiguous(),
    }


def make_noisy(target: torch.Tensor, sigma: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(target.shape, generator=gen, dtype=torch.float32).to(target)
    sigma_tensor = torch.tensor([sigma * 1000.0], device=target.device, dtype=target.dtype)
    noisy = (1.0 - sigma) * target + sigma * noise
    velocity = noise - target
    return noisy, velocity, sigma_tensor


def masked_mse(error: torch.Tensor) -> torch.Tensor:
    return error.float().pow(2).mean()


def per_slot_l2(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 5:
        raise ValueError(f"Expected [B, C, F, H, W] tensor, got shape={tuple(tensor.shape)}")
    return tensor.float().pow(2).sum(dim=(0, 1, 3, 4)).sqrt()


@torch.no_grad()
def forward_pred(pipe, item: Dict[str, Any], context: torch.Tensor, noisy: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    timestep = timestep.to(device=noisy.device, dtype=noisy.dtype)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=noisy.is_cuda):
        return pipe.model_fn(
            dit=pipe.dit,
            latents=noisy,
            timestep=timestep,
            context=context,
            clip_feature=item["clip_feature"],
            y=item["y"],
            fuse_vae_embedding_in_latents=item["fuse_vae_embedding_in_latents"],
            use_gradient_checkpointing=False,
            use_gradient_checkpointing_offload=False,
        )


def run_prompt_experiments(
    pipe,
    samples: List[Tuple[str, str]],
    cfg: Dict[str, Any],
    empty_context: torch.Tensor,
    out_dir: Path,
    sigmas: List[float],
    slot_max_samples: int | None,
) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
    data_cfg = cfg["data"]
    device = pipe.device
    expected_slots = int(data_cfg["expected_num_slots"])
    loaded = []
    for split, sample_dir in samples:
        item = trainer.load_cached_item(sample_dir, data_cfg, "local_only", device, expected_slots)
        loaded.append((split, sample_dir, item))
    wrong_map = {}
    for idx, (_, _, item) in enumerate(loaded):
        sid = item["sample_id"]
        other = None
        for _, _, candidate in loaded:
            if task_name(candidate["sample_id"]) != task_name(sid):
                other = candidate
                break
        if other is None:
            other = loaded[(idx + 1) % len(loaded)][2]
        wrong_map[sid] = other["conditions"]["context"]

    rows = []
    slot_matrix_sum = torch.zeros(expected_slots, expected_slots, dtype=torch.float64)
    slot_matrix_count = 0

    for sample_index, (split, sample_dir, item) in enumerate(loaded):
        print(f"[prompt] sample {sample_index + 1}/{len(loaded)} {split} {item['sample_id']}", flush=True)
        correct_context = item["conditions"]["context"]
        variants = context_variants(correct_context, wrong_map[item["sample_id"]], empty_context.to(device))
        for sigma in sigmas:
            print(f"[prompt]   sigma={sigma}", flush=True)
            target = item["target_latents"]
            noisy, velocity, timestep = make_noisy(target, sigma, seed=20260715 + sample_index * 100 + int(sigma * 1000))
            correct_pred = forward_pred(pipe, item, variants["correct"], noisy, timestep)
            correct_loss = masked_mse(correct_pred - velocity)
            correct_norm = correct_pred.float().norm()
            for variant_name, variant_context in variants.items():
                pred = correct_pred if variant_name == "correct" else forward_pred(pipe, item, variant_context, noisy, timestep)
                loss = masked_mse(pred - velocity)
                sensitivity = ((pred - correct_pred).float().norm() / (correct_norm + 1e-12)).item()
                rows.append(
                    {
                        "split": split,
                        "sample_id": item["sample_id"],
                        "sigma": sigma,
                        "variant": variant_name,
                        "sensitivity": sensitivity,
                        "loss": float(loss.item()),
                        "delta_loss": float((loss - correct_loss).item()),
                        "correct_loss": float(correct_loss.item()),
                    }
                )
                if variant_name != "correct":
                    del pred
            del correct_pred, noisy, velocity
            torch.cuda.empty_cache()

        if slot_max_samples is not None and slot_matrix_count >= slot_max_samples:
            continue
        print(f"[slot] sample {slot_matrix_count + 1}/{slot_max_samples or len(loaded)} {item['sample_id']}", flush=True)
        sigma = 0.5
        target = item["target_latents"]
        noisy, velocity, timestep = make_noisy(target, sigma, seed=303000 + sample_index)
        correct_pred = forward_pred(pipe, item, correct_context, noisy, timestep)
        wrong_context = wrong_map[item["sample_id"]].to(correct_context)
        denom = per_slot_l2(correct_pred).clamp_min(1e-12)
        for j in range(expected_slots):
            print(f"[slot]   edit slot {j}", flush=True)
            edited = correct_context.clone()
            edited[:, j] = wrong_context[:, j]
            edited_pred = forward_pred(pipe, item, edited, noisy, timestep)
            diff = per_slot_l2(edited_pred - correct_pred) / denom
            slot_matrix_sum[:, j] += diff.detach().cpu().double()
            del edited, edited_pred
        slot_matrix_count += 1
        del correct_pred, noisy, velocity
        torch.cuda.empty_cache()

    with open(out_dir / "prompt_causal_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "prompt_causal_results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    return rows, (slot_matrix_sum / max(1, slot_matrix_count)).float()


def summarize_prompt_rows(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    summary = {}
    for variant in sorted({row["variant"] for row in rows}):
        if variant == "correct":
            continue
        subset = [row for row in rows if row["variant"] == variant]
        summary[variant] = {
            "mean_sensitivity": sum(row["sensitivity"] for row in subset) / len(subset),
            "mean_delta_loss": sum(row["delta_loss"] for row in subset) / len(subset),
            "positive_delta_loss_fraction": sum(row["delta_loss"] > 0 for row in subset) / len(subset),
        }
    return summary


def write_report(out_dir: Path, checkpoint: Dict[str, Any], lora_rows: List[Dict[str, Any]], prompt_summary: Dict[str, Any], slot_matrix: torch.Tensor) -> None:
    def mean_for(attn: str, proj: str) -> float:
        vals = [row["delta_over_base"] for row in lora_rows if row["attn_type"] == attn and row["proj"] == proj]
        return sum(vals) / len(vals) if vals else 0.0

    cross_mean = sum(row["delta_over_base"] for row in lora_rows if row["attn_type"] == "cross_attn") / max(1, sum(row["attn_type"] == "cross_attn" for row in lora_rows))
    self_mean = sum(row["delta_over_base"] for row in lora_rows if row["attn_type"] == "self_attn") / max(1, sum(row["attn_type"] == "self_attn" for row in lora_rows))
    diag = torch.diag(slot_matrix).mean().item()
    off = ((slot_matrix.sum() - torch.diag(slot_matrix).sum()) / (slot_matrix.numel() - slot_matrix.shape[0])).item()

    lines = [
        "# Exp4 Step 2500 Prompt Usage Diagnostics",
        "",
        f"checkpoint_global_step: {checkpoint.get('global_step')}",
        f"checkpoint_loss: {checkpoint.get('loss')}",
        "",
        "## LoRA Delta Summary",
        "",
        f"mean self_attn delta/base: {self_mean:.6g}",
        f"mean cross_attn delta/base: {cross_mean:.6g}",
        f"cross/self ratio: {cross_mean / (self_mean + 1e-12):.6g}",
        "",
        "| attention | q | k | v | o | k_img | v_img |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| self | {mean_for('self_attn','q'):.3e} | {mean_for('self_attn','k'):.3e} | {mean_for('self_attn','v'):.3e} | {mean_for('self_attn','o'):.3e} | - | - |",
        f"| cross | {mean_for('cross_attn','q'):.3e} | {mean_for('cross_attn','k'):.3e} | {mean_for('cross_attn','v'):.3e} | {mean_for('cross_attn','o'):.3e} | {mean_for('cross_attn','k_img'):.3e} | {mean_for('cross_attn','v_img'):.3e} |",
        "",
        "See `lora_delta_stats.csv`, `lora_self_attn_delta_over_base.jpg`, and `lora_cross_attn_delta_over_base.jpg`.",
        "",
        "## Prompt Causal Summary",
        "",
        "| variant | mean sensitivity | mean delta loss | fraction delta loss > 0 |",
        "|---|---:|---:|---:|",
    ]
    for variant, item in prompt_summary.items():
        lines.append(
            f"| {variant} | {item['mean_sensitivity']:.6g} | {item['mean_delta_loss']:.6g} | {item['positive_delta_loss_fraction']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Sensitivity is ||pred_variant - pred_correct|| / ||pred_correct||.",
            "Positive delta loss means correct prompt was better for that sample/sigma.",
            "",
            "## Single Slot Intervention",
            "",
            f"mean diagonal sensitivity at sigma=0.5: {diag:.6g}",
            f"mean off-diagonal sensitivity at sigma=0.5: {off:.6g}",
            f"diag/off ratio: {diag / (off + 1e-12):.6g}",
            "",
            "See `slot_intervention_sigma_0p5.jpg`.",
            "",
            "## Interpretation Rules",
            "",
            "- Near-zero sensitivities mean the checkpoint is mostly ignoring local prompts.",
            "- Positive delta loss for wrong/shuffled/reversed means the correct prompt helps.",
            "- A diagonal-heavy slot matrix means per-slot routing is working.",
            "- A flat slot matrix means prompt changes act globally rather than locally.",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/agibot/train/exp4_aigbot_final_local_only_lora_rank128.json")
    parser.add_argument("--checkpoint", default="runs/exp4_aigbot_final_local_only_lora_rank128/checkpoints/trainable_step_002500.pt")
    parser.add_argument("--out_dir", default="runs/diagnostics/exp4_step2500_failure_analysis")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train_count", type=int, default=4)
    parser.add_argument("--val_count", type=int, default=4)
    parser.add_argument("--sigmas", default="0.2,0.5,0.8")
    parser.add_argument("--slot_max_samples", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_json(Path(args.config))
    cfg["train"]["conditioning_mode"] = "local_only"
    samples = select_samples(cfg, args.train_count, args.val_count)

    print("[stage] encode empty prompt context")
    empty_context = encode_empty_context(cfg, args.device, int(cfg["data"]["expected_num_slots"]))

    print("[stage] build Exp4 DiT")
    pipe = trainer.build_pipe(cfg, "local_only", local_rank=0)
    pipe.load_models_to_device(["dit"])
    pipe.dit.eval()
    groups, named_trainable = trainer.configure_trainable(pipe.dit, cfg["train"], rank=0)
    expected_names = {name for name, _ in pipe.dit.named_parameters() if _.requires_grad}
    print(f"[check] expected trainable tensors={len(expected_names)}")
    checkpoint = strict_load_trainable(pipe.dit, Path(args.checkpoint))
    pipe.dit.eval()
    print(f"[check] strict checkpoint load ok keys={len(checkpoint['trainable_state_dict'])}")

    print("[stage] lora necropsy")
    lora_rows = lora_delta_stats(pipe.dit, out_dir)
    draw_lora_heatmap(lora_rows, out_dir / "lora_self_attn_delta_over_base.jpg", "self_attn", ["q", "k", "v", "o"])
    draw_lora_heatmap(lora_rows, out_dir / "lora_cross_attn_delta_over_base.jpg", "cross_attn", ["q", "k", "v", "o", "k_img", "v_img"])

    print("[stage] prompt causal and slot intervention")
    sigmas = [float(item) for item in args.sigmas.split(",") if item.strip()]
    rows, slot_matrix = run_prompt_experiments(
        pipe=pipe,
        samples=samples,
        cfg=cfg,
        empty_context=empty_context,
        out_dir=out_dir,
        sigmas=sigmas,
        slot_max_samples=args.slot_max_samples,
    )
    torch.save(slot_matrix, out_dir / "slot_intervention_sigma_0p5.pt")
    draw_matrix_heatmap(slot_matrix, out_dir / "slot_intervention_sigma_0p5.jpg", "slot intervention sigma=0.5")
    prompt_summary = summarize_prompt_rows(rows)
    with open(out_dir / "prompt_summary.json", "w", encoding="utf-8") as f:
        json.dump(prompt_summary, f, indent=2)
    write_report(out_dir, checkpoint, lora_rows, prompt_summary, slot_matrix)
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
