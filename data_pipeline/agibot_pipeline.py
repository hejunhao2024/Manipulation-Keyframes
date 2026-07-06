import argparse
import asyncio
import base64
import io
import json
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont
from openai import AsyncOpenAI


DEFAULT_MODEL = "Qwen3-VL-32B-Instruct"
DEFAULT_ENDPOINTS = "http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1"

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
]


# ============================================================
# Utilities
# ============================================================

def load_font(size: int):
    for fp in FONT_CANDIDATES:
        if Path(fp).exists():
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()


def pil_to_data_url(img: Image.Image, quality: int = 92) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def image_file_to_data_url(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def extract_json(text: str) -> Dict[str, Any]:
    """Extract a JSON object from model output."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError("top-level JSON is not an object")
        return obj
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise
        obj = json.loads(m.group(0))
        if not isinstance(obj, dict):
            raise ValueError("top-level JSON is not an object")
        return obj


def as_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v < 0:
            return 0.0
        if v > 1:
            return 1.0
        return v
    except Exception:
        return default


def normalize_local_id(x: Any, prefix: str = "F") -> Optional[str]:
    s = str(x).strip().upper()
    m = re.search(rf"{prefix}?\s*(\d+)", s)
    if not m:
        return None
    return f"{prefix}{int(m.group(1)):02d}"


def list_numeric_images(d: Path) -> List[Path]:
    imgs = [p for p in d.glob("*.jpg") if p.stem.isdigit()]
    imgs.sort(key=lambda p: int(p.stem))
    return imgs


def find_sample_dirs(input_root: Path, required_frames: int = 60) -> List[Path]:
    """
    Recursively find directories that directly contain exactly required_frames numeric jpgs.
    """
    out: List[Path] = []

    root_imgs = list_numeric_images(input_root)
    if len(root_imgs) == required_frames:
        out.append(input_root)

    for d in input_root.rglob("*"):
        if not d.is_dir():
            continue
        imgs = list_numeric_images(d)
        if len(imgs) == required_frames:
            out.append(d)

    uniq = []
    seen = set()
    for d in sorted(out, key=lambda x: str(x)):
        s = str(d)
        if s not in seen:
            seen.add(s)
            uniq.append(d)
    return uniq


def uniform_indices(n: int, k: int) -> List[int]:
    if n <= 0:
        return []
    if k >= n:
        return list(range(n))
    if k <= 1:
        return [0]
    return sorted(set(round(i * (n - 1) / (k - 1)) for i in range(k)))


def safe_json_dumps(obj: Any, max_chars: Optional[int] = None) -> str:
    s = json.dumps(obj, ensure_ascii=False, indent=2)
    if max_chars is not None and len(s) > max_chars:
        return s[:max_chars] + "\n... [truncated]"
    return s


# ============================================================
# Human task hints and sample selection
# ============================================================

def extract_observation_id(path: Path) -> Optional[str]:
    """Extract observations/{task_id} from a sample path."""
    parts = path.parts
    if "observations" not in parts:
        return None
    i = parts.index("observations")
    if i + 1 < len(parts):
        return parts[i + 1]
    return None


def normalize_task_hint_record(obs_id: str, record: Any) -> Dict[str, Any]:
    """Normalize a task-hints entry into a compact dict used by prompts.

    Supported formats:
    1. {"tasks": {"327": {"match": "observations/327", "task_type": "...", "task_hint": "..."}}}
    2. {"tasks": {"327": {"核心任务": "...", "标注说明": "...", "容易误解": "..."}}}
    3. {"327": "..."} or {"327": {"task_hint": "..."}}
    """
    if isinstance(record, str):
        return {
            "observation_id": obs_id,
            "match": f"observations/{obs_id}",
            "task_type": "",
            "task_hint": record.strip(),
            "raw": record,
        }
    if not isinstance(record, dict):
        return {
            "observation_id": obs_id,
            "match": f"observations/{obs_id}",
            "task_type": "",
            "task_hint": "",
            "raw": record,
        }

    task_type = str(record.get("task_type") or record.get("task_type_cn") or record.get("任务类型") or record.get("核心任务类型") or "").strip()
    task_hint = str(record.get("task_hint") or "").strip()

    # Chinese/simple template compatibility.
    if not task_hint:
        core = str(record.get("核心任务") or record.get("core_task") or "").strip()
        guide = str(record.get("标注说明") or record.get("annotation_guidance") or "").strip()
        avoid = str(record.get("容易误解") or record.get("avoid_misinterpretation") or "").strip()
        parts = []
        if core:
            parts.append(f"Core task: {core}")
        if guide:
            parts.append(f"Annotation guidance: {guide}")
        if avoid:
            parts.append(f"Common mistakes to avoid: {avoid}")
        task_hint = "\n".join(parts)

    return {
        "observation_id": obs_id,
        "match": str(record.get("match") or f"observations/{obs_id}").strip(),
        "task_type": task_type,
        "task_hint": task_hint,
        "raw": record,
    }


def load_task_hints(path: str) -> Dict[str, Dict[str, Any]]:
    """Load human-provided task hints.

    Returns a mapping: observation_id -> normalized task hint record.
    If the file is missing, returns an empty mapping.
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        print(f"[task_hints] not found: {p}; running without human task priors")
        return {}
    obj = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and "tasks" in obj and isinstance(obj["tasks"], dict):
        raw_tasks = obj["tasks"]
    elif isinstance(obj, dict):
        raw_tasks = obj
    else:
        raise ValueError(f"Unsupported task hints format: {path}")

    hints: Dict[str, Dict[str, Any]] = {}
    for obs_id, rec in raw_tasks.items():
        if str(obs_id).startswith("_"):
            continue
        hints[str(obs_id)] = normalize_task_hint_record(str(obs_id), rec)
    print(f"[task_hints] loaded {len(hints)} observation-level hints from {p}")
    return hints


def get_task_hint_for_sample(task_hints: Dict[str, Dict[str, Any]], sample_dir: Path) -> Dict[str, Any]:
    obs_id = extract_observation_id(sample_dir)
    if obs_id and obs_id in task_hints:
        return task_hints[obs_id]
    return {
        "observation_id": obs_id or "",
        "match": "",
        "task_type": "",
        "task_hint": "",
        "raw": None,
    }


def format_task_hint_for_prompt(task_hint: Dict[str, Any], max_chars: int = 2500) -> str:
    hint = str(task_hint.get("task_hint") or "").strip()
    task_type = str(task_hint.get("task_type") or "").strip()
    obs_id = str(task_hint.get("observation_id") or "").strip()
    if not hint and not task_type:
        return "No human task prior is available for this sample."
    lines = []
    if obs_id:
        lines.append(f"observation_id: {obs_id}")
    if task_type:
        lines.append(f"task_type: {task_type}")
    if hint:
        lines.append("task_hint:")
        lines.append(hint)
    s = "\n".join(lines).strip()
    if len(s) > max_chars:
        s = s[:max_chars] + "\n... [truncated]"
    return s


def merge_task_hint_into_context(context: Dict[str, Any], task_hint: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(context)
    out["human_task_prior"] = {
        "observation_id": task_hint.get("observation_id", ""),
        "task_type": task_hint.get("task_type", ""),
        "task_hint": task_hint.get("task_hint", ""),
    }
    return out


def select_sample_dirs_for_run(
    sample_dirs: List[Path],
    strategy: str,
    max_samples: int,
    start_index: int,
    seed: int,
) -> List[Path]:
    """Select sample dirs for one run.

    sequential: original sorted order, with start_index then max_samples.
    random: random subset from all samples after start_index.
    stratified: split the sorted list into max_samples bins and randomly choose one from each bin.
    """
    if start_index > 0:
        sample_dirs = sample_dirs[start_index:]

    if max_samples <= 0 or max_samples >= len(sample_dirs):
        if strategy == "random":
            rng = random.Random(seed)
            sample_dirs = list(sample_dirs)
            rng.shuffle(sample_dirs)
        return sample_dirs

    if strategy == "sequential":
        return sample_dirs[:max_samples]

    rng = random.Random(seed)

    if strategy == "random":
        picked = rng.sample(sample_dirs, max_samples)
        picked.sort(key=lambda x: str(x))
        return picked

    if strategy == "stratified":
        n = len(sample_dirs)
        picked: List[Path] = []
        used = set()
        for i in range(max_samples):
            lo = int(i * n / max_samples)
            hi = int((i + 1) * n / max_samples)
            if hi <= lo:
                hi = min(lo + 1, n)
            candidates = [x for x in sample_dirs[lo:hi] if str(x) not in used]
            if not candidates:
                continue
            x = rng.choice(candidates)
            picked.append(x)
            used.add(str(x))

        if len(picked) < max_samples:
            rest = [x for x in sample_dirs if str(x) not in used]
            need = min(max_samples - len(picked), len(rest))
            picked.extend(rng.sample(rest, need))

        picked.sort(key=lambda x: str(x))
        return picked[:max_samples]

    raise ValueError(f"Unknown sample strategy: {strategy}")


class EndpointPool:
    def __init__(self, endpoints: List[str]):
        self.clients: List[Tuple[str, AsyncOpenAI]] = [
            (ep, AsyncOpenAI(base_url=ep, api_key="EMPTY")) for ep in endpoints
        ]
        if not self.clients:
            raise ValueError("No endpoints provided")
        self.idx = 0
        self.lock = asyncio.Lock()

    async def next(self) -> Tuple[str, AsyncOpenAI]:
        async with self.lock:
            ep, client = self.clients[self.idx % len(self.clients)]
            self.idx += 1
            return ep, client


async def call_vlm_json(
    endpoint_pool: EndpointPool,
    sem: asyncio.Semaphore,
    model: str,
    content: List[Dict[str, Any]],
    max_tokens: int,
    retries: int,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    last_error = None
    last_raw = ""
    for attempt in range(retries + 1):
        try:
            async with sem:
                endpoint, client = await endpoint_pool.next()
                t0 = time.time()
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": content}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                latency = time.time() - t0
            raw = resp.choices[0].message.content or ""
            last_raw = raw
            obj = extract_json(raw)
            obj["_endpoint"] = endpoint
            obj["_latency_sec"] = round(latency, 4)
            return obj
        except Exception as e:
            last_error = repr(e)
            await asyncio.sleep(0.8 * (attempt + 1) + random.random() * 0.4)
    return {
        "_endpoint": "fallback",
        "_error": last_error,
        "_raw": last_raw,
    }


# ============================================================
# Selection
# Hard-coded 4 batches:
# batch 0: frames [0,14], select 6, and F00 must be kept
# batch 1: frames [15,29], select 5
# batch 2: frames [30,44], select 5
# batch 3: frames [45,59], select 5
# total = 21
# ============================================================

BATCH_SPECS = [
    {"start": 0, "end": 15, "quota": 6, "must_keep_first": True},
    {"start": 15, "end": 30, "quota": 5, "must_keep_first": False},
    {"start": 30, "end": 45, "quota": 5, "must_keep_first": False},
    {"start": 45, "end": 60, "quota": 5, "must_keep_first": False},
]


def make_selection_labeled_data_url(src: Path, local_id: str, global_idx: int) -> str:
    img = Image.open(src).convert("RGB")
    w, h = img.size
    bar_h = max(60, int(h * 0.10))
    canvas = Image.new("RGB", (w, h + bar_h), "white")
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = load_font(34)
    small_font = load_font(20)

    draw.rectangle([0, h, w, h + bar_h], fill=(245, 245, 245))
    draw.line([0, h, w, h], fill=(180, 180, 180), width=2)
    draw.text((16, h + 6), local_id, fill=(0, 0, 0), font=font)
    draw.text((16, h + 36), f"global_frame={global_idx}, source={src.name}", fill=(70, 70, 70), font=small_font)
    return pil_to_data_url(canvas)


def make_keyframe_labeled_data_url(src: Path, label: str, extra: str = "") -> str:
    img = Image.open(src).convert("RGB")
    w, h = img.size
    bar_h = max(58, int(h * 0.09))
    canvas = Image.new("RGB", (w, h + bar_h), "white")
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = load_font(30)
    small_font = load_font(18)
    draw.rectangle([0, h, w, h + bar_h], fill=(245, 245, 245))
    draw.line([0, h, w, h], fill=(180, 180, 180), width=2)
    draw.text((14, h + 5), label, fill=(0, 0, 0), font=font)
    if extra:
        draw.text((14, h + 35), extra, fill=(70, 70, 70), font=small_font)
    return pil_to_data_url(canvas)


def make_contact_sheet_data_url(
    paths: List[Path],
    labels: List[str],
    cols: int = 7,
    thumb_w: int = 300,
    quality: int = 88,
) -> str:
    """Make a labeled contact sheet for global temporal context."""
    assert len(paths) == len(labels)
    if not paths:
        raise ValueError("empty paths")

    thumbs: List[Image.Image] = []
    font = load_font(max(18, thumb_w // 13))
    small_font = load_font(max(12, thumb_w // 22))
    pad = max(8, thumb_w // 30)
    label_h = max(42, thumb_w // 6)

    for p, lab in zip(paths, labels):
        img = Image.open(p).convert("RGB")
        w, h = img.size
        thumb_h = max(1, round(h * thumb_w / w))
        img = img.resize((thumb_w, thumb_h), Image.BICUBIC)
        tile = Image.new("RGB", (thumb_w, thumb_h + label_h), "white")
        tile.paste(img, (0, 0))
        draw = ImageDraw.Draw(tile)
        draw.rectangle([0, thumb_h, thumb_w, thumb_h + label_h], fill=(245, 245, 245))
        draw.line([0, thumb_h, thumb_w, thumb_h], fill=(180, 180, 180), width=1)
        draw.text((8, thumb_h + 4), lab, fill=(0, 0, 0), font=font)
        draw.text((8, thumb_h + 26), p.name, fill=(70, 70, 70), font=small_font)
        thumbs.append(tile)

    rows = (len(thumbs) + cols - 1) // cols
    tile_w = thumbs[0].size[0]
    tile_h = thumbs[0].size[1]
    sheet = Image.new("RGB", (cols * tile_w + (cols + 1) * pad, rows * tile_h + (rows + 1) * pad), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, tile in enumerate(thumbs):
        r = idx // cols
        c = idx % cols
        x = pad + c * (tile_w + pad)
        y = pad + r * (tile_h + pad)
        sheet.paste(tile, (x, y))
        draw.rectangle([x, y, x + tile_w - 1, y + tile_h - 1], outline=(180, 180, 180), width=1)
    return pil_to_data_url(sheet, quality=quality)


def build_selection_content(
    batch_paths: List[Path],
    global_indices: List[int],
    batch_idx: int,
    quota: int,
    must_keep_first: bool,
) -> List[Dict[str, Any]]:
    local_ids = [f"F{i:02d}" for i in range(len(batch_paths))]
    force_rule = ""
    if must_keep_first:
        force_rule = "Frame F00 is the first frame of the whole 60-frame video and MUST be included."

    prompt = f"""
You are selecting keyframes from a 60-frame dual-arm tabletop manipulation video.

This is batch {batch_idx + 1} of 4.
You will see exactly 15 consecutive frames from the video.
You must select EXACTLY {quota} frames from this batch.

{force_rule}

Selection goal:
Choose keyframes that best preserve the sparse long-horizon visual trajectory of the dual-arm manipulation process.
A keyframe may capture an atomic action state, a transition waypoint, a static pose, or a visually important state that helps preserve the trajectory.

Important rules:
1. Preserve the task progression.
2. Do not make the manipulated object appear to teleport from one place to another.
3. If an arm carries an object across space, preserve meaningful intermediate waypoints when visible.
4. Keep frames where the left arm or right arm reaches a new meaningful pose, region, contact relation, or gripper state.
5. Remove only clearly redundant near-duplicates.
6. Return strict JSON only.

Valid frame IDs:
{local_ids}

Output schema:
{{
  "selected_ids": ["F00", "..."],
  "reason": "Short reason for the selection.",
  "batch_summary": "One concise sentence summarizing the visible progression in this 15-frame batch. This is only a soft note and may be imperfect.",
  "selected_frame_notes": [
    {{
      "id": "F00",
      "note": "Short neutral note about this selected frame: arm pose, possible object, and whether it is approach/hold/release/transition/static. Avoid overclaiming grasp from overlap."
    }}
  ]
}}
""".strip()

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for lid, path, gidx in zip(local_ids, batch_paths, global_indices):
        content.append({
            "type": "text",
            "text": f"Frame {lid}, global_frame={gidx}, filename={path.name}.",
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": make_selection_labeled_data_url(path, lid, gidx)},
        })
    return content


def repair_selection_exact(
    raw_selected_ids: List[Any],
    valid_ids: List[str],
    quota: int,
    required_ids: List[str],
) -> List[str]:
    valid_set = set(valid_ids)

    selected: List[str] = []
    for x in raw_selected_ids:
        fid = normalize_local_id(x, "F")
        if fid in valid_set and fid not in selected:
            selected.append(fid)

    for rid in required_ids:
        if rid in valid_set and rid not in selected:
            selected.insert(0, rid)

    if len(selected) < quota:
        missing = [x for x in valid_ids if x not in selected]
        need = quota - len(selected)
        if missing:
            if need == 1:
                add = [missing[len(missing) // 2]]
            else:
                idxs = [round(i * (len(missing) - 1) / (need - 1)) for i in range(need)]
                add = []
                for idx in idxs:
                    x = missing[int(idx)]
                    if x not in add:
                        add.append(x)
                for x in missing:
                    if len(add) >= need:
                        break
                    if x not in add:
                        add.append(x)
            selected.extend(add[:need])

    if len(selected) > quota:
        req = [x for x in valid_ids if x in required_ids]
        req_set = set(req)
        rest = [x for x in selected if x not in req_set]
        keep_rest_n = max(0, quota - len(req))
        if keep_rest_n == 0:
            selected = req[:quota]
        elif len(rest) <= keep_rest_n:
            selected = req + rest
        else:
            if keep_rest_n == 1:
                picked = [rest[len(rest) // 2]]
            else:
                idxs = [round(i * (len(rest) - 1) / (keep_rest_n - 1)) for i in range(keep_rest_n)]
                picked = []
                for idx in idxs:
                    x = rest[int(idx)]
                    if x not in picked:
                        picked.append(x)
                for x in rest:
                    if len(picked) >= keep_rest_n:
                        break
                    if x not in picked:
                        picked.append(x)
            selected = req + picked[:keep_rest_n]

    selected_set = set(selected)
    selected = [x for x in valid_ids if x in selected_set]
    return selected[:quota]


async def select_one_batch(
    endpoint_pool: EndpointPool,
    sem: asyncio.Semaphore,
    model: str,
    batch_paths: List[Path],
    global_indices: List[int],
    batch_idx: int,
    quota: int,
    must_keep_first: bool,
    max_tokens: int,
    retries: int,
) -> Dict[str, Any]:
    content = build_selection_content(batch_paths, global_indices, batch_idx, quota, must_keep_first)
    valid_ids = [f"F{i:02d}" for i in range(len(batch_paths))]
    required_ids = ["F00"] if must_keep_first else []

    obj = await call_vlm_json(endpoint_pool, sem, model, content, max_tokens, retries)
    if "_error" not in obj:
        selected_local_ids = repair_selection_exact(
            obj.get("selected_ids", []),
            valid_ids,
            quota,
            required_ids,
        )
        selected_global_indices = [global_indices[int(x[1:])] for x in selected_local_ids]
        raw_notes = obj.get("selected_frame_notes", [])
        note_map: Dict[str, str] = {}
        if isinstance(raw_notes, list):
            for item in raw_notes:
                if not isinstance(item, dict):
                    continue
                fid = normalize_local_id(item.get("id"), "F")
                if fid:
                    note_map[fid] = str(item.get("note", "")).strip()
        selected_frame_notes = [
            {
                "local_id": lid,
                "global_index": global_indices[int(lid[1:])],
                "note": note_map.get(lid, ""),
            }
            for lid in selected_local_ids
        ]
        return {
            "batch_idx": batch_idx,
            "selected_local_ids": selected_local_ids,
            "selected_global_indices": selected_global_indices,
            "reason": str(obj.get("reason", "")).strip(),
            "batch_summary": str(obj.get("batch_summary", "")).strip(),
            "selected_frame_notes": selected_frame_notes,
            "endpoint": obj.get("_endpoint"),
            "latency_sec": obj.get("_latency_sec"),
        }

    # Fallback: exact evenly spaced selection.
    if quota == 1:
        idxs = [0]
    else:
        idxs = [round(i * (len(batch_paths) - 1) / (quota - 1)) for i in range(quota)]
    if must_keep_first and 0 not in idxs:
        idxs = [0] + idxs
    idxs = sorted(set(int(x) for x in idxs))
    while len(idxs) < quota:
        for i in range(len(batch_paths)):
            if i not in idxs:
                idxs.append(i)
            if len(idxs) >= quota:
                break
    idxs = sorted(idxs[:quota])
    return {
        "batch_idx": batch_idx,
        "selected_local_ids": [f"F{i:02d}" for i in idxs],
        "selected_global_indices": [global_indices[i] for i in idxs],
        "reason": "Fallback exact selection.",
        "batch_summary": "Fallback selection; no visual batch summary available.",
        "selected_frame_notes": [
            {"local_id": f"F{i:02d}", "global_index": global_indices[i], "note": "fallback selected frame"}
            for i in idxs
        ],
        "endpoint": "fallback",
        "error": obj.get("_error"),
    }


async def select_21_keyframes(
    endpoint_pool: EndpointPool,
    sem: asyncio.Semaphore,
    model: str,
    images: List[Path],
    max_tokens: int,
    retries: int,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    if len(images) < 60:
        raise ValueError(f"Need at least 60 frames, got {len(images)}")
    images = images[:60]

    tasks = []
    for bi, spec in enumerate(BATCH_SPECS):
        s, e = spec["start"], spec["end"]
        tasks.append(
            select_one_batch(
                endpoint_pool=endpoint_pool,
                sem=sem,
                model=model,
                batch_paths=images[s:e],
                global_indices=list(range(s, e)),
                batch_idx=bi,
                quota=spec["quota"],
                must_keep_first=spec["must_keep_first"],
                max_tokens=max_tokens,
                retries=retries,
            )
        )

    decisions = await asyncio.gather(*tasks)
    decisions.sort(key=lambda x: x["batch_idx"])

    selected_indices: List[int] = []
    for d in decisions:
        selected_indices.extend(d["selected_global_indices"])

    selected_indices = sorted(selected_indices)

    if len(selected_indices) != 21:
        raise RuntimeError(f"Internal error: expected 21 selected frames, got {len(selected_indices)}")
    if selected_indices[0] != 0:
        raise RuntimeError("Internal error: first selected frame must be source frame 0")

    return selected_indices, decisions


def select_21_keyframes_uniform() -> Tuple[List[int], List[Dict[str, Any]]]:
    """Fast fallback selector: keep the same [6,5,5,5] temporal coverage without VLM calls."""
    selected_indices: List[int] = []
    decisions: List[Dict[str, Any]] = []
    for bi, spec in enumerate(BATCH_SPECS):
        s, e, quota = spec["start"], spec["end"], spec["quota"]
        local_n = e - s
        if quota <= 1:
            local = [0]
        else:
            local = [round(i * (local_n - 1) / (quota - 1)) for i in range(quota)]
        if spec.get("must_keep_first", False) and 0 not in local:
            local = [0] + local
        local = sorted(set(int(x) for x in local))
        while len(local) < quota:
            for j in range(local_n):
                if j not in local:
                    local.append(j)
                if len(local) >= quota:
                    break
        local = sorted(local[:quota])
        global_ids = [s + j for j in local]
        selected_indices.extend(global_ids)
        decisions.append({
            "batch_idx": bi,
            "selected_local_ids": [f"F{j:02d}" for j in local],
            "selected_global_indices": global_ids,
            "reason": "Uniform fast selection without VLM.",
            "batch_summary": "Uniform temporal selection; no visual batch summary available.",
            "selected_frame_notes": [
                {"local_id": f"F{j:02d}", "global_index": s + j, "note": "uniform selected frame"}
                for j in local
            ],
            "endpoint": "uniform",
        })
    selected_indices = sorted(selected_indices)
    if len(selected_indices) != 21 or selected_indices[0] != 0:
        raise RuntimeError("Uniform selector internal error")
    return selected_indices, decisions


# ============================================================
# Sequence context
# ============================================================

def build_sequence_context_content(
    keyframe_paths: List[Path],
    contact_sheet_url: str,
    global_image_count: int,
    task_hint: Dict[str, Any],
) -> List[Dict[str, Any]]:
    # v9: this pass is a retrospective global task plan, not a final caption.
    # It uses 1 contact sheet + at most 10 individual images, so each request stays <= 11 images.
    chosen = uniform_indices(len(keyframe_paths), min(global_image_count, 10))
    task_prior_text = format_task_hint_for_prompt(task_hint)
    prompt = f"""
You are building a GLOBAL TASK PLAN for a sparse 21-keyframe robot manipulation sequence with two visible arms.

Human-provided high-level task prior:
{task_prior_text}

This is a retrospective annotation task: you may use later keyframes to infer what earlier approach/hover frames were moving toward. However, do not overclaim contact in a specific frame unless the frame-level evidence supports it.

Important:
- The human task prior is reliable for the task type and intended workflow. Use it to avoid direction-level hallucinations.
- Frame-level states still require visual evidence. Do not force the image to match the prior if the frames clearly contradict it.
- Do NOT assume every task is pick-and-place. Some tasks are scanning, tool-use, wiping, ironing, washing, folding, opening/closing appliances, packing, or collaborative carrying.
- Distinguish tools/devices from target products/items. For example, a scanner, iron, brush, faucet, kettle, or air-column packaging material should not be mislabeled as an ordinary product unless the task prior says so.
- The 21 keyframes are sparse and may include static frames, transition waypoints, or ambiguous poses.
- Do NOT infer a grasp merely from 2D visual overlap between the gripper and an object.
- A grasp/holding claim needs either visible object-between-fingers evidence or motion evidence across nearby frames.
- This plan will guide overlap-chunk frame annotation. It should identify the intended task sequence, functional roles of the arms, and coarse segment boundaries, but it is NOT the final global prompt.

You will receive:
1. A labeled contact sheet with all 21 keyframes K00-K20 in temporal order.
2. {len(chosen)} individual keyframes for higher-resolution inspection.

Return strict JSON only:
{{
  "scene_summary": "short description of scene and camera view",
  "task_type": "pick_and_place | handoff | tool_use | scanning | cleaning | washing | folding | ironing | packing | appliance_operation | collaborative_carrying | uncertain",
  "task_goal": "task-level goal in one sentence",
  "left_arm_overall_role": "functional visual role of the arm appearing on the left side, e.g. item holder, tool user, bag opener, door opener, inactive, uncertain",
  "right_arm_overall_role": "functional visual role of the arm appearing on the right side, e.g. item holder, scanner/tool user, container closer, inactive, uncertain",
  "target_container": "target container if visible, otherwise unknown",
  "visible_object_categories": ["object categories visible in the workspace"],
  "tools_or_devices": ["tools/devices used in the task, if any"],
  "products_or_target_items": ["items being acted on, transferred, scanned, cleaned, packed, folded, or carried"],
  "likely_manipulated_objects_order": ["ordered objects/items acted on if visually supported; for tool-use tasks list target products/items, not the tool itself unless picked up or placed"],
  "planned_segments": [
    {{
      "frame_range": [0, 6],
      "intended_object": "object category or unknown/uncertain",
      "segment_summary": "coarse retrospective description of this segment",
      "phase_sequence": ["approach_or_hover", "grasp_or_lift", "transport", "release_or_retreat"],
      "confidence": 0.0
    }}
  ],
  "coarse_timeline": [
    {{
      "frame_range": [0, 3],
      "description": "coarse event or transition, not too detailed",
      "likely_object": "object or unknown/uncertain",
      "confidence": 0.0
    }}
  ],
  "annotation_warnings": [
    "warnings useful for later frame prompts, e.g. do not confuse hovering with holding"
  ]
}}
""".strip()
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.append({"type": "text", "text": "Contact sheet of all selected keyframes K00-K20:"})
    content.append({"type": "image_url", "image_url": {"url": contact_sheet_url}})
    for idx in chosen:
        p = keyframe_paths[idx]
        content.append({"type": "text", "text": f"Individual keyframe K{idx:02d}, filename={p.name}."})
        content.append({"type": "image_url", "image_url": {"url": make_keyframe_labeled_data_url(p, f'K{idx:02d}', p.name)}})
    return content


async def build_sequence_context(
    endpoint_pool: EndpointPool,
    sem: asyncio.Semaphore,
    model: str,
    keyframe_paths: List[Path],
    contact_sheet_url: str,
    global_image_count: int,
    max_tokens: int,
    retries: int,
    task_hint: Dict[str, Any],
) -> Dict[str, Any]:
    content = build_sequence_context_content(keyframe_paths, contact_sheet_url, global_image_count, task_hint)
    obj = await call_vlm_json(endpoint_pool, sem, model, content, max_tokens, retries)
    if "_error" in obj:
        return {
            "scene_summary": "unknown dual-arm robot manipulation sequence",
            "task_type": task_hint.get("task_type", "unknown") or "unknown",
            "task_goal": "",
            "left_arm_overall_role": "unknown",
            "right_arm_overall_role": "unknown",
            "target_container": "unknown",
            "visible_object_categories": [],
            "tools_or_devices": [],
            "products_or_target_items": [],
            "likely_manipulated_objects_order": [],
            "planned_segments": [],
            "coarse_timeline": [],
            "annotation_warnings": [
                "Fallback context. Be conservative about grasping/holding claims."
            ],
            "endpoint": "fallback",
            "error": obj.get("_error"),
        }
    return {
        "scene_summary": str(obj.get("scene_summary", "")).strip(),
        "task_type": str(obj.get("task_type", "")).strip() or str(task_hint.get("task_type", "")).strip(),
        "task_goal": str(obj.get("task_goal", "")).strip(),
        "left_arm_overall_role": str(obj.get("left_arm_overall_role", "")).strip(),
        "right_arm_overall_role": str(obj.get("right_arm_overall_role", "")).strip(),
        "target_container": str(obj.get("target_container", "")).strip(),
        "visible_object_categories": obj.get("visible_object_categories", []),
        "tools_or_devices": obj.get("tools_or_devices", []),
        "products_or_target_items": obj.get("products_or_target_items", []),
        "likely_manipulated_objects_order": obj.get("likely_manipulated_objects_order", []),
        "planned_segments": obj.get("planned_segments", []),
        "coarse_timeline": obj.get("coarse_timeline", []),
        "annotation_warnings": obj.get("annotation_warnings", []),
        "endpoint": obj.get("_endpoint"),
        "latency_sec": obj.get("_latency_sec"),
    }


def make_empty_sequence_context(reason: str = "disabled", task_hint: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Lightweight context used when the sequence-context VLM pass is skipped."""
    task_hint = task_hint or {}
    return {
        "scene_summary": "unknown robot manipulation sequence",
        "task_type": task_hint.get("task_type", "") or "unknown",
        "task_goal": "",
        "left_arm_overall_role": "unknown",
        "right_arm_overall_role": "unknown",
        "target_container": "unknown",
        "visible_object_categories": [],
        "tools_or_devices": [],
        "products_or_target_items": [],
        "likely_manipulated_objects_order": [],
        "planned_segments": [],
        "coarse_timeline": [],
        "annotation_warnings": [
            "No sequence-context pass was used. Be conservative about grasping/holding claims.",
            "Treat left/right only as visual identifiers; infer each arm state from nearby-frame evidence."
        ],
        "endpoint": reason,
    }


def build_sequence_context_from_selection(decisions: List[Dict[str, Any]], task_hint: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build a cheap soft sequence context by reusing the VLM keyframe-selection outputs.

    This avoids the extra expensive sequence-context VLM call. The notes are treated as
    weak priors only, because selection is optimized for temporal coverage rather than
    exact object/contact annotation.
    """
    timeline: List[Dict[str, Any]] = []
    selected_notes: List[Dict[str, Any]] = []
    for d in sorted(decisions, key=lambda x: x.get("batch_idx", 0)):
        gids = d.get("selected_global_indices", []) or []
        if gids:
            fr = [int(min(gids)), int(max(gids))]
        else:
            fr = [0, 0]
        desc = str(d.get("batch_summary") or d.get("reason") or "").strip()
        timeline.append({
            "frame_range": fr,
            "description": desc,
            "likely_object": "unknown",
            "confidence": 0.35 if desc else 0.0,
        })
        for note in d.get("selected_frame_notes", []) or []:
            if isinstance(note, dict):
                selected_notes.append(note)
    task_hint = task_hint or {}
    return {
        "scene_summary": "soft context reused from the VLM keyframe-selection stage",
        "task_type": task_hint.get("task_type", "") or "unknown",
        "task_goal": "",
        "left_arm_overall_role": "unknown",
        "right_arm_overall_role": "unknown",
        "target_container": "unknown",
        "visible_object_categories": [],
        "tools_or_devices": [],
        "products_or_target_items": [],
        "likely_manipulated_objects_order": [],
        "planned_segments": [],
        "coarse_timeline": timeline,
        "selection_selected_frame_notes": selected_notes,
        "annotation_warnings": [
            "This context is reused from keyframe selection and may be imperfect.",
            "Use it only as a weak temporal hint; rely on local visual evidence for object/contact claims.",
            "Do not infer grasping/holding from 2D overlap alone."
        ],
        "endpoint": "selection_reuse",
    }


# ============================================================
# Structured frame prompts: transition + current state
# ============================================================

GRIPPER_RELATIONS = {
    "empty",
    "hovering_over_object",
    "near_object",
    "touching_object",
    "holding_object",
    "releasing_object",
    "uncertain",
}

ACTION_PHASES = {
    "idle",
    "approach",
    "grasp",
    "lift",
    "transport",
    "release",
    "retreat",
    "transition",
    "uncertain",
}

CONTACT_EVIDENCE = {
    "visible_between_gripper",
    "moves_with_gripper",
    "visual_overlap_only",
    "occluded",
    "no_evidence",
    "uncertain",
}


def normalize_enum(x: Any, allowed: set, default: str) -> str:
    s = str(x or "").strip().lower().replace(" ", "_")
    if s in allowed:
        return s
    # simple aliases
    aliases = {
        "holding": "holding_object",
        "grasping": "holding_object",
        "releasing": "releasing_object",
        "hovering": "hovering_over_object",
        "near": "near_object",
        "touching": "touching_object",
        "none": "empty",
        "unknown": "uncertain",
    }
    return aliases.get(s, default)


def clean_frame_prompt_text(text: str) -> str:
    """Remove formulaic transition prefixes while keeping useful motion/state content."""
    s = str(text or "").strip()
    if not s:
        return s
    patterns = [
        r"^Compared\s+(?:to|with)\s+the\s+previous\s+(?:selected\s+)?keyframe,?\s*",
        r"^Compared\s+(?:to|with)\s+K\d{1,2},?\s*",
        r"^From\s+the\s+previous\s+(?:selected\s+)?keyframe,?\s*",
    ]
    for pat in patterns:
        s = re.sub(pat, "", s, flags=re.IGNORECASE)
    s = re.sub(r"\band\s+in\s+this\s+frame,\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    return s


def build_frame_prompt_content(
    keyframe_paths: List[Path],
    image_cache: Dict[str, str],
    sequence_context: Dict[str, Any],
    target_idx: int,
    radius: int,
) -> List[Dict[str, Any]]:
    n = len(keyframe_paths)
    left = max(0, target_idx - radius)
    right = min(n - 1, target_idx + radius)
    target = keyframe_paths[target_idx]

    if target_idx == 0:
        previous_desc = "This is the first selected keyframe. There is no previous selected keyframe."
    else:
        previous_desc = f"The previous selected keyframe is K{target_idx - 1:02d}, filename {keyframe_paths[target_idx - 1].name}."

    prompt = f"""
You are annotating one selected keyframe from a sparse robot manipulation sequence with two visible arms.

Target frame:
- keyframe index: K{target_idx:02d}
- filename: {target.name}

Previous selected keyframe:
{previous_desc}

Sequence-level context, which may be imperfect and should be used only as a soft guide:
{safe_json_dumps(sequence_context, max_chars=5000)}

Your task:
Describe the TARGET frame using BOTH:
1. transition_from_previous: how the scene/arms changed from the previous selected keyframe to the target frame;
2. current_state: what the left arm and right arm are doing or where they are in the target frame.

Important rules:
- The selected keyframe may be a static frame, a transition waypoint, or an ambiguous pose. Do not force every frame to contain a clear action.
- Use nearby frames only as temporal context. The final frame_prompt must describe the TARGET frame, not future frames.
- Treat left/right only as visual identifiers in the image. Do not infer activity, inactivity, object ownership, or task role from the side label itself.
- Annotate each visible arm independently from visual and temporal evidence. If a role or contact relation is unclear, mark it uncertain instead of filling in a default pattern.
- Do not say "grasping" or "holding" merely because a gripper visually overlaps with an object in the camera view.
- A gripper is holding an object only if the object is visibly between the gripper fingers OR the object moves together with the gripper across nearby frames.
- If a gripper is above/near/occluding an object but actual contact is unclear, use neutral wording such as "hovers over", "moves near", or "is positioned above".
- If evidence is insufficient, use "uncertain" in structured fields and keep the frame_prompt neutral.
- The frame_prompt should be one or two moderately detailed English sentences, usually around 35-70 words total. Preserve the concise style of the original workflow, but add enough concrete local detail to distinguish this keyframe from its neighbors. It should explicitly mention both the LEFT arm and the RIGHT arm when both are relevant.
- Avoid formulaic phrases such as "Compared to/with the previous keyframe" in frame_prompt.
- Express motion naturally when useful, e.g. "The right arm shifts toward the cucumber while the left arm stays open above the tray."
- Prefer a clean current-state sentence if the transition is minor or ambiguous. Keep transition details in transition_from_previous.

Return strict JSON only:
{{
  "index": {target_idx},
  "transition_from_previous": "How K{target_idx:02d} changed from K{max(0, target_idx - 1):02d}; for K00, say initial state.",
  "current_state": "Current state of both arms in K{target_idx:02d}.",
  "left_arm": "state/pose/action of left arm in the target frame",
  "right_arm": "state/pose/action of right arm in the target frame",
  "left_gripper_relation": "empty | hovering_over_object | near_object | touching_object | holding_object | releasing_object | uncertain",
  "left_held_object": "none | object category | unknown | uncertain",
  "left_action_phase": "idle | approach | grasp | lift | transport | release | retreat | transition | uncertain",
  "left_contact_evidence": "visible_between_gripper | moves_with_gripper | visual_overlap_only | occluded | no_evidence | uncertain",
  "right_gripper_relation": "empty | hovering_over_object | near_object | touching_object | holding_object | releasing_object | uncertain",
  "right_held_object": "none | object category | unknown | uncertain",
  "right_action_phase": "idle | approach | grasp | lift | transport | release | retreat | transition | uncertain",
  "right_contact_evidence": "visible_between_gripper | moves_with_gripper | visual_overlap_only | occluded | no_evidence | uncertain",
  "active_arms": ["left", "right", "none", "uncertain"],
  "confidence": 0.0,
  "frame_prompt": "One or two moderately detailed English sentences, usually around 35-70 words total, describing the target frame in natural language. Include the manipulated object, relevant arm roles, gripper/contact state, useful spatial relations, and the meaningful change from the previous selected frame when supported. Do not describe future actions and do not start with 'Compared to/with the previous keyframe'."
}}
""".strip()

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for i in range(left, right + 1):
        role = "TARGET FRAME" if i == target_idx else "CONTEXT ONLY"
        p = keyframe_paths[i]
        content.append({"type": "text", "text": f"Keyframe K{i:02d}, filename={p.name}. {role}."})
        if i == target_idx:
            content.append({"type": "image_url", "image_url": {"url": make_keyframe_labeled_data_url(p, f'K{i:02d} TARGET', p.name)}})
        else:
            content.append({"type": "image_url", "image_url": {"url": image_cache[p.name]}})
    return content


async def annotate_one_frame_structured(
    endpoint_pool: EndpointPool,
    sem: asyncio.Semaphore,
    model: str,
    keyframe_paths: List[Path],
    image_cache: Dict[str, str],
    sequence_context: Dict[str, Any],
    target_idx: int,
    radius: int,
    max_tokens: int,
    retries: int,
) -> Dict[str, Any]:
    content = build_frame_prompt_content(
        keyframe_paths=keyframe_paths,
        image_cache=image_cache,
        sequence_context=sequence_context,
        target_idx=target_idx,
        radius=radius,
    )
    obj = await call_vlm_json(endpoint_pool, sem, model, content, max_tokens, retries)
    if "_error" in obj:
        return {
            "index": target_idx,
            "transition_from_previous": "",
            "current_state": "",
            "left_arm": "",
            "right_arm": "",
            "left_gripper_relation": "uncertain",
            "left_held_object": "uncertain",
            "left_action_phase": "uncertain",
            "left_contact_evidence": "uncertain",
            "right_gripper_relation": "uncertain",
            "right_held_object": "uncertain",
            "right_action_phase": "uncertain",
            "right_contact_evidence": "uncertain",
            "active_arms": ["uncertain"],
            "confidence": 0.0,
            "frame_prompt": "",
            "endpoint": "fallback",
            "error": obj.get("_error"),
        }

    frame_prompt = clean_frame_prompt_text(obj.get("frame_prompt", ""))
    return {
        "index": target_idx,
        "transition_from_previous": str(obj.get("transition_from_previous", "")).strip(),
        "current_state": str(obj.get("current_state", "")).strip(),
        "left_arm": str(obj.get("left_arm", "")).strip(),
        "right_arm": str(obj.get("right_arm", "")).strip(),
        "left_gripper_relation": normalize_enum(obj.get("left_gripper_relation"), GRIPPER_RELATIONS, "uncertain"),
        "left_held_object": str(obj.get("left_held_object", "uncertain")).strip().lower() or "uncertain",
        "left_action_phase": normalize_enum(obj.get("left_action_phase"), ACTION_PHASES, "uncertain"),
        "left_contact_evidence": normalize_enum(obj.get("left_contact_evidence"), CONTACT_EVIDENCE, "uncertain"),
        "right_gripper_relation": normalize_enum(obj.get("right_gripper_relation"), GRIPPER_RELATIONS, "uncertain"),
        "right_held_object": str(obj.get("right_held_object", "uncertain")).strip().lower() or "uncertain",
        "right_action_phase": normalize_enum(obj.get("right_action_phase"), ACTION_PHASES, "uncertain"),
        "right_contact_evidence": normalize_enum(obj.get("right_contact_evidence"), CONTACT_EVIDENCE, "uncertain"),
        "active_arms": obj.get("active_arms", ["uncertain"]),
        "confidence": as_float(obj.get("confidence"), 0.0),
        "frame_prompt": frame_prompt,
        "endpoint": obj.get("_endpoint"),
        "latency_sec": obj.get("_latency_sec"),
    }


# ============================================================
# Chunked frame prompts for production-scale annotation
# ============================================================

def chunk_indices(n: int, chunk_size: int) -> List[List[int]]:
    """Write ranges. For 21 frames and chunk_size=7 -> [0-6], [7-13], [14-20]."""
    return [list(range(i, min(i + chunk_size, n))) for i in range(0, n, chunk_size)]


def chunk_context_indices(n: int, write_indices: List[int], overlap: int, max_images: int) -> List[int]:
    """Return context frames for an overlap chunk, capped by max_images.

    The VLM annotates write_indices, but also sees nearby frames before/after the write range.
    We prioritize all write frames, then boundary-nearest overlap frames. This avoids feeding
    more than 11 individual images to vLLM.
    """
    if max_images < len(write_indices):
        # Keep the full write range whenever possible; if user sets an impossible cap,
        # still return the write range because missing target images would be worse.
        return list(write_indices)

    first, last = write_indices[0], write_indices[-1]
    context = set(write_indices)
    # Add context frames in boundary-nearest order: previous, next, previous-2, next+2, ...
    ordered_extra: List[int] = []
    for d in range(1, overlap + 1):
        for idx in (first - d, last + d):
            if 0 <= idx < n and idx not in context and idx not in ordered_extra:
                ordered_extra.append(idx)
    for idx in ordered_extra:
        if len(context) >= max_images:
            break
        context.add(idx)
    return sorted(context)


def make_fallback_frame_result(index: int, prompt: str = "") -> Dict[str, Any]:
    return {
        "index": index,
        "transition_from_previous": "",
        "current_state": "",
        "left_arm": "",
        "right_arm": "",
        "left_gripper_relation": "uncertain",
        "left_held_object": "uncertain",
        "left_action_phase": "uncertain",
        "left_contact_evidence": "uncertain",
        "right_gripper_relation": "uncertain",
        "right_held_object": "uncertain",
        "right_action_phase": "uncertain",
        "right_contact_evidence": "uncertain",
        "active_arms": ["uncertain"],
        "confidence": 0.0,
        "frame_prompt": prompt,
        "endpoint": "fallback",
    }


def normalize_frame_result(item: Dict[str, Any], index: int, endpoint: str = "") -> Dict[str, Any]:
    frame_prompt = clean_frame_prompt_text(item.get("frame_prompt", ""))
    return {
        "index": index,
        "transition_from_previous": str(item.get("transition_from_previous", "")).strip(),
        "current_state": str(item.get("current_state", "")).strip(),
        "left_arm": str(item.get("left_arm", "")).strip(),
        "right_arm": str(item.get("right_arm", "")).strip(),
        "left_gripper_relation": normalize_enum(item.get("left_gripper_relation"), GRIPPER_RELATIONS, "uncertain"),
        "left_held_object": str(item.get("left_held_object", "uncertain")).strip().lower() or "uncertain",
        "left_action_phase": normalize_enum(item.get("left_action_phase"), ACTION_PHASES, "uncertain"),
        "left_contact_evidence": normalize_enum(item.get("left_contact_evidence"), CONTACT_EVIDENCE, "uncertain"),
        "right_gripper_relation": normalize_enum(item.get("right_gripper_relation"), GRIPPER_RELATIONS, "uncertain"),
        "right_held_object": str(item.get("right_held_object", "uncertain")).strip().lower() or "uncertain",
        "right_action_phase": normalize_enum(item.get("right_action_phase"), ACTION_PHASES, "uncertain"),
        "right_contact_evidence": normalize_enum(item.get("right_contact_evidence"), CONTACT_EVIDENCE, "uncertain"),
        "active_arms": item.get("active_arms", ["uncertain"]),
        "confidence": as_float(item.get("confidence"), 0.0),
        "frame_prompt": frame_prompt,
        "endpoint": endpoint,
    }


def build_overlap_chunk_annotation_content(
    keyframe_paths: List[Path],
    image_cache: Dict[str, str],
    sequence_context: Dict[str, Any],
    write_indices: List[int],
    overlap: int,
    max_images_per_request: int,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    n = len(keyframe_paths)
    first = write_indices[0]
    last = write_indices[-1]
    context_indices = chunk_context_indices(
        n=n,
        write_indices=write_indices,
        overlap=overlap,
        max_images=max_images_per_request,
    )
    valid_ids = [f"K{i:02d}" for i in write_indices]
    context_ids = [f"K{i:02d}" for i in context_indices]
    prev_info = "This write chunk starts at K00, the first selected keyframe." if first == 0 else f"The previous selected keyframe before this write chunk is K{first - 1:02d}."
    next_info = "This write chunk reaches the last selected keyframe K20." if last == n - 1 else f"The next selected keyframe after this write chunk is K{last + 1:02d}."

    prompt = f"""
You are annotating an OVERLAP CHUNK from a sparse 21-keyframe dual-arm robot manipulation sequence.

Write range: K{first:02d}-K{last:02d}
Frames to annotate exactly: {valid_ids}
Frames you will see as images: {context_ids}
{prev_info}
{next_info}

Global task plan from a previous retrospective pass:
{safe_json_dumps(sequence_context, max_chars=7000)}

Important idea:
Use the global task plan and later context to infer the intended target of early approach/hover frames. For example, if the arm later grasps a cucumber, earlier ambiguous approach frames in the same segment may be described as approaching the cucumber. But do NOT claim the target frame is grasping/holding unless that specific frame shows contact or the object moving with the gripper.

Task:
For EVERY WRITE frame only, output a structured state and one moderately detailed local frame_prompt. Context-only frames are for continuity and must not receive outputs.

Rules:
- Do not force every frame to contain a new action. Sparse keyframes may be static, transitional, or repeated waypoints.
- Treat left/right only as visual identifiers. Infer each arm independently from visual evidence and the human task prior.
- Do not say "grasping" or "holding" merely because of 2D overlap.
- Holding is valid only if the object is visibly between the gripper fingers OR moves together with the gripper across this context window.
- If the gripper only passes above, occludes, or approaches an object, use neutral wording: "hovers above", "moves near", "is positioned over", or "approaches".
- Keep object names and functional roles consistent with the human task prior, global task plan, and surrounding frames, unless the images clearly contradict them.
- Do not rewrite tool-use/scanning/cleaning/folding/ironing/packing/appliance-operation tasks as generic pick-and-place.
- For cooperative tasks, describe the relation between the arms, e.g. one arm opens a bag, holds a product, presents an item, supports a cloth, or uses a tool while the other arm acts.
- At chunk boundaries, make the first/last write prompt continuous with the previous/next context frame.
- Each frame_prompt should be one or two natural English sentences, usually around 35-70 words total.
- Keep the original concise, high-quality style, but add concrete local details that help distinguish this frame from adjacent keyframes.
- Include the manipulated object, the relevant role/state of each arm, gripper/contact state, and useful spatial relations such as above, inside, beside, near the opening, over the tray, or moving toward a target when visually supported.
- Mention the meaningful change from the previous selected keyframe when useful, but do not describe future actions that are not yet visible.
- Mention both arms when both are relevant; do not mechanically repeat an unchanged arm if a task-centric sentence is clearer.
- Do not pad the prompt with generic scene description or repeat the full global task. Every added phrase should be locally useful.
- Prefer the pattern: visible change from previous selected frame + current state. If the change is minimal or unclear, write a clean current-state sentence.
- Avoid formulaic starts like "Compared to the previous keyframe".

Return strict JSON only:
{{
  "chunk_summary": "one sentence describing the write range",
  "frames": [
    {{
      "index": {first},
      "transition_from_previous": "short description of how this frame changes from the previous selected frame, or initial state for K00",
      "current_state": "current state of both arms in this frame",
      "left_arm": "state/pose/action of the arm visually on the left",
      "right_arm": "state/pose/action of the arm visually on the right",
      "left_gripper_relation": "empty | hovering_over_object | near_object | touching_object | holding_object | releasing_object | uncertain",
      "left_held_object": "none | object category | unknown | uncertain",
      "left_action_phase": "idle | approach | grasp | lift | transport | release | retreat | transition | uncertain",
      "left_contact_evidence": "visible_between_gripper | moves_with_gripper | visual_overlap_only | occluded | no_evidence | uncertain",
      "right_gripper_relation": "empty | hovering_over_object | near_object | touching_object | holding_object | releasing_object | uncertain",
      "right_held_object": "none | object category | unknown | uncertain",
      "right_action_phase": "idle | approach | grasp | lift | transport | release | retreat | transition | uncertain",
      "right_contact_evidence": "visible_between_gripper | moves_with_gripper | visual_overlap_only | occluded | no_evidence | uncertain",
      "active_arms": ["left", "right", "none", "uncertain"],
      "confidence": 0.0,
      "frame_prompt": "One or two moderately detailed English sentences, usually around 35-70 words total, describing the visible local state and the meaningful transition from the previous selected keyframe."
    }}
  ]
}}
""".strip()

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for i in context_indices:
        p = keyframe_paths[i]
        role = "WRITE_THIS_FRAME" if i in write_indices else "CONTEXT_ONLY"
        content.append({"type": "text", "text": f"Keyframe K{i:02d}, filename={p.name}. {role}."})
        if i in write_indices:
            content.append({"type": "image_url", "image_url": {"url": make_keyframe_labeled_data_url(p, f'K{i:02d} WRITE', p.name)}})
        else:
            content.append({"type": "image_url", "image_url": {"url": image_cache[p.name]}})
    return content, context_indices


async def annotate_one_overlap_chunk_structured(
    endpoint_pool: EndpointPool,
    sem: asyncio.Semaphore,
    model: str,
    keyframe_paths: List[Path],
    image_cache: Dict[str, str],
    sequence_context: Dict[str, Any],
    write_indices: List[int],
    overlap: int,
    max_images_per_request: int,
    max_tokens: int,
    retries: int,
) -> Dict[str, Any]:
    content, context_indices = build_overlap_chunk_annotation_content(
        keyframe_paths=keyframe_paths,
        image_cache=image_cache,
        sequence_context=sequence_context,
        write_indices=write_indices,
        overlap=overlap,
        max_images_per_request=max_images_per_request,
    )
    obj = await call_vlm_json(endpoint_pool, sem, model, content, max_tokens, retries)
    if "_error" in obj:
        return {
            "write_indices": write_indices,
            "context_indices": context_indices,
            "frames": [make_fallback_frame_result(i) for i in write_indices],
            "endpoint": "fallback",
            "error": obj.get("_error"),
        }

    raw_frames = obj.get("frames", [])
    if not isinstance(raw_frames, list):
        raw_frames = []

    by_index: Dict[int, Dict[str, Any]] = {}
    for item in raw_frames:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("index"))
        except Exception:
            continue
        if idx in write_indices:
            by_index[idx] = normalize_frame_result(item, idx, endpoint=obj.get("_endpoint", ""))

    frames = []
    for idx in write_indices:
        frames.append(by_index.get(idx, make_fallback_frame_result(idx)))

    return {
        "write_indices": write_indices,
        "context_indices": context_indices,
        "chunk_summary": str(obj.get("chunk_summary", "")).strip(),
        "frames": frames,
        "endpoint": obj.get("_endpoint"),
        "latency_sec": obj.get("_latency_sec"),
    }


async def annotate_frames_overlap_chunked(
    endpoint_pool: EndpointPool,
    sem: asyncio.Semaphore,
    model: str,
    keyframe_paths: List[Path],
    image_cache: Dict[str, str],
    sequence_context: Dict[str, Any],
    chunk_size: int,
    chunk_overlap: int,
    max_images_per_request: int,
    max_tokens: int,
    retries: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    write_chunks = chunk_indices(len(keyframe_paths), chunk_size)
    tasks = [
        annotate_one_overlap_chunk_structured(
            endpoint_pool=endpoint_pool,
            sem=sem,
            model=model,
            keyframe_paths=keyframe_paths,
            image_cache=image_cache,
            sequence_context=sequence_context,
            write_indices=idxs,
            overlap=chunk_overlap,
            max_images_per_request=max_images_per_request,
            max_tokens=max_tokens,
            retries=retries,
        )
        for idxs in write_chunks
    ]
    chunk_results = await asyncio.gather(*tasks)
    frame_results: List[Dict[str, Any]] = []
    for cr in chunk_results:
        frame_results.extend(cr.get("frames", []))
    frame_results.sort(key=lambda x: x["index"])
    return frame_results, chunk_results


# Backward-compatible alias: v8-style chunking now uses v9 overlap chunking by default.
async def annotate_frames_chunked(
    endpoint_pool: EndpointPool,
    sem: asyncio.Semaphore,
    model: str,
    keyframe_paths: List[Path],
    image_cache: Dict[str, str],
    contact_sheet_url: str,
    sequence_context: Dict[str, Any],
    chunk_size: int,
    max_tokens: int,
    retries: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    return await annotate_frames_overlap_chunked(
        endpoint_pool=endpoint_pool,
        sem=sem,
        model=model,
        keyframe_paths=keyframe_paths,
        image_cache=image_cache,
        sequence_context=sequence_context,
        chunk_size=chunk_size,
        chunk_overlap=2,
        max_images_per_request=11,
        max_tokens=max_tokens,
        retries=retries,
    )


# ============================================================
# Lightweight QC for production filtering
# ============================================================

OBJECT_ALIASES = {
    "cucumber": ["cucumber", "green cucumber"],
    "tomato": ["tomato", "red tomato"],
    "corn": ["corn", "corn cob", "yellow corn"],
    "pepper": ["pepper", "bell pepper", "red pepper", "yellow pepper"],
    "chili": ["chili", "chili pepper", "red chili"],
    "potato": ["potato"],
    "mushroom": ["mushroom"],
    "onion": ["onion"],
    "garlic": ["garlic"],
    "eggplant": ["eggplant"],
    "pumpkin": ["pumpkin"],
}

ACTION_WORDS = ["grasp", "grasps", "grasping", "hold", "holds", "holding", "lift", "lifts", "lifting", "transport", "transports", "carry", "carries", "lower", "lowers", "release", "releases", "place", "places", "placing"]


def canonical_object_name(text: str) -> str:
    t = str(text or "").lower()
    for canonical, aliases in OBJECT_ALIASES.items():
        for a in aliases:
            if a in t:
                return canonical
    if t in {"none", "unknown", "uncertain", ""}:
        return t or "unknown"
    return t


def objects_in_text(text: str) -> List[str]:
    t = str(text or "").lower()
    found = []
    for canonical, aliases in OBJECT_ALIASES.items():
        if any(a in t for a in aliases):
            found.append(canonical)
    return found


def prompt_claims_action_on_object(prompt: str, obj: str) -> bool:
    t = str(prompt or "").lower()
    if obj not in objects_in_text(t):
        return False
    return any(w in t for w in ACTION_WORDS)


def qc_annotation(frame_results: List[Dict[str, Any]], frame_prompts: List[str], global_prompt: str = "") -> Dict[str, Any]:
    issues = []

    supported = set()
    timeline = []
    for fr in frame_results:
        idx = int(fr.get("index", -1))
        for side in ["left", "right"]:
            obj = canonical_object_name(fr.get(f"{side}_held_object", ""))
            phase = str(fr.get(f"{side}_action_phase", "")).lower()
            rel = str(fr.get(f"{side}_gripper_relation", "")).lower()
            ev = str(fr.get(f"{side}_contact_evidence", "")).lower()
            if obj not in {"", "none", "unknown", "uncertain"} and (phase in {"grasp", "lift", "transport", "release"} or rel in {"holding_object", "releasing_object"}):
                supported.add(obj)
                timeline.append((idx, side, obj, phase, rel, ev))
            if rel == "holding_object" and ev in {"visual_overlap_only", "no_evidence", "uncertain"}:
                issues.append({
                    "severity": "warning",
                    "type": "weak_holding_evidence",
                    "frame": idx,
                    "side": side,
                    "description": f"{side} arm is marked holding_object with weak evidence={ev}."
                })

    # Prompt-action claims not supported by structured held_object timeline.
    for i, p in enumerate(frame_prompts):
        for obj in objects_in_text(p):
            if prompt_claims_action_on_object(p, obj) and obj not in supported:
                # Red/yellow pepper/chili is especially prone to false positives.
                severity = "hard" if obj in {"pepper", "chili"} else "warning"
                issues.append({
                    "severity": severity,
                    "type": "prompt_object_not_structurally_supported",
                    "frame": i,
                    "object": obj,
                    "description": f"Frame prompt claims action involving {obj}, but structured held-object timeline does not support it."
                })

    # Last-frame distraction check.
    if frame_prompts:
        last = frame_prompts[-1].lower()
        last_objects = objects_in_text(last)
        for obj in last_objects:
            if obj not in supported and any(x in last for x in ["near", "toward", "above", "over"]):
                issues.append({
                    "severity": "warning",
                    "type": "last_frame_unmanipulated_object_distraction",
                    "frame": len(frame_prompts) - 1,
                    "object": obj,
                    "description": "Final frame mentions an unmanipulated nearby object; this may distract generation."
                })

    # Global prompt should not describe unsupported manipulated objects.
    for obj in objects_in_text(global_prompt):
        if obj not in supported and prompt_claims_action_on_object(global_prompt, obj):
            severity = "hard" if obj in {"pepper", "chili"} else "warning"
            issues.append({
                "severity": severity,
                "type": "global_object_not_supported_by_timeline",
                "object": obj,
                "description": f"Global prompt appears to describe manipulation of {obj}, but frame timeline does not support it."
            })

    hard_count = sum(1 for x in issues if x.get("severity") == "hard")
    warning_count = sum(1 for x in issues if x.get("severity") == "warning")
    return {
        "status": "fail" if hard_count > 0 else "pass",
        "hard_count": hard_count,
        "warning_count": warning_count,
        "supported_objects": sorted(supported),
        "timeline": timeline,
        "issues": issues,
    }


# ============================================================
# Verifier / repair pass, not a classifier
# ============================================================

def build_verifier_content(
    keyframe_paths: List[Path],
    contact_sheet_url: str,
    sequence_context: Dict[str, Any],
    frame_results: List[Dict[str, Any]],
    task_hint: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rough_prompts = [x.get("frame_prompt", "") for x in frame_results]
    prompt = f"""
You are verifying frame-level prompts for a sparse 21-keyframe dual-arm robot manipulation sequence.

Goal:
Fix only clear temporal/object/contact inconsistencies while preserving the moderately detailed local information in each prompt. Do not shorten a correct prompt merely for conciseness.

You will receive:
1. sequence-level context;
2. 21 structured frame annotations;
3. 21 moderately detailed frame prompts;
4. a contact sheet of all keyframes K00-K20.

Check for these problems:
- a manipulated object suddenly changes without visual/temporal evidence;
- the task is rewritten into the wrong type, e.g. scanning/tool-use becomes generic pick-and-place;
- tools/devices are mistaken for products/items, or containers/appliance doors are mistaken for objects to be placed;
- either arm says holding/grasping based only on 2D overlap or occlusion;
- a placed object is inconsistent with the object transported in previous frames;
- impossible temporal reversals, e.g. an item is placed into the bag and then described as being lifted out, unless clearly visible;
- hallucinated objects that are not supported by the sequence;
- frame prompts that ignore an important cooperative role, e.g. bag opening, item presenting, scanning, wiping, folding, ironing, pouring, or door opening/closing;
- annotations that assign active/static roles based on arm side rather than visual or temporal evidence.

Repair policy:
- Revise only clearly problematic prompts.
- If a contact relation is ambiguous, use neutral wording: "moves near", "hovers above", "is positioned over", "moves toward".
- Keep every revised prompt as one or two natural English sentences, usually around 35-70 words total.
- Every prompt must preserve useful local details about the manipulated object, relevant arm roles, gripper/contact state, and spatial relations, using neutral wording when evidence is ambiguous.
- It is good if a prompt says both how the frame changed from the previous selected frame and what the current state is.

Human-provided task prior:
{format_task_hint_for_prompt(task_hint, max_chars=2200)}

Sequence context:
{safe_json_dumps(sequence_context, max_chars=5000)}

Structured frame annotations:
{safe_json_dumps(frame_results, max_chars=14000)}

Original frame prompts:
{chr(10).join([f"K{i:02d}. {p}" for i, p in enumerate(rough_prompts)])}

Return strict JSON only:
{{
  "issues": [
    {{"frames": [0, 1], "type": "short issue type", "description": "what was wrong and how it was repaired"}}
  ],
  "revised_frame_prompts": [
    "K00 prompt", "K01 prompt", "... exactly 21 strings total"
  ]
}}
""".strip()
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.append({"type": "text", "text": "Contact sheet of all selected keyframes K00-K20:"})
    content.append({"type": "image_url", "image_url": {"url": contact_sheet_url}})

    # Add a few high-res frames to help repair beginning/middle/end details.
    for idx in uniform_indices(len(keyframe_paths), 7):
        p = keyframe_paths[idx]
        content.append({"type": "text", "text": f"Individual keyframe K{idx:02d}, filename={p.name}."})
        content.append({"type": "image_url", "image_url": {"url": make_keyframe_labeled_data_url(p, f'K{idx:02d}', p.name)}})
    return content


async def verify_and_revise_frame_prompts(
    endpoint_pool: EndpointPool,
    sem: asyncio.Semaphore,
    model: str,
    keyframe_paths: List[Path],
    contact_sheet_url: str,
    sequence_context: Dict[str, Any],
    frame_results: List[Dict[str, Any]],
    max_tokens: int,
    retries: int,
    task_hint: Dict[str, Any],
) -> Dict[str, Any]:
    original = [str(x.get("frame_prompt", "")).strip() for x in frame_results]
    content = build_verifier_content(keyframe_paths, contact_sheet_url, sequence_context, frame_results, task_hint)
    obj = await call_vlm_json(endpoint_pool, sem, model, content, max_tokens, retries)
    if "_error" in obj:
        return {
            "issues": [],
            "revised_frame_prompts": original,
            "endpoint": "fallback",
            "error": obj.get("_error"),
        }
    revised = obj.get("revised_frame_prompts", original)
    if not isinstance(revised, list):
        revised = original
    revised = [clean_frame_prompt_text(x) for x in revised]
    if len(revised) != len(original) or any(not x for x in revised):
        revised = original
    return {
        "issues": obj.get("issues", []),
        "revised_frame_prompts": revised,
        "endpoint": obj.get("_endpoint"),
        "latency_sec": obj.get("_latency_sec"),
    }


# ============================================================
# Global prompt
# ============================================================

async def build_global_prompt_from_verified(
    endpoint_pool: EndpointPool,
    sem: asyncio.Semaphore,
    model: str,
    keyframe_paths: List[Path],
    contact_sheet_url: str,
    sequence_context: Dict[str, Any],
    frame_results: List[Dict[str, Any]],
    frame_prompts: List[str],
    global_image_count: int,
    max_tokens: int,
    retries: int,
    task_hint: Dict[str, Any],
) -> Dict[str, Any]:
    chosen = uniform_indices(len(keyframe_paths), global_image_count)

    prompt = f"""
You are writing TWO global prompts for the same sparse 21-keyframe dual-arm robot manipulation sequence.

The long prompt should preserve the original high-quality workflow and level of detail.
The short prompt should summarize only the task-level information needed across all keyframes.

You will receive:
1. a sequence-level context;
2. verified frame prompts K00-K20;
3. structured frame annotations;
4. a contact sheet and several high-resolution keyframes.

Write:
1. long_prompt: a clear English global prompt for keyframe-conditioned image/video generation, around 180-220 words. Preserve completeness and consistency over aggressive shortening.
2. short_prompt: a concise global task prompt around 35-60 words, containing only the task goal, principal manipulated objects, target container/tool/appliance, and the overall division of labor between the two arms when important. Do not narrate every stage.

The global prompt should include only:
- camera/viewpoint and workspace;
- the high-level task type and task goal;
- the functional role of each arm when relevant, e.g. item holder, scanner/tool user, bag opener, door opener, cloth manipulator, or support arm;
- the target container, appliance, tool, or workspace region if visible;
- the supported task sequence. For tool-use/scanning/cleaning/folding/ironing/packing/appliance-operation tasks, describe the real functional action instead of rewriting it as generic pick-and-place;
- consistency constraints such as fixed camera, lighting, layout, and unmanipulated objects.

Important:
- Do NOT write a verification report.
- Do NOT mention confidence, motion evidence, "visible between gripper fingers", frame IDs, K-numbers, debug details, or uncertainty notes.
- Do NOT list every visible background object unless useful; summarize background objects when possible.
- Do NOT introduce manipulated objects not supported by the verified frame prompts or images.
- Do NOT force all tasks into grasp-lift-place. Use the human task prior and verified frame prompts to preserve the true task meaning.
- Do NOT overclaim grasping/holding if the verified prompts only support hovering or positioning.
- Keep the style natural, compact, and suitable as a generation prompt.

Human-provided task prior:
{format_task_hint_for_prompt(task_hint, max_chars=2200)}

Sequence context:
{safe_json_dumps(sequence_context, max_chars=4500)}

Verified frame prompts:
{chr(10).join([f"K{i:02d}. {p}" for i, p in enumerate(frame_prompts)])}

Structured frame annotations:
{safe_json_dumps(frame_results, max_chars=9000)}

Return strict JSON only:
{{
  "long_prompt": "A clear global prompt of around 180-220 words.",
  "short_prompt": "A concise global task prompt of around 35-60 words."
}}
""".strip()

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.append({"type": "text", "text": "Contact sheet of all selected keyframes K00-K20:"})
    content.append({"type": "image_url", "image_url": {"url": contact_sheet_url}})
    for idx in chosen:
        p = keyframe_paths[idx]
        content.append({"type": "text", "text": f"Uniform sample keyframe K{idx:02d}, filename={p.name}."})
        content.append({"type": "image_url", "image_url": {"url": make_keyframe_labeled_data_url(p, f'K{idx:02d}', p.name)}})

    obj = await call_vlm_json(endpoint_pool, sem, model, content, max_tokens, retries)
    if "_error" in obj:
        return {
            "prompt": "",
            "long_prompt": "",
            "short_prompt": "",
            "global_image_indices": chosen,
            "endpoint": "fallback",
            "error": obj.get("_error"),
        }
    long_prompt = str(obj.get("long_prompt", obj.get("prompt", ""))).strip()
    short_prompt = str(obj.get("short_prompt", "")).strip()
    return {
        # Backward compatibility: prompt remains the long global prompt.
        "prompt": long_prompt,
        "long_prompt": long_prompt,
        "short_prompt": short_prompt,
        "global_image_indices": chosen,
        "endpoint": obj.get("_endpoint"),
        "latency_sec": obj.get("_latency_sec"),
    }


# ============================================================
# Main per-sample pipeline
# ============================================================

async def process_one_sample(
    sample_index_1based: int,
    sample_dir: Path,
    output_data_root: Path,
    endpoint_pool: EndpointPool,
    sample_sem: asyncio.Semaphore,
    selection_sem: asyncio.Semaphore,
    caption_sem: asyncio.Semaphore,
    summary_sem: asyncio.Semaphore,
    args,
) -> None:
    async with sample_sem:
        sample_id = f"sample_{sample_index_1based:06d}"
        out_dir = output_data_root / sample_id
        ann_path = out_dir / "annotation.json"

        if args.skip_existing and ann_path.exists():
            print(f"[SKIP] sample={sample_id} source={sample_dir}")
            return

        tmp_dir = output_data_root / f".{sample_id}.tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        images_dir = tmp_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        images = list_numeric_images(sample_dir)
        if len(images) != 60:
            print(f"[SKIP] sample={sample_id} source={sample_dir} len={len(images)} != 60")
            shutil.rmtree(tmp_dir)
            return

        task_hint = get_task_hint_for_sample(getattr(args, "task_hints_data", {}), sample_dir)
        hint_type = task_hint.get("task_type", "") or "unknown"
        has_hint = bool(task_hint.get("task_hint"))
        print(f"\n[SAMPLE {sample_id}] source={sample_dir} frames=60 target=21 task_hint={has_hint} type={hint_type}")

        if args.selection_mode == "uniform":
            selected_indices, decisions = select_21_keyframes_uniform()
        else:
            selected_indices, decisions = await select_21_keyframes(
                endpoint_pool=endpoint_pool,
                sem=selection_sem,
                model=args.model,
                images=images,
                max_tokens=args.selection_max_tokens,
                retries=args.retries,
            )

        # Copy selected keyframes to images/000.jpg ... images/020.jpg
        keyframe_paths: List[Path] = []
        for i, src_idx in enumerate(selected_indices):
            dst = images_dir / f"{i:03d}.jpg"
            shutil.copy2(images[src_idx], dst)
            keyframe_paths.append(dst)

        # Original unlabeled image cache for local windows.
        image_cache = {p.name: image_file_to_data_url(p) for p in keyframe_paths}
        labels = [f"K{i:02d}" for i in range(len(keyframe_paths))]
        contact_sheet_url = make_contact_sheet_data_url(
            keyframe_paths,
            labels,
            cols=args.contact_sheet_cols,
            thumb_w=args.contact_sheet_thumb_w,
        )

        # Optional sequence context pass.
        # v5 can reuse VLM keyframe-selection notes as a cheap soft context,
        # avoiding the extra large sequence-context VLM call.
        if args.disable_sequence_context:
            sequence_context = make_empty_sequence_context(reason=f"disabled:{args.annotation_mode}", task_hint=task_hint)
        elif args.sequence_context_source == "selection" and args.selection_mode == "vlm":
            sequence_context = build_sequence_context_from_selection(decisions, task_hint=task_hint)
        elif args.sequence_context_source == "vlm" and args.annotation_mode in {"balanced", "full", "chunked", "chunked_full", "overlap_chunked", "overlap_chunked_full"}:
            sequence_context = await build_sequence_context(
                endpoint_pool=endpoint_pool,
                sem=summary_sem,
                model=args.model,
                keyframe_paths=keyframe_paths,
                contact_sheet_url=contact_sheet_url,
                global_image_count=args.context_images,
                max_tokens=args.context_max_tokens,
                retries=args.retries,
                task_hint=task_hint,
            )
        else:
            sequence_context = make_empty_sequence_context(reason=f"disabled:{args.annotation_mode}", task_hint=task_hint)

        sequence_context = merge_task_hint_into_context(sequence_context, task_hint)

        # Structured frame prompts.
        # Production default: v10 uses overlap chunks guided by human task prior + global plan.
        chunk_results: List[Dict[str, Any]] = []
        if args.annotation_mode in {"overlap_chunked", "overlap_chunked_full", "chunked", "chunked_full"}:
            frame_results, chunk_results = await annotate_frames_overlap_chunked(
                endpoint_pool=endpoint_pool,
                sem=caption_sem,
                model=args.model,
                keyframe_paths=keyframe_paths,
                image_cache=image_cache,
                sequence_context=sequence_context,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                max_images_per_request=args.max_images_per_request,
                max_tokens=args.chunk_max_tokens,
                retries=args.retries,
            )
        else:
            tasks = [
                annotate_one_frame_structured(
                    endpoint_pool=endpoint_pool,
                    sem=caption_sem,
                    model=args.model,
                    keyframe_paths=keyframe_paths,
                    image_cache=image_cache,
                    sequence_context=sequence_context,
                    target_idx=i,
                    radius=args.caption_radius,
                    max_tokens=args.caption_max_tokens,
                    retries=args.retries,
                )
                for i in range(len(keyframe_paths))
            ]
            frame_results = await asyncio.gather(*tasks)
            frame_results.sort(key=lambda x: x["index"])
        raw_frame_prompts = [str(x.get("frame_prompt", "") or "").strip() for x in frame_results]

        # Safety guard: never write an annotation.json with mostly empty local prompts.
        nonempty_raw_prompts = [p for p in raw_frame_prompts if p]
        if len(nonempty_raw_prompts) < max(5, len(raw_frame_prompts) // 2):
            debug_fail = {
                "source_dir": str(sample_dir),
                "selected_source_frame_indices": selected_indices,
                "task_hint": task_hint,
                "sequence_context": sequence_context,
                "raw_frame_prompts": raw_frame_prompts,
                "frame_results": frame_results,
                "chunk_results": chunk_results,
                "error": "too_many_empty_frame_prompts_after_chunk_annotation"
            }
            tmp_dir.mkdir(parents=True, exist_ok=True)
            (tmp_dir / "failed_debug.json").write_text(
                json.dumps(debug_fail, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            print(
                f"[FAIL SAMPLE {sample_id}] too many empty frame prompts: "
                f"{len(nonempty_raw_prompts)}/{len(raw_frame_prompts)} source={sample_dir}",
                flush=True,
            )
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return

        # Optional verification/repair pass.
        # In chunked production mode, verifier is disabled by default and should be used only for small quality runs.
        use_verifier = (args.annotation_mode in {"full", "chunked_full", "overlap_chunked_full"}) and (not args.disable_verifier)
        if not use_verifier:
            verifier_result = {
                "issues": [],
                "revised_frame_prompts": raw_frame_prompts,
                "endpoint": f"disabled:{args.annotation_mode}",
            }
            frame_prompts = raw_frame_prompts
        else:
            verifier_result = await verify_and_revise_frame_prompts(
                endpoint_pool=endpoint_pool,
                sem=summary_sem,
                model=args.model,
                keyframe_paths=keyframe_paths,
                contact_sheet_url=contact_sheet_url,
                sequence_context=sequence_context,
                frame_results=frame_results,
                max_tokens=args.verifier_max_tokens,
                retries=args.retries,
                task_hint=task_hint,
            )
            frame_prompts = verifier_result["revised_frame_prompts"]

        story_rewrite_result = {
            "enabled": False,
            "revised_frame_prompts": frame_prompts,
            "issues": [],
            "endpoint": "disabled",
        }
        if getattr(args, "story_rewrite", False):
            story_rewrite_result = await rewrite_frame_prompts_as_story(
                endpoint_pool=endpoint_pool,
                sem=summary_sem,
                model=args.model,
                sequence_context=sequence_context,
                frame_prompts=frame_prompts,
                frame_results=frame_results,
                max_tokens=args.story_max_tokens,
                retries=args.retries,
            )
            if isinstance(story_rewrite_result, dict):
                revised_story_prompts = story_rewrite_result.get("revised_frame_prompts", frame_prompts)
                if isinstance(revised_story_prompts, list) and len(revised_story_prompts) == len(frame_prompts) and all(str(x).strip() for x in revised_story_prompts):
                    frame_prompts = [clean_frame_prompt_text(x) for x in revised_story_prompts]

        # Global prompt.
        global_result = await build_global_prompt_from_verified(
            endpoint_pool=endpoint_pool,
            sem=summary_sem,
            model=args.model,
            keyframe_paths=keyframe_paths,
            contact_sheet_url=contact_sheet_url,
            sequence_context=sequence_context,
            frame_results=frame_results,
            frame_prompts=frame_prompts,
            global_image_count=args.global_images,
            max_tokens=args.global_max_tokens,
            retries=args.retries,
            task_hint=task_hint,
        )

        qc_result = qc_annotation(frame_results, frame_prompts, global_result.get("prompt", "")) if args.enable_qc else {
            "status": "disabled",
            "hard_count": 0,
            "warning_count": 0,
            "issues": [],
        }

        # If requested, run the expensive verifier only for samples that failed cheap QC, then rebuild global prompt.
        if args.repair_failed_only and qc_result.get("status") == "fail" and not use_verifier:
            verifier_result = await verify_and_revise_frame_prompts(
                endpoint_pool=endpoint_pool,
                sem=summary_sem,
                model=args.model,
                keyframe_paths=keyframe_paths,
                contact_sheet_url=contact_sheet_url,
                sequence_context=sequence_context,
                frame_results=frame_results,
                max_tokens=args.verifier_max_tokens,
                retries=args.retries,
                task_hint=task_hint,
            )
            frame_prompts = verifier_result["revised_frame_prompts"]
            global_result = await build_global_prompt_from_verified(
                endpoint_pool=endpoint_pool,
                sem=summary_sem,
                model=args.model,
                keyframe_paths=keyframe_paths,
                contact_sheet_url=contact_sheet_url,
                sequence_context=sequence_context,
                frame_results=frame_results,
                frame_prompts=frame_prompts,
                global_image_count=args.global_images,
                max_tokens=args.global_max_tokens,
                retries=args.retries,
                task_hint=task_hint,
            )
            qc_result = qc_annotation(frame_results, frame_prompts, global_result.get("prompt", "")) if args.enable_qc else qc_result

        keyframes = [f"images/{i:03d}.jpg" for i in range(len(keyframe_paths))]
        annotation = {
            "id": sample_id,
            "image": keyframes[0],
            "keyframes": keyframes,
            # Backward compatibility: prompt remains the original-style long global prompt.
            "prompt": global_result.get("long_prompt", global_result.get("prompt", "")),
            "global_prompt_long": global_result.get("long_prompt", global_result.get("prompt", "")),
            "global_prompt_short": global_result.get("short_prompt", ""),
            "frame_prompts": frame_prompts,
        }
        (tmp_dir / "annotation.json").write_text(
            json.dumps(annotation, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        if args.write_debug_json:
            debug = {
                "source_dir": str(sample_dir),
                "selected_source_frame_indices": selected_indices,
                "batch_selection_decisions": decisions,
                "task_hint": task_hint,
                "sequence_context": sequence_context,
                "raw_frame_prompts": raw_frame_prompts,
                "frame_results": frame_results,
                "chunk_results": chunk_results,
                "verifier_result": verifier_result,
                "story_rewrite_result": story_rewrite_result,
                "global_result": global_result,
                "qc_result": qc_result,
            }
            (tmp_dir / "debug.json").write_text(
                json.dumps(debug, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        if out_dir.exists():
            shutil.rmtree(out_dir)
        tmp_dir.rename(out_dir)

        issue_n = len(verifier_result.get("issues", [])) if isinstance(verifier_result, dict) else 0
        qc_status = qc_result.get("status", "unknown") if isinstance(qc_result, dict) else "unknown"
        qc_hard = qc_result.get("hard_count", 0) if isinstance(qc_result, dict) else 0
        qc_warn = qc_result.get("warning_count", 0) if isinstance(qc_result, dict) else 0
        print(f"[OK SAMPLE {sample_id}] selected=21 issues={issue_n} qc={qc_status}/hard{qc_hard}/warn{qc_warn} out={out_dir}")


async def main_async(args):
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_data_root = output_root / "data" / "keyframes"
    output_data_root.mkdir(parents=True, exist_ok=True)

    sample_dirs = find_sample_dirs(input_root, required_frames=60)
    sample_dirs = select_sample_dirs_for_run(
        sample_dirs=sample_dirs,
        strategy=args.sample_strategy,
        max_samples=args.max_samples,
        start_index=args.start_index,
        seed=args.sample_seed,
    )

    args.task_hints_data = load_task_hints(args.task_hints)

    endpoints = [x.strip() for x in args.endpoints.split(",") if x.strip()]
    endpoint_pool = EndpointPool(endpoints)

    print(f"input_root={input_root}")
    print(f"output_root={output_root}")
    print(f"output_data_root={output_data_root}")
    print(f"num_samples={len(sample_dirs)}")
    print(f"task_hints={args.task_hints}")
    print(f"sample_strategy={args.sample_strategy} sample_seed={args.sample_seed}")
    print("selection_quota_per_batch=[6,5,5,5]")
    print("batch0 first frame is hard-required")
    print(f"sample_concurrency={args.sample_concurrency}")
    print(f"selection_concurrency={args.selection_concurrency}")
    print(f"caption_concurrency={args.caption_concurrency}")
    print(f"summary_concurrency={args.summary_concurrency}")
    print(f"selection_mode={args.selection_mode}")
    print(f"annotation_mode={args.annotation_mode}")
    print(f"sequence_context_source={args.sequence_context_source}")
    print(f"caption_radius={args.caption_radius}")
    print(f"disable_sequence_context={args.disable_sequence_context}")
    print(f"disable_verifier={args.disable_verifier}")
    print(f"chunk_size={getattr(args, 'chunk_size', None)}")
    print(f"chunk_overlap={getattr(args, 'chunk_overlap', None)}")
    print(f"max_images_per_request={getattr(args, 'max_images_per_request', None)}")
    print(f"enable_qc={getattr(args, 'enable_qc', None)}")
    print(f"repair_failed_only={getattr(args, 'repair_failed_only', None)}")

    sample_sem = asyncio.Semaphore(args.sample_concurrency)
    selection_sem = asyncio.Semaphore(args.selection_concurrency)
    caption_sem = asyncio.Semaphore(args.caption_concurrency)
    summary_sem = asyncio.Semaphore(args.summary_concurrency)

    tasks = []
    for i, d in enumerate(sample_dirs, start=1):
        tasks.append(
            process_one_sample(
                sample_index_1based=i,
                sample_dir=d,
                output_data_root=output_data_root,
                endpoint_pool=endpoint_pool,
                sample_sem=sample_sem,
                selection_sem=selection_sem,
                caption_sem=caption_sem,
                summary_sem=summary_sem,
                args=args,
            )
        )

    done = 0
    for coro in asyncio.as_completed(tasks):
        try:
            await coro
            done += 1
        except Exception as e:
            print(f"[ERROR] {repr(e)}")

    print("=" * 100)
    print(f"completed_tasks={done}/{len(tasks)}")
    print(f"final_data_root={output_data_root}")


def main():
    parser = argparse.ArgumentParser(
        description="Build 21-keyframe annotations. v10 uses human task hints + global planning + overlap chunk annotation + cheap QC + optional repair."
    )

    parser.add_argument("--input-root", default="/cache/data/head_frames_60_mse0015_trimmed/agibot_v2")
    parser.add_argument("--output-root", default="/cache/annotated")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoints", default=DEFAULT_ENDPOINTS)
    parser.add_argument("--task-hints", default="/cache/data/head_frames_60_mse0015_trimmed/observations/task_hints.json",
                        help="JSON file with human-provided observation-level task hints.")

    parser.add_argument("--selection-mode", choices=["vlm", "uniform"], default="uniform",
                        help="vlm keeps the original VLM-based 60-to-21 selection; uniform skips selection VLM calls for speed.")
    parser.add_argument("--annotation-mode", choices=["overlap_chunked", "overlap_chunked_full", "chunked", "chunked_full", "fast", "balanced", "full"], default="overlap_chunked",
                        help="overlap_chunked: global plan + overlapped chunks; overlap_chunked_full adds verifier; fast/balanced/full keep per-frame modes.")
    parser.add_argument("--sequence-context-source", choices=["none", "selection", "vlm"], default="vlm",
                        help="selection reuses keyframe-selection output as weak context; vlm runs one sequence-context call; none disables it.")

    parser.add_argument("--sample-concurrency", type=int, default=8)
    parser.add_argument("--selection-concurrency", type=int, default=8)
    parser.add_argument("--caption-concurrency", type=int, default=64)
    parser.add_argument("--summary-concurrency", type=int, default=4)

    parser.add_argument("--selection-max-tokens", type=int, default=500)
    parser.add_argument("--caption-radius", type=int, default=2)
    parser.add_argument("--caption-max-tokens", type=int, default=360)
    parser.add_argument("--chunk-size", type=int, default=7, help="Number of WRITE frames per chunk. 7 gives 3 chunks for 21 keyframes.")
    parser.add_argument("--chunk-overlap", type=int, default=2, help="Number of context frames to include before/after each write chunk.")
    parser.add_argument("--max-images-per-request", type=int, default=11, help="Maximum number of individual frame images in each chunk request. v9 default keeps this <=11.")
    parser.add_argument("--chunk-max-tokens", type=int, default=1400)
    parser.add_argument("--context-images", type=int, default=7, help="Individual high-res images for global planning; plus one contact sheet, so default is 11 images total.")
    parser.add_argument("--context-max-tokens", type=int, default=750)
    parser.add_argument("--global-images", type=int, default=5)
    parser.add_argument("--global-max-tokens", type=int, default=520)
    parser.add_argument("--verifier-max-tokens", type=int, default=1200)
    parser.add_argument("--disable-sequence-context", action="store_true")
    parser.add_argument("--disable-verifier", action="store_true")
    parser.add_argument("--enable-qc", action="store_true", default=True)
    parser.add_argument("--disable-qc", dest="enable_qc", action="store_false")
    parser.add_argument("--repair-failed-only", action="store_true", help="Run expensive verifier only when cheap QC fails, then rebuild the global prompt.")
    parser.add_argument("--contact-sheet-cols", type=int, default=7)
    parser.add_argument("--contact-sheet-thumb-w", type=int, default=240)
    parser.add_argument("--retries", type=int, default=2)

    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sample-strategy", choices=["sequential", "random", "stratified"], default="sequential",
                        help="How to choose samples after scanning: sequential, random, or stratified.")
    parser.add_argument("--sample-seed", type=int, default=42,
                        help="Random seed for random/stratified sample selection.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--story-rewrite", action="store_true", help="Run a lightweight text-only pass to make 21 frame prompts a coherent story.")
    parser.add_argument("--story-max-tokens", type=int, default=1200)
    parser.add_argument("--write-debug-json", action="store_true")

    args = parser.parse_args()
    asyncio.run(main_async(args))




# ============================================================
# FAST V10 OVERRIDES
# Goal:
# - use human task prior;
# - skip heavy global-plan pass when --disable-sequence-context is used;
# - make chunk JSON much shorter and more stable;
# - prevent writing annotation.json when local prompts are mostly empty.
# ============================================================

def extract_json(text: str) -> Dict[str, Any]:
    """More robust JSON extractor for VLM outputs."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    def try_load(x: str):
        x = re.sub(r",\s*([}\]])", r"\1", x)
        return json.loads(x)

    try:
        obj = try_load(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Find first balanced JSON object instead of greedy {.*}
    start = text.find("{")
    while start >= 0:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        cand = text[start:i+1]
                        try:
                            obj = try_load(cand)
                            if isinstance(obj, dict):
                                return obj
                        except Exception:
                            break
        start = text.find("{", start + 1)

    raise json.JSONDecodeError("Could not extract valid JSON object", text, 0)


def normalize_frame_result(item: Dict[str, Any], index: int, endpoint: str = "") -> Dict[str, Any]:
    """Fast schema: only require transition/current_state/frame_prompt, fill other fields conservatively."""
    transition = str(item.get("transition_from_previous", "") or "").strip()
    current_state = str(item.get("current_state", "") or "").strip()
    frame_prompt = clean_frame_prompt_text(item.get("frame_prompt", ""))

    if not frame_prompt:
        # fallback inside successful JSON, not global fallback
        frame_prompt = clean_frame_prompt_text(current_state or transition)

    return {
        "index": index,
        "transition_from_previous": transition,
        "current_state": current_state,
        "left_arm": str(item.get("left_arm", "") or "").strip(),
        "right_arm": str(item.get("right_arm", "") or "").strip(),
        "left_gripper_relation": "uncertain",
        "left_held_object": "uncertain",
        "left_action_phase": "uncertain",
        "left_contact_evidence": "uncertain",
        "right_gripper_relation": "uncertain",
        "right_held_object": "uncertain",
        "right_action_phase": "uncertain",
        "right_contact_evidence": "uncertain",
        "active_arms": item.get("active_arms", ["uncertain"]),
        "confidence": as_float(item.get("confidence"), 0.5),
        "frame_prompt": frame_prompt,
        "endpoint": endpoint,
    }


def build_overlap_chunk_annotation_content(
    keyframe_paths: List[Path],
    image_cache: Dict[str, str],
    sequence_context: Dict[str, Any],
    write_indices: List[int],
    overlap: int,
    max_images_per_request: int,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    n = len(keyframe_paths)
    first = write_indices[0]
    last = write_indices[-1]
    context_indices = chunk_context_indices(
        n=n,
        write_indices=write_indices,
        overlap=overlap,
        max_images=max_images_per_request,
    )

    valid_ids = [f"K{i:02d}" for i in write_indices]
    context_ids = [f"K{i:02d}" for i in context_indices]

    human_prior = sequence_context.get("human_task_prior", {}) if isinstance(sequence_context, dict) else {}
    task_type = human_prior.get("task_type") or sequence_context.get("task_type", "unknown")
    task_hint = human_prior.get("task_hint", "")

    prompt = f"""
You are annotating a short overlap chunk from a sparse 21-keyframe dual-arm robot manipulation sequence.

WRITE frames: K{first:02d}-K{last:02d}
Frames to output exactly: {valid_ids}
Frames shown as images: {context_ids}

Human-provided task prior:
task_type: {task_type}
task_hint: {task_hint}

Soft sequence context, if available:
{safe_json_dumps(sequence_context, max_chars=2500)}

Important:
- Use the human task prior to understand the overall task direction.
- This prior is reliable at task-type level, but frame-level state must still follow the image.
- Use later/nearby frames to infer what early approach frames are aiming for.
- Do NOT claim grasping/holding/releasing unless the target frame or nearby motion supports it.
- If a hand is using a tool, scanner, iron, brush, faucet, kettle, hanger, or packaging material, describe the functional relation instead of forcing pick-and-place.
- For cloth tasks, describe fabric shape change: fold, unfold, align, smooth, press.
- For scanning/tool-use tasks, describe product/tool relation.
- For packing tasks, distinguish product, box, cushioning, lid/flap.
- Mention both arms when both matter; do not mechanically list both if one is irrelevant, but avoid hiding a visible active arm.
- Each frame_prompt should be one natural English sentence.
- Do not start with "Compared to the previous keyframe".

Return strict JSON only. Use this compact schema:
{{
  "chunk_summary": "short summary of this write range",
  "frames": [
    {{
      "index": {first},
      "transition_from_previous": "short change from the previous selected keyframe, or initial state for K00",
      "current_state": "short current state of the visible arms and key objects",
      "frame_prompt": "one concise natural English sentence for this frame"
    }}
  ]
}}

Output exactly {len(write_indices)} frame objects, one for each write frame: {valid_ids}.
""".strip()

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for i in context_indices:
        p = keyframe_paths[i]
        role = "WRITE_THIS_FRAME" if i in write_indices else "CONTEXT_ONLY"
        content.append({"type": "text", "text": f"Keyframe K{i:02d}, filename={p.name}. {role}."})
        if i in write_indices:
            content.append({"type": "image_url", "image_url": {"url": make_keyframe_labeled_data_url(p, f"K{i:02d} WRITE", p.name)}})
        else:
            content.append({"type": "image_url", "image_url": {"url": image_cache[p.name]}})

    return content, context_indices


def qc_annotation(frame_results: List[Dict[str, Any]], frame_prompts: List[str], global_prompt: str = "") -> Dict[str, Any]:
    """Fast lightweight QC: mainly detect empty local prompts and obvious broken outputs."""
    issues = []
    nonempty = [p for p in frame_prompts if str(p or "").strip()]
    if len(nonempty) < max(5, len(frame_prompts) // 2):
        issues.append({
            "severity": "hard",
            "type": "too_many_empty_frame_prompts",
            "description": f"Only {len(nonempty)}/{len(frame_prompts)} frame prompts are non-empty."
        })

    too_short = [i for i, p in enumerate(frame_prompts) if 0 < len(str(p).strip()) < 15]
    if len(too_short) >= 5:
        issues.append({
            "severity": "warning",
            "type": "many_too_short_frame_prompts",
            "frames": too_short[:10],
            "description": "Many frame prompts are suspiciously short."
        })

    hard_count = sum(1 for x in issues if x.get("severity") == "hard")
    warning_count = sum(1 for x in issues if x.get("severity") == "warning")
    return {
        "status": "fail" if hard_count > 0 else "pass",
        "hard_count": hard_count,
        "warning_count": warning_count,
        "supported_objects": [],
        "timeline": [],
        "issues": issues,
    }




# ============================================================
# STORY COHERENCE PASS
# Lightweight text-only rewrite after chunk annotation.
# It fixes chunk-boundary restart/repetition while preserving visual facts from raw prompts.
# ============================================================



async def rewrite_frame_prompts_as_story(
    endpoint_pool: EndpointPool,
    sem: asyncio.Semaphore,
    model: str,
    sequence_context: Dict[str, Any],
    frame_prompts: List[str],
    frame_results: Optional[List[Dict[str, Any]]] = None,
    max_tokens: int = 1800,
    retries: int = 1,
) -> Dict[str, Any]:
    """Final task-aware event-story rewrite.

    Text-only pass. It first builds an event-level plan, then minimally edits
    K00-K20 so the local prompts form one coherent progressive story.
    It may weaken overstrong claims like premature "complete/final", but must
    not strengthen weak visual claims such as hover -> grasp.
    """
    original = [str(x or "").strip() for x in frame_prompts]
    if len(original) != 21 or sum(bool(x) for x in original) < 10:
        return {
            "enabled": True,
            "mode": "final_task_aware_event_story_rewrite",
            "event_plan": [],
            "revised_frame_prompts": original,
            "issues": [{"type": "skip", "description": "invalid or mostly empty original prompts"}],
            "endpoint": "skipped",
        }

    frame_results = frame_results or []
    frame_evidence_lines = []
    for i, p in enumerate(original):
        fr = frame_results[i] if i < len(frame_results) and isinstance(frame_results[i], dict) else {}
        transition = str(fr.get("transition_from_previous", "") or "").strip()
        current_state = str(fr.get("current_state", "") or "").strip()
        left_arm = str(fr.get("left_arm", "") or "").strip()
        right_arm = str(fr.get("right_arm", "") or "").strip()

        parts = [f"K{i:02d} raw_prompt: {p}"]
        if transition:
            parts.append(f"transition: {transition}")
        if current_state:
            parts.append(f"current_state: {current_state}")
        if left_arm:
            parts.append(f"left_arm: {left_arm}")
        if right_arm:
            parts.append(f"right_arm: {right_arm}")
        frame_evidence_lines.append(" | ".join(parts))

    human_prior = {}
    if isinstance(sequence_context, dict):
        human_prior = sequence_context.get("human_task_prior", {}) or {}

    task_type = human_prior.get("task_type", sequence_context.get("task_type", "unknown") if isinstance(sequence_context, dict) else "unknown")
    task_hint = human_prior.get("task_hint", "")

    prompt = f"""
You are editing 21 frame-level prompts for a sparse robot manipulation video.

Your job:
First infer an EVENT-LEVEL PLAN from the human task prior and raw frame prompts.
Then perform a TASK-AWARE MINIMAL CONTINUITY EDIT so K00-K20 read as one coherent progressive story.

This is NOT free rewriting:
- Do not re-detect images.
- Do not invent new objects or task stages.
- Do not strengthen visual claims.
- The raw prompt/current_state for each frame is the source of truth for that frame.
- You may WEAKEN overstrong or premature words, such as "complete", "final", "finished", "places", or "releases", if they create impossible repetition with later frames.
- You may turn a premature "completing the insertion" into "guiding toward the cup" if later frames clearly continue the same insertion.

Human-provided task prior:
task_type: {task_type}
task_hint:
{task_hint}

Per-frame evidence:
{chr(10).join(frame_evidence_lines)}

Step A: Build a compact event plan.
The event plan should divide K00-K20 into 3-6 continuous stages.
Examples:
- tea task: kettle grasp/pour/set down -> tea bag pickup -> tag/string guidance -> insertion -> retract
- shopping task: first product transfer -> second product transfer -> third product transfer -> next item approach
- cloth task: edge grasp -> fold inward -> align/smooth -> final release/retract
- packing task: bottom cushioning -> item insertion -> top cushioning -> close/settle
- scanning task: product presentation -> scanner alignment -> scan -> product moved away

Step B: Rewrite frame prompts.
Rules:
1. Return exactly 21 revised prompts, one for K00-K20.
2. Each revised prompt should remain one or two moderately detailed natural English sentences, usually around 35-70 words total.
3. Make the sequence progressive: each prompt should naturally follow the previous one while preserving useful local visual details.
4. Use the event plan to remove chunk-boundary restarts at K06->K07 and K13->K14.
5. Preserve the main object of each frame.
6. Preserve which arm performs the main action in each frame.
7. Do NOT strengthen contact claims:
   - approach/near/hover/positioned-over must not become grasp/hold/release.
   - Only keep grasp/hold/release if the raw prompt already claims it.
8. It is allowed to weaken a claim:
   - "completing insertion" -> "guiding toward insertion"
   - "final folded form" -> "continues smoothing the fold"
   - "places the item" -> "positions/lowers the item" if later frames still perform the placement.
9. Avoid "begins", "starts", "preparing", or "initiating" in later chunks if the same event is already underway.
10. Avoid "complete", "final", "finished", "neatly folded", or "task complete" before K18 unless the entire sequence is clearly done.
11. Do not add extra support actions such as "stabilizing the kettle" or "resting on the table" unless present in the raw prompt/current_state.
12. For tool-use/scanning tasks, keep the product-tool relation.
13. For tea/pouring tasks, keep kettle, cup, tea bag, yellow tag/string roles clear.
14. For cloth tasks, describe fabric deformation rather than generic object transport.
15. For packing tasks, distinguish product, box, air-column cushioning, flaps/lids.
16. If a frame is ambiguous, keep it conservative.
17. Preserve useful local details from the raw prompt; do not compress a correct detailed prompt back into a very short caption.
18. Do not repeat the full global task or add generic background details merely to increase length.

Return strict JSON only:
{{
  "event_plan": [
    {{
      "frame_range": [0, 7],
      "event": "short event name",
      "description": "what is happening across this range"
    }}
  ],
  "issues": [
    {{
      "frames": [13, 14],
      "type": "chunk_boundary_restart",
      "description": "what was minimally smoothed"
    }}
  ],
  "revised_frame_prompts": [
    "K00 revised prompt",
    "K01 revised prompt",
    "... exactly 21 strings total"
  ]
}}
""".strip()

    content = [{"type": "text", "text": prompt}]
    obj = await call_vlm_json(
        endpoint_pool,
        sem,
        model,
        content,
        max_tokens=max_tokens,
        retries=retries,
        temperature=0.0,
    )

    if "_error" in obj:
        return {
            "enabled": True,
            "mode": "final_task_aware_event_story_rewrite",
            "event_plan": [],
            "revised_frame_prompts": original,
            "issues": [{"type": "fallback", "description": obj.get("_error", "")}],
            "endpoint": "fallback",
            "error": obj.get("_error"),
        }

    revised = obj.get("revised_frame_prompts", original)
    if not isinstance(revised, list):
        revised = original
    revised = [clean_frame_prompt_text(x) for x in revised]

    # Guard against bad rewrite outputs.
    if len(revised) != 21 or any(not str(x).strip() for x in revised):
        revised = original

    return {
        "enabled": True,
        "mode": "final_task_aware_event_story_rewrite",
        "event_plan": obj.get("event_plan", []),
        "issues": obj.get("issues", []),
        "revised_frame_prompts": revised,
        "endpoint": obj.get("_endpoint"),
        "latency_sec": obj.get("_latency_sec"),
    }


if __name__ == "__main__":
    main()
