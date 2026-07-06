#!/usr/bin/env python3
import argparse
import asyncio
import base64
import csv
import io
import json
import logging
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
from PIL import Image, ImageDraw, ImageFont
from openai import AsyncOpenAI

DEFAULT_MODEL = "Qwen3-VL-32B-Instruct"
DEFAULT_ENDPOINTS = "http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1"
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
]
LOGGER = logging.getLogger("ego4d_full_pipeline")


def setup_logging(path: Path, console: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    LOGGER.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    LOGGER.addHandler(fh)
    if console:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        LOGGER.addHandler(sh)


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def load_font(size: int):
    for fp in FONT_CANDIDATES:
        if Path(fp).exists():
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()


def pil_to_data_url(img: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def resize_keep_aspect(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale >= 1.0:
        return img
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)


def make_labeled_url(path: Path, label: str, timestamp: float, max_side: int) -> str:
    img = resize_keep_aspect(Image.open(path).convert("RGB"), max_side)
    w, h = img.size
    bar = max(36, int(h * 0.10))
    canvas = Image.new("RGB", (w, h + bar), "white")
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, h, w, h + bar], fill=(245, 245, 245))
    draw.line([0, h, w, h], fill=(170, 170, 170), width=2)
    draw.text((8, h + 2), label, fill=(0, 0, 0), font=load_font(max(16, w // 22)))
    draw.text((max(100, w // 5), h + 5), f"{timestamp:.2f}s", fill=(70, 70, 70), font=load_font(max(12, w // 34)))
    return pil_to_data_url(canvas)


def extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        obj = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise ValueError("No JSON object found")
        obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("Top-level JSON must be an object")
    return obj


def recursively_find_video_objects(obj: Any):
    if isinstance(obj, dict):
        if isinstance(obj.get("video_uid"), str):
            yield obj
        for v in obj.values():
            yield from recursively_find_video_objects(v)
    elif isinstance(obj, list):
        for x in obj:
            yield from recursively_find_video_objects(x)


def build_annotation_index(path: str) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        LOGGER.warning("[ANNOTATION] missing file=%s", p)
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, Any]] = {}
    for item in recursively_find_video_objects(raw):
        uid = str(item.get("video_uid", "")).strip()
        if uid and uid not in out:
            out[uid] = item
    LOGGER.info("[ANNOTATION] indexed=%d file=%s", len(out), p)
    return out


def flatten_strings(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x.strip()] if x.strip() else []
    if isinstance(x, list):
        out: List[str] = []
        for y in x:
            out.extend(flatten_strings(y))
        return out
    return [str(x)]


def compact_annotation(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if raw is None:
        return {"goal_descriptions": [], "goal_categories": [], "summaries": [], "steps": []}
    goals: List[str] = []
    categories: List[str] = []
    summaries: List[str] = []
    steps: List[Dict[str, Any]] = []

    def visit(x: Any) -> None:
        if isinstance(x, dict):
            goals.extend(flatten_strings(x.get("goal_description")))
            categories.extend(flatten_strings(x.get("goal_category")))
            summaries.extend(flatten_strings(x.get("summary")))
            desc = x.get("step_description") or x.get("step_category") or x.get("narration_text")
            if desc:
                d = " | ".join(flatten_strings(desc))
                if d:
                    steps.append({"start_time": x.get("start_time"), "end_time": x.get("end_time"), "description": d})
            for v in x.values():
                visit(v)
        elif isinstance(x, list):
            for y in x:
                visit(y)
    visit(raw)

    def dedup(xs: Sequence[str]) -> List[str]:
        seen, out = set(), []
        for x in xs:
            s = re.sub(r"\s+", " ", str(x)).strip()
            if s and s.lower() not in seen:
                seen.add(s.lower())
                out.append(s)
        return out

    return {
        "goal_descriptions": dedup(goals),
        "goal_categories": dedup(categories),
        "summaries": dedup(summaries),
        "steps": steps,
    }


class AnnotationResolver:
    def __init__(self, train: str, val: str):
        self.train = build_annotation_index(train)
        self.val = build_annotation_index(val)

    def resolve(self, uid: str) -> Tuple[str, Optional[Dict[str, Any]], Dict[str, Any]]:
        if uid in self.train:
            raw = self.train[uid]
            return "goalstep_train", raw, compact_annotation(raw)
        if uid in self.val:
            raw = self.val[uid]
            return "goalstep_val", raw, compact_annotation(raw)
        return "visual_only", None, compact_annotation(None)


def overlapping_context(compact: Dict[str, Any], start: float, end: float) -> Dict[str, Any]:
    overlaps = []
    for s in compact.get("steps", []):
        try:
            a, b = float(s.get("start_time")), float(s.get("end_time"))
        except Exception:
            continue
        if b >= start and a <= end:
            overlaps.append(s)
    return {
        "goal_descriptions": compact.get("goal_descriptions", [])[:5],
        "goal_categories": compact.get("goal_categories", [])[:5],
        "window_steps": overlaps[:10],
    }


class EndpointPool:
    def __init__(self, endpoints: List[str], timeout: float):
        self.clients = [(ep, AsyncOpenAI(base_url=ep, api_key="EMPTY", timeout=timeout)) for ep in endpoints]
        if not self.clients:
            raise ValueError("No endpoints provided")
        self.idx = 0
        self.lock = asyncio.Lock()

    async def next(self):
        async with self.lock:
            item = self.clients[self.idx % len(self.clients)]
            self.idx += 1
            return item


async def call_vlm_json(pool: EndpointPool, sem: asyncio.Semaphore, model: str, content: List[Dict[str, Any]], max_tokens: int, retries: int, label: str) -> Dict[str, Any]:
    last_error, last_raw = "", ""
    for attempt in range(retries + 1):
        endpoint = "unassigned"
        try:
            async with sem:
                endpoint, client = await pool.next()
                LOGGER.info("[SEND] %s attempt=%d/%d endpoint=%s images=%d", label, attempt + 1, retries + 1, endpoint, sum(1 for x in content if x.get("type") == "image_url"))
                t0 = time.time()
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": content}],
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                latency = time.time() - t0
            raw = resp.choices[0].message.content or ""
            last_raw = raw
            obj = extract_json_object(raw)
            obj["_endpoint"] = endpoint
            obj["_latency_sec"] = round(latency, 3)
            LOGGER.info("[SUCCESS] %s endpoint=%s latency=%.2fs", label, endpoint, latency)
            return obj
        except Exception as exc:
            last_error = repr(exc)
            LOGGER.warning("[CALL_ERROR] %s endpoint=%s error=%s", label, endpoint, last_error)
            lower = last_error.lower()
            if "maximum context length" in lower or "input length" in lower:
                break
            if attempt < retries:
                await asyncio.sleep(0.8 * (attempt + 1) + random.random() * 0.3)
    return {"_error": last_error, "_raw": last_raw, "_endpoint": "failed"}


def video_info(path: Path) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": fps, "frame_count": count, "width": width, "height": height, "duration": count / fps if fps > 0 else 0.0}


def extract_1fps(video: Path, frame_dir: Path, meta_path: Path, interval: float, max_side: int, quality: int, reuse: bool) -> List[Dict[str, Any]]:
    if reuse and meta_path.exists():
        meta = read_json(meta_path, [])
        if isinstance(meta, list) and meta and all(Path(x.get("path", "")).exists() for x in meta):
            LOGGER.info("[EXTRACT_REUSE] video=%s frames=%d", video.name, len(meta))
            return meta
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    info = video_info(video)
    if info["duration"] <= 0:
        raise RuntimeError("Invalid duration")
    cap = cv2.VideoCapture(str(video))
    records = []
    t = 0.0
    idx = 0
    LOGGER.info("[EXTRACT_START] video=%s duration=%.2fs source=%dx%d", video.name, info["duration"], info["width"], info["height"])
    while t < info["duration"]:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            scale = min(1.0, max_side / max(w, h))
            if scale < 1.0:
                frame = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
            actual = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            out = frame_dir / f"{idx:06d}.jpg"
            if cv2.imwrite(str(out), frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]):
                h2, w2 = frame.shape[:2]
                records.append({"candidate_index": idx, "target_time": t, "timestamp": actual, "width": w2, "height": h2, "path": str(out)})
                idx += 1
        t += interval
    cap.release()
    atomic_json(meta_path, records)
    LOGGER.info("[EXTRACT_DONE] video=%s frames=%d", video.name, len(records))
    return records


def normalize_selection(obj: Dict[str, Any], n: int) -> Tuple[bool, List[Dict[str, Any]]]:
    usable = bool(obj.get("usable", True))
    raw = obj.get("selected_frames", [])
    if not isinstance(raw, list):
        raw = []
    by_id: Dict[str, Dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        m = re.search(r"F?\s*(\d+)", str(item.get("id", "")).upper())
        if not m:
            continue
        i = int(m.group(1))
        if not 0 <= i < n:
            continue
        fid = f"F{i:02d}"
        try:
            importance = max(1, min(5, int(item.get("importance", 3))))
        except Exception:
            importance = 3
        current = {
            "id": fid,
            "caption": str(item.get("caption", "")).strip(),
            "semantic_role": str(item.get("semantic_role", "")).strip(),
            "importance": importance,
            "reason": str(item.get("reason", "")).strip(),
        }
        if fid not in by_id or importance > by_id[fid]["importance"]:
            by_id[fid] = current
    selected = sorted(by_id.values(), key=lambda x: int(x["id"][1:]))
    if len(selected) > 4:
        selected = sorted(sorted(selected, key=lambda x: (-x["importance"], int(x["id"][1:])))[:4], key=lambda x: int(x["id"][1:]))
    if not usable:
        return False, []
    if not selected:
        mid = n // 2
        selected = [{"id": f"F{mid:02d}", "caption": "", "semantic_role": "scene", "importance": 1, "reason": "Fallback middle frame."}]
    return True, selected


def selection_content(records: Sequence[Dict[str, Any]], context: Dict[str, Any], chunk_idx: int, side: int) -> List[Dict[str, Any]]:
    ids = [f"F{i:02d}" for i in range(len(records))]
    prompt = f"""
You are selecting sparse visual states for a long-horizon Ego4D VIDEO GENERATION dataset.
This window contains {len(records)} consecutive frames sampled at about 1 fps.

Weak annotation prior:
{json.dumps(context, ensure_ascii=False, indent=2)}

Selection count:
- Default: exactly 3 frames.
- Select 2 only if the window is redundant or partially poor.
- Select 1 only if nearly the whole window is unchanged or poor.
- Select 4 only if four clearly distinct useful states exist.
- Select 0 only if the entire window is unusable: severe blur, black frames, complete occlusion, or no recognizable content.

Selection criteria:
1. Preserve a sparse but learnable visual trajectory.
2. Prefer clear views of hands, people, objects, tools, workspace, meaningful scene transitions, and intermediate task states.
3. Avoid near-duplicates and avoid clustering selections in one short region.
4. Spread selections across the full window when possible.
5. Usually avoid consecutive frame IDs unless they show important before/after change.
6. Annotation is weak; trust the images first.
7. Every selected frame must be useful as an independent image-generation target.

Valid IDs: {ids}

Return strict JSON only:
{{
  "usable": true,
  "window_summary": "Brief visual evolution summary.",
  "selected_frames": [
    {{
      "id": "F02",
      "caption": "Short visually grounded description.",
      "semantic_role": "scene | transition | action | interaction | state_change | result",
      "importance": 1,
      "reason": "Why it is useful and non-redundant."
    }}
  ],
  "discard_reason": ""
}}
""".strip()
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for i, r in enumerate(records):
        fid = f"F{i:02d}"
        content.append({"type": "text", "text": f"{fid}; timestamp={r['timestamp']:.2f}s; candidate_index={r['candidate_index']}"})
        content.append({"type": "image_url", "image_url": {"url": make_labeled_url(Path(r["path"]), fid, float(r["timestamp"]), side)}})
    return content


async def select_one_window(pool, sem, model, records, context, chunk_idx, side, max_tokens, retries, uid):
    # A 15-image request normally fits at 512px, but automatically downscale if a
    # backend/model configuration reports context overflow.
    candidate_sides = []
    for candidate in (side, 448, 384, 320):
        if candidate > 0 and candidate not in candidate_sides:
            candidate_sides.append(candidate)

    last_obj: Dict[str, Any] = {}
    for current_side in candidate_sides:
        content = selection_content(records, context, chunk_idx, current_side)
        label = (
            f"select video={uid} chunk={chunk_idx:04d} "
            f"time={records[0]['timestamp']:.1f}-{records[-1]['timestamp']:.1f}s "
            f"side={current_side}"
        )
        obj = await call_vlm_json(pool, sem, model, content, max_tokens, retries, label)
        last_obj = obj
        if "_error" not in obj:
            usable, selected = normalize_selection(obj, len(records))
            LOGGER.info(
                "[SELECT_RESULT] video=%s chunk=%04d usable=%s selected=%d ids=%s side=%d",
                uid,
                chunk_idx,
                usable,
                len(selected),
                ",".join(x["id"] for x in selected) if selected else "-",
                current_side,
            )
            return {
                "window_index": chunk_idx,
                "status": "ok",
                "usable": usable,
                "selected_frames": selected,
                "window_summary": str(obj.get("window_summary", "")).strip(),
                "discard_reason": str(obj.get("discard_reason", "")).strip(),
                "endpoint": obj.get("_endpoint"),
                "latency_sec": obj.get("_latency_sec"),
                "image_side_used": current_side,
            }

        error = str(obj.get("_error", "")).lower()
        is_context_error = "maximum context length" in error or "input length" in error
        if not is_context_error:
            break
        LOGGER.warning(
            "[SELECT_DOWNSCALE] video=%s chunk=%04d failed_side=%d",
            uid,
            chunk_idx,
            current_side,
        )

    return {
        "window_index": chunk_idx,
        "status": "failed",
        "usable": False,
        "selected_frames": [],
        "error": last_obj.get("_error", ""),
    }


async def select_keyframes(uid: str, frames: List[Dict[str, Any]], compact: Dict[str, Any], sample_dir: Path, pool, sem, args) -> Dict[str, Any]:
    path = sample_dir / "keyframe_selection.json"
    images_dir = sample_dir / "images"
    if args.resume and path.exists() and images_dir.exists() and not args.overwrite_selection:
        existing = read_json(path, {})
        if existing.get("num_selected_keyframes", 0) > 0:
            LOGGER.info("[SELECT_REUSE] video=%s selected=%d", uid, existing["num_selected_keyframes"])
            return existing
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    windows = []
    for start in range(0, len(frames), args.window_stride):
        end = min(start + args.window_size, len(frames))
        if end - start < args.min_window_frames:
            break
        recs = frames[start:end]
        windows.append({"window_index": len(windows), "start": start, "end": end, "records": recs, "context": overlapping_context(compact, float(recs[0]["timestamp"]), float(recs[-1]["timestamp"]))})
    tasks = [select_one_window(pool, sem, args.model, w["records"], w["context"], w["window_index"], args.selection_image_side, args.selection_max_tokens, args.retries, uid) for w in windows]
    results = await asyncio.gather(*tasks)
    results.sort(key=lambda x: x["window_index"])
    selected_by_candidate: Dict[int, Dict[str, Any]] = {}
    for w, res in zip(windows, results):
        res["start"] = w["start"]
        res["end"] = w["end"]
        res["annotation_context"] = w["context"]
        for item in res.get("selected_frames", []):
            local_idx = int(item["id"][1:])
            rec = w["records"][local_idx]
            cidx = int(rec["candidate_index"])
            merged = {"candidate_index": cidx, "timestamp": float(rec["timestamp"]), "target_time": float(rec["target_time"]), "source_path": rec["path"], "selection_caption": item.get("caption", ""), "semantic_role": item.get("semantic_role", ""), "importance": item.get("importance", 1), "selection_reason": item.get("reason", ""), "selected_by_windows": [w["window_index"]]}
            if cidx not in selected_by_candidate:
                selected_by_candidate[cidx] = merged
            else:
                selected_by_candidate[cidx]["selected_by_windows"].append(w["window_index"])
    selected = [selected_by_candidate[k] for k in sorted(selected_by_candidate)]
    for order, r in enumerate(selected):
        dst = images_dir / f"{order:03d}.jpg"
        shutil.copy2(r["source_path"], dst)
        r["selected_order"] = order
        r["output_path"] = str(dst)
        r["relative_path"] = f"images/{order:03d}.jpg"
    usable = sum(1 for x in results if x.get("status") == "ok" and x.get("usable"))
    failed = sum(1 for x in results if x.get("status") == "failed")
    counts = [len(x.get("selected_frames", [])) for x in results if x.get("status") == "ok" and x.get("usable")]
    summary = {"video_uid": uid, "num_candidate_frames": len(frames), "num_windows": len(windows), "num_usable_windows": usable, "num_failed_windows": failed, "mean_selected_per_usable_window": sum(counts) / len(counts) if counts else 0.0, "num_selected_keyframes": len(selected), "selected_keyframes": selected, "window_results": results}
    atomic_json(path, summary)
    return summary


def uniform_pick(items: Sequence[Any], k: int) -> List[Any]:
    if not items or k <= 0:
        return []
    if len(items) <= k:
        return list(items)
    if k == 1:
        return [items[len(items) // 2]]
    idxs = [round(i * (len(items) - 1) / (k - 1)) for i in range(k)]
    return [items[i] for i in idxs]


def global_content(selected: Sequence[Dict[str, Any]], source: str, compact: Dict[str, Any], side: int, rep_count: int) -> List[Dict[str, Any]]:
    reps = uniform_pick(selected, rep_count)
    ann = {"annotation_source": source, "goal_descriptions": compact.get("goal_descriptions", [])[:5], "goal_categories": compact.get("goal_categories", [])[:5], "summaries": compact.get("summaries", [])[:5], "step_descriptions": [x.get("description", "") for x in compact.get("steps", [])[:30]]}
    prompt = f"""
Write one GLOBAL PROMPT for a long-horizon Ego4D keyframe-generation sample.

Weak annotation:
{json.dumps(ann, ensure_ascii=False, indent=2)}

Requirements:
- 30-70 English words.
- Describe the overall activity, broad environment, main visible actors/hands, important objects/tools, and major task progression.
- If annotation exists, use it only as a weak prior and verify against images.
- If annotation is missing, infer a conservative overall activity from representative keyframes.
- Do not invent identities, exact locations, hidden causes, unseen objects, or unsupported outcomes.
- Do not produce a frame-by-frame list.
- Use concrete generation-friendly language.

Return strict JSON only:
{{
  "global_prompt": "30-70 word English description",
  "confidence": "high | medium | low",
  "evidence_summary": "Short support note"
}}
""".strip()
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for i, r in enumerate(reps):
        content.append({"type": "text", "text": f"Representative frame G{i:02d}; timestamp={r['timestamp']:.2f}s"})
        content.append({"type": "image_url", "image_url": {"url": make_labeled_url(Path(r["output_path"]), f"G{i:02d}", float(r["timestamp"]), side)}})
    return content


async def generate_global(uid, selection, source, compact, sample_dir, pool, sem, args):
    path = sample_dir / "global_prompt.json"
    if args.resume and path.exists() and not args.overwrite_prompts:
        existing = read_json(path, {})
        if existing.get("global_prompt"):
            return existing
    content = global_content(selection["selected_keyframes"], source, compact, args.global_image_side, args.global_representative_frames)
    obj = await call_vlm_json(pool, sem, args.model, content, args.global_max_tokens, args.retries, f"global video={uid}")
    if "_error" in obj:
        raise RuntimeError(f"global prompt failed: {obj.get('_error')}")
    gp = re.sub(r"\s+", " ", str(obj.get("global_prompt", "")).strip())
    out = {"video_uid": uid, "annotation_source": source, "global_prompt": gp, "confidence": str(obj.get("confidence", "")).strip(), "evidence_summary": str(obj.get("evidence_summary", "")).strip(), "endpoint": obj.get("_endpoint"), "latency_sec": obj.get("_latency_sec")}
    atomic_json(path, out)
    LOGGER.info("[GLOBAL_DONE] video=%s words=%d", uid, len(gp.split()))
    return out


def local_content(selected: Sequence[Dict[str, Any]], idx: int, global_prompt: str, source: str, compact: Dict[str, Any], side: int) -> List[Dict[str, Any]]:
    cur = selected[idx]
    prev = selected[idx - 1] if idx > 0 else None
    nxt = selected[idx + 1] if idx + 1 < len(selected) else None
    context = overlapping_context(compact, float(cur["timestamp"]) - 3.0, float(cur["timestamp"]) + 3.0)
    prompt = f"""
Generate a highly detailed LOCAL PROMPT for the CURRENT target frame in an Ego4D long-horizon keyframe-generation dataset.

Global prompt:
{global_prompt}

Annotation source: {source}
Weak local annotation:
{json.dumps(context, ensure_ascii=False, indent=2)}

You may inspect PREVIOUS and NEXT images only to resolve ambiguity, object identity, contact, and scene continuity. The final prompt must describe CURRENT as a self-contained image. Do not narrate the temporal sequence.

Primary objective:
The prompt will condition an image/video generator that must reconstruct the current frame as faithfully as possible. Therefore, preserve both task-relevant action details and rich scene/background information. Do NOT omit a visible background detail merely because it also appeared in adjacent frames; every target frame must be independently reconstructable.

Required visual coverage, from near to far:
1. Camera/viewpoint: first-person orientation, camera direction, framing, visible body parts, and whether the view looks down, forward, into a drawer, across a counter, etc.
2. Hands and people: left/right hand visibility, pose, finger/grip state, clothing or sleeves, exact hand-object contact, occlusion, and other visible people.
3. Main action and objects: current physical state only; object category, color, shape, material, texture, contents, open/closed state, wet/dry state, orientation, and exact relation to hands, containers, surfaces, or tools.
4. Spatial composition: foreground, central work area, left/right/above/below/inside/beside relations, object placement, distances when visually clear, and which objects partially occlude others.
5. Full environment: counters, sinks, drawers, cabinets, appliances, furniture, walls, backsplash, floor, containers, utensils, clutter, and other visually meaningful background elements. Include colors, materials, and approximate placement.
6. Image appearance: lighting direction or type when visible, shadows, reflections, highlights, depth, blur, water, steam, transparency, and major texture cues. Avoid the empty phrase "well-lit" unless you describe what the lighting visibly does.

Grounding rules:
- Target length: normally 110-180 English words; complex frames may use up to 220 words. A genuinely simple frame may use 85-110 words.
- Prioritize concrete visual facts over inferred intent.
- State only what is visible in CURRENT or strongly disambiguated by adjacent frames.
- Do not import an object from PREVIOUS/NEXT if it is not visible in CURRENT.
- Do not use temporal narration such as "previously", "next", "will", "going to", or "is about to".
- Do not claim grasping, holding, pouring, releasing, or contact unless visually supported. Use "near", "above", "partly occluding", or "reaching toward" when contact is uncertain.
- Do not guess brands, written labels, exact food identities, or locations unless clearly legible/visible or strongly supported by annotation.
- Do not describe emotions, identity, hidden causes, or intentions.
- Do not repeat the global task as filler.
- Start with the dominant current composition/action, then describe hands and objects, then the surrounding environment and visual appearance.
- Use fluent, concrete, generation-friendly English rather than a checklist.

Return strict JSON only:
{{
  "local_prompt": "A self-contained, highly detailed current-frame visual description",
  "camera_view": "Short camera/viewpoint summary",
  "left_hand_state": "Visible left-hand state or not visible",
  "right_hand_state": "Visible right-hand state or not visible",
  "current_action": "Short visually grounded current action/state",
  "main_objects": ["main visible object 1", "main visible object 2"],
  "background_elements": ["important visible background element 1", "important visible background element 2"],
  "uncertainties": ["Any important ambiguity; empty list if none"],
  "confidence": "high | medium | low"
}}
""".strip()
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for role, rec in [("PREVIOUS_CONTEXT_ONLY", prev), ("CURRENT_TARGET", cur), ("NEXT_CONTEXT_ONLY", nxt)]:
        if rec is None:
            continue
        content.append({"type": "text", "text": f"{role}; selected_order={rec['selected_order']}; timestamp={rec['timestamp']:.2f}s"})
        content.append({"type": "image_url", "image_url": {"url": make_labeled_url(Path(rec["output_path"]), role, float(rec["timestamp"]), side)}})
    return content


async def generate_one_local(uid, selected, idx, global_result, source, compact, local_dir, pool, sem, args):
    path = local_dir / f"{idx:06d}.json"
    if args.resume and path.exists() and not args.overwrite_prompts:
        existing = read_json(path, {})
        if existing.get("local_prompt"):
            return existing
    obj = await call_vlm_json(pool, sem, args.model, local_content(selected, idx, global_result["global_prompt"], source, compact, args.local_image_side), args.local_max_tokens, args.retries, f"local video={uid} frame={idx:06d}")
    cur = selected[idx]
    if "_error" in obj:
        out = {"video_uid": uid, "selected_order": idx, "timestamp": cur["timestamp"], "keyframe_path": cur["relative_path"], "status": "failed", "local_prompt": "", "error": obj.get("_error", "")}
    else:
        lp = re.sub(r"\s+", " ", str(obj.get("local_prompt", "")).strip())
        out = {
            "video_uid": uid,
            "selected_order": idx,
            "timestamp": cur["timestamp"],
            "keyframe_path": cur["relative_path"],
            "status": "ok",
            "local_prompt": lp,
            "camera_view": str(obj.get("camera_view", "")).strip(),
            "left_hand_state": str(obj.get("left_hand_state", "")).strip(),
            "right_hand_state": str(obj.get("right_hand_state", "")).strip(),
            "current_action": str(obj.get("current_action", "")).strip(),
            "main_objects": obj.get("main_objects", []),
            "background_elements": obj.get("background_elements", []),
            "uncertainties": obj.get("uncertainties", []),
            "confidence": str(obj.get("confidence", "")).strip(),
            "endpoint": obj.get("_endpoint"),
            "latency_sec": obj.get("_latency_sec"),
            "prompt_version": "ego4d_local_detailed_v2",
        }
        LOGGER.info("[LOCAL_DONE] video=%s frame=%d/%d words=%d", uid, idx + 1, len(selected), len(lp.split()))
    atomic_json(path, out)
    return out


async def generate_locals(uid, selection, global_result, source, compact, sample_dir, pool, sem, args):
    selected = selection["selected_keyframes"]
    local_dir = sample_dir / "local_prompts"
    if args.overwrite_prompts and local_dir.exists():
        shutil.rmtree(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    tasks = [generate_one_local(uid, selected, i, global_result, source, compact, local_dir, pool, sem, args) for i in range(len(selected))]
    results = await asyncio.gather(*tasks)
    results.sort(key=lambda x: x["selected_order"])
    atomic_json(sample_dir / "local_prompts.json", results)
    return results


def write_annotation(uid, video_path, source, compact, selection, global_result, locals_, sample_dir):
    selected = selection["selected_keyframes"]
    local_by_order = {int(x["selected_order"]): x for x in locals_}
    keyframes = [x["relative_path"] for x in selected]
    frame_prompts = [str(local_by_order.get(i, {}).get("local_prompt", "")) for i in range(len(selected))]
    failed = sum(1 for x in frame_prompts if not x.strip())
    annotation = {
        "id": f"ego4d_{uid}",
        "video_uid": uid,
        "image": keyframes[0] if keyframes else "",
        "keyframes": keyframes,
        "prompt": global_result["global_prompt"],
        "global_prompt": global_result["global_prompt"],
        "frame_prompts": frame_prompts,
        "annotation_source": source,
        "source_annotation": compact,
        "prompt_version": {
            "global": "ego4d_global_v1",
            "local": "ego4d_local_detailed_v2"
        },
        "statistics": {
            "num_candidate_frames": selection.get("num_candidate_frames", 0),
            "num_keyframes": len(keyframes),
            "num_failed_windows": selection.get("num_failed_windows", 0),
            "num_failed_local_prompts": failed,
            "mean_seconds_per_keyframe": ((float(selected[-1]["timestamp"]) - float(selected[0]["timestamp"])) / max(1, len(selected) - 1)) if len(selected) >= 2 else 0.0,
        },
    }
    atomic_json(sample_dir / "annotation.json", annotation)
    return annotation


def append_manifest(path: Path, row: Dict[str, Any]) -> None:
    fields = ["video_uid", "status", "annotation_source", "num_keyframes", "failed_windows", "failed_local_prompts", "output_dir", "message"]
    exists = path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


async def process_video(video: Path, resolver: AnnotationResolver, pool, selection_sem, caption_sem, summary_sem, args, manifest: Path):
    uid = video.stem
    final_dir = Path(args.output_root) / uid
    final_json = final_dir / "annotation.json"
    if args.resume and final_json.exists() and not args.overwrite_prompts and not args.overwrite_selection and not args.overwrite_frames:
        existing = read_json(final_json, {})
        if existing.get("keyframes") and existing.get("global_prompt") and existing.get("frame_prompts"):
            LOGGER.info("[SKIP_COMPLETE] video=%s", uid)
            return "complete"
    tmp_dir = Path(args.output_root) / f".{uid}.tmp"
    if tmp_dir.exists() and not args.resume:
        shutil.rmtree(tmp_dir)
    work_dir = final_dir if args.resume else tmp_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    source, raw, compact = resolver.resolve(uid)
    LOGGER.info("[VIDEO_START] video=%s annotation_source=%s", uid, source)
    try:
        frames = extract_1fps(video, work_dir / "frames_1fps", work_dir / "frames_1fps.json", args.sample_interval, args.frame_max_side, args.jpeg_quality, args.resume and not args.overwrite_frames)
        if len(frames) < args.min_candidate_frames:
            raise RuntimeError(f"too few candidate frames: {len(frames)} < {args.min_candidate_frames}")
        selection = await select_keyframes(uid, frames, compact, work_dir, pool, selection_sem, args)
        if selection["num_selected_keyframes"] < args.min_keyframes:
            raise RuntimeError(f"too few keyframes: {selection['num_selected_keyframes']} < {args.min_keyframes}")
        global_result = await generate_global(uid, selection, source, compact, work_dir, pool, summary_sem, args)
        locals_ = await generate_locals(uid, selection, global_result, source, compact, work_dir, pool, caption_sem, args)
        annotation = write_annotation(uid, video, source, compact, selection, global_result, locals_, work_dir)
        if args.write_debug_json:
            atomic_json(work_dir / "debug.json", {"video_path": str(video), "annotation_source": source, "raw_annotation": raw, "selection": selection, "global_result": global_result, "local_results": locals_})
        if not args.resume:
            if final_dir.exists():
                shutil.rmtree(final_dir)
            tmp_dir.rename(final_dir)
        if args.cleanup_candidates:
            shutil.rmtree(final_dir / "frames_1fps", ignore_errors=True)
        failed_local = annotation["statistics"]["num_failed_local_prompts"]
        status = "complete" if failed_local == 0 else "complete_with_local_failures"
        append_manifest(manifest, {"video_uid": uid, "status": status, "annotation_source": source, "num_keyframes": len(annotation["keyframes"]), "failed_windows": annotation["statistics"]["num_failed_windows"], "failed_local_prompts": failed_local, "output_dir": str(final_dir), "message": ""})
        LOGGER.info("[VIDEO_DONE] video=%s keyframes=%d failed_local=%d", uid, len(annotation["keyframes"]), failed_local)
        return status
    except Exception as exc:
        LOGGER.exception("[VIDEO_FAILED] video=%s error=%s", uid, repr(exc))
        append_manifest(manifest, {"video_uid": uid, "status": "failed", "annotation_source": source, "num_keyframes": 0, "failed_windows": 0, "failed_local_prompts": 0, "output_dir": str(final_dir), "message": repr(exc)})
        return "failed"


def discover_videos(root: Path, recursive: bool, uid_file: str) -> List[Path]:
    vids = sorted(root.glob("**/*.mp4" if recursive else "*.mp4"))
    if uid_file:
        allowed = {line.strip().split()[0] for line in Path(uid_file).read_text(encoding="utf-8").splitlines() if line.strip()}
        vids = [x for x in vids if x.stem in allowed]
    return vids


def collect_dataset(output_root: Path) -> List[Dict[str, Any]]:
    out = []
    for p in sorted(output_root.glob("*/annotation.json")):
        obj = read_json(p, {})
        if obj.get("video_uid") and obj.get("keyframes") and obj.get("global_prompt"):
            out.append(obj)
    return out


def write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


async def main_async(args):
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    setup_logging(output_root / "bulk_annotation.log", not args.no_console_log)
    resolver = AnnotationResolver(args.train_annotation, args.val_annotation)
    endpoints = [x.strip() for x in args.endpoints.split(",") if x.strip()]
    pool = EndpointPool(endpoints, args.request_timeout)
    sample_sem = asyncio.Semaphore(args.sample_concurrency)
    # One global cap is safer and faster than independent stage caps whose totals can
    # accidentally stack when several videos are processed at once.
    request_sem = asyncio.Semaphore(args.request_concurrency)
    videos = discover_videos(Path(args.video_dir), args.recursive, args.uid_file)
    if args.start_index > 0:
        videos = videos[args.start_index:]
    if args.max_videos > 0:
        videos = videos[:args.max_videos]
    LOGGER.info("[DATASET_START] videos=%d sample_concurrency=%d request_concurrency=%d endpoints=%d", len(videos), args.sample_concurrency, args.request_concurrency, len(endpoints))
    manifest = output_root / "run_manifest.csv"

    async def wrapped(video: Path):
        async with sample_sem:
            return await process_video(video, resolver, pool, request_sem, request_sem, request_sem, args, manifest)

    tasks = [asyncio.create_task(wrapped(v)) for v in videos]
    counters: Dict[str, int] = {}
    done = 0
    for fut in asyncio.as_completed(tasks):
        status = await fut
        counters[status] = counters.get(status, 0) + 1
        done += 1
        LOGGER.info("[DATASET_PROGRESS] done=%d/%d status_counts=%s", done, len(tasks), counters)
        if args.aggregate_every > 0 and done % args.aggregate_every == 0:
            records = collect_dataset(output_root)
            write_jsonl(output_root / "dataset.jsonl", records)
    records = collect_dataset(output_root)
    write_jsonl(output_root / "dataset.jsonl", records)
    atomic_json(output_root / "dataset_summary.json", {"num_discovered_videos": len(videos), "num_final_records": len(records), "status_counts": counters})
    LOGGER.info("[DATASET_DONE] records=%d status_counts=%s", len(records), counters)


def parse_args():
    p = argparse.ArgumentParser(description="Full Ego4D keyframe selection + global/local prompt annotation pipeline")
    p.add_argument("--video-dir", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--train-annotation", default="")
    p.add_argument("--val-annotation", default="")
    p.add_argument("--uid-file", default="")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--endpoints", default=DEFAULT_ENDPOINTS)
    p.add_argument("--request-timeout", type=float, default=300.0)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--sample-concurrency", type=int, default=4)
    p.add_argument("--request-concurrency", type=int, default=8, help="Total concurrent VLM requests across all endpoints and stages")
    p.add_argument("--sample-interval", type=float, default=1.0)
    p.add_argument("--frame-max-side", type=int, default=512)
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--min-candidate-frames", type=int, default=60)
    p.add_argument("--window-size", type=int, default=15)
    p.add_argument("--window-stride", type=int, default=15)
    p.add_argument("--min-window-frames", type=int, default=5)
    p.add_argument("--selection-image-side", type=int, default=512)
    p.add_argument("--selection-max-tokens", type=int, default=900)
    p.add_argument("--min-keyframes", type=int, default=41)
    p.add_argument("--global-representative-frames", type=int, default=16)
    p.add_argument("--global-image-side", type=int, default=384)
    p.add_argument("--global-max-tokens", type=int, default=500)
    p.add_argument("--local-image-side", type=int, default=512)
    p.add_argument("--local-max-tokens", type=int, default=1100)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite-frames", action="store_true")
    p.add_argument("--overwrite-selection", action="store_true")
    p.add_argument("--overwrite-prompts", action="store_true")
    p.add_argument("--cleanup-candidates", action="store_true")
    p.add_argument("--write-debug-json", action="store_true")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--max-videos", type=int, default=0)
    p.add_argument("--aggregate-every", type=int, default=10)
    p.add_argument("--no-console-log", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
