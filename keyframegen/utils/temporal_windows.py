from dataclasses import dataclass, asdict
from typing import List, Sequence


@dataclass(frozen=True)
class TemporalWindow:
    """Prompt-only metadata for one autoregressive keyframe window."""

    window_id: int
    start: int
    end: int
    prompt_indices: List[int]
    history_count: int
    emit_start_slot: int
    emit_count: int

    @property
    def valid_count(self) -> int:
        return self.end - self.start

    @property
    def emit_indices(self) -> List[int]:
        return self.prompt_indices[self.emit_start_slot:]

    @property
    def drop_count(self) -> int:
        return self.emit_start_slot

    def to_dict(self) -> dict:
        data = asdict(self)
        data["valid_count"] = self.valid_count
        data["emit_indices"] = self.emit_indices
        data["drop_count"] = self.drop_count
        return data


def build_temporal_windows(
    frame_prompts: Sequence[str],
    window_size: int = 21,
    history_size: int = 1,
) -> List[TemporalWindow]:
    """
    Split prompt slots into overlapping autoregressive windows.

    Example for 61 prompts with window_size=21 and history_size=1:
      [0:21], [20:41], [40:61]
    """
    n = len(frame_prompts)
    if n == 0:
        raise ValueError("frame_prompts must not be empty")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if history_size < 0:
        raise ValueError(f"history_size must be non-negative, got {history_size}")
    if history_size >= window_size:
        raise ValueError(
            f"history_size must be smaller than window_size, got "
            f"history_size={history_size}, window_size={window_size}"
        )

    stride = window_size - history_size
    windows: List[TemporalWindow] = []
    start = 0

    while start < n:
        end = min(start + window_size, n)
        history_count = 0 if start == 0 else min(history_size, start)
        prompt_indices = list(range(start, end))
        emit_start_slot = history_count
        emit_count = max(0, len(prompt_indices) - emit_start_slot)

        windows.append(
            TemporalWindow(
                window_id=len(windows),
                start=start,
                end=end,
                prompt_indices=prompt_indices,
                history_count=history_count,
                emit_start_slot=emit_start_slot,
                emit_count=emit_count,
            )
        )

        if end >= n:
            break
        start += stride

    return windows


def select_window_prompts(
    frame_prompts: Sequence[str],
    window: TemporalWindow,
) -> List[str]:
    return [frame_prompts[i] for i in window.prompt_indices]
