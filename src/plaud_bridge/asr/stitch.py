"""
Chunk stitching.

Chunks overlap on purpose so a word cut in half at a boundary survives in one
of the two copies. The cost is duplicated speech in the overlap region, which
this module removes. The approach is deliberately simple and deterministic:
drop segments from the incoming chunk that both fall inside the overlap window
and closely resemble text we already have. No model, no randomness, no
surprises at 2am.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from ..models import Segment

# Unicode-aware on purpose. An ASCII-only class reduces every Chinese, Russian,
# Greek, or Hebrew segment to the empty string, at which point any two of them
# compare as identical and the dedup pass deletes the second one. That silently
# destroys a large fraction of a non-Latin transcript.
_NORM = re.compile(r"[^\w ]+", re.UNICODE)


def _normalise(text: str) -> str:
    return _NORM.sub("", text.lower()).strip()


def _similar(a: str, b: str) -> float:
    na, nb = _normalise(a), _normalise(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    return SequenceMatcher(None, na, nb).ratio()


def stitch(chunk_results: list[list[Segment]], overlaps: list[float],
           starts: list[float] | None = None,
           similarity_threshold: float = 0.72) -> list[Segment]:
    """
    Merge per-chunk segment lists into one timeline.

    `overlaps[i]` is how many seconds at the START of chunk i repeat the end of
    chunk i-1. `starts[i]` is where chunk i begins in absolute time. Segments
    are already shifted into absolute time by the provider.

    `starts` matters more than it looks. The duplicate window is
    `[chunk_start, chunk_start + overlap]`, and anchoring it to the first
    surviving segment instead is wrong whenever the chunk opens with silence --
    which is most of the time, because the local provider runs a VAD filter that
    trims it. The window then extends past the real overlap and deletes genuine
    speech that merely resembles something said a moment earlier. Callers that
    know the chunk boundaries should always pass them.
    """
    if not chunk_results:
        return []

    merged: list[Segment] = list(chunk_results[0])

    for idx in range(1, len(chunk_results)):
        incoming = chunk_results[idx]
        overlap = overlaps[idx] if idx < len(overlaps) else 0.0
        if overlap <= 0 or not merged:
            merged.extend(incoming)
            continue

        if starts is not None and idx < len(starts):
            chunk_start = starts[idx]
        else:
            chunk_start = incoming[0].start if incoming else 0.0
        boundary = chunk_start + overlap
        # Only compare against the tail of what we already have. Comparing the
        # whole transcript would be O(n^2) and would also produce false
        # positives on genuinely repeated phrases much earlier in the meeting.
        tail = [s for s in merged if s.end >= boundary - overlap - 2.0]

        kept: list[Segment] = []
        for seg in incoming:
            if seg.start < boundary:
                if any(_similar(seg.text, prev.text) >= similarity_threshold for prev in tail):
                    continue  # duplicate from the overlap window
            kept.append(seg)
        merged.extend(kept)

    merged.sort(key=lambda s: (s.start, s.end))

    # Collapse exact adjacent repeats that survived, then drop empties. Same
    # speaker only: two people each answering "Yeah." a second apart is a real
    # exchange, not a duplicated segment, and merging them loses a turn.
    deduped: list[Segment] = []
    for seg in merged:
        if not seg.text.strip():
            continue
        if (
            deduped
            and deduped[-1].speaker == seg.speaker
            and _normalise(deduped[-1].text) == _normalise(seg.text)
            and abs(deduped[-1].start - seg.start) < 1.5
        ):
            deduped[-1].end = max(deduped[-1].end, seg.end)
            continue
        deduped.append(seg)

    return deduped
