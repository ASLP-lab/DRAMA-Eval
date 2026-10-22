"""Alignment helpers for main-sim (turn spans from forced_alignment)."""

from __future__ import annotations

import json
import unicodedata
from collections import defaultdict
from typing import Any


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return 0x3400 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF


def _aligner_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in text.split():
        cleaned = "".join(
            ch
            for ch in raw
            if ch == "'" or unicodedata.category(ch).startswith(("L", "N"))
        )
        buf = ""
        for ch in cleaned:
            if _is_cjk(ch):
                if buf:
                    tokens.append(buf)
                    buf = ""
                tokens.append(ch)
            else:
                buf += ch
        if buf:
            tokens.append(buf)
    return tokens


def _aligned_speaker_spans(
    record: dict[str, Any],
) -> dict[str, list[list[dict[str, Any]]]]:
    raw_turns = record.get("generate_text") or "[]"
    if isinstance(raw_turns, str):
        turns = json.loads(raw_turns)
    else:
        turns = list(raw_turns)
    alignment = record.get("forced_alignment") or []
    if not turns or not alignment:
        raise ValueError("missing generate_text or forced_alignment; run aligned WER first")

    cursor = 0
    speaker_spans: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for speaker, text in turns:
        count = len(_aligner_tokens(str(text)))
        items = alignment[cursor : cursor + count]
        cursor += count
        if items:
            speaker_spans[str(speaker)].append(items)
    if cursor != len(alignment):
        raise ValueError(
            f"alignment/turn token mismatch: consumed={cursor}, aligned={len(alignment)}"
        )
    return speaker_spans
