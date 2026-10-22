"""WER via local Qwen3-ASR (+ optional ForcedAligner loaded with ASR)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from metrics.wer_backend import transcribe
from metrics.wer_utils import compute_wer, normalize_for_wer


def score_wer(record: dict[str, Any], *, cfg: dict[str, Any]) -> dict[str, Any]:
    wer_cfg = cfg.get("wer") or {}
    mode = str(wer_cfg.get("mode", "local")).lower()
    if mode != "local":
        return {
            "wer": None,
            "wer_status": "skipped",
            "wer_detail": f"unsupported wer.mode={mode!r}",
        }

    audio_path = str(record.get("audio_path") or "")
    reference = str(record.get("target_text") or "").strip()
    language = str(record.get("language") or "zh")

    if not audio_path or not Path(audio_path).is_file():
        return {"wer": None, "wer_status": "error", "wer_detail": "missing audio_path"}
    if not reference:
        return {"wer": None, "wer_status": "error", "wer_detail": "missing target_text"}

    try:
        asr_out = transcribe(
            audio_path,
            language=language,
            wer_cfg=wer_cfg,
            alignment_text=reference,
        )
        hypothesis = asr_out["hypothesis"]
        wer = compute_wer(reference, hypothesis, language=language)
        return {
            "wer": round(float(wer), 4),
            "wer_status": "ok",
            "wer_backend": "qwen3_asr_local",
            "asr_model": wer_cfg.get("asr_model", "Qwen/Qwen3-ASR-0.6B"),
            "forced_aligner_model": wer_cfg.get(
                "forced_aligner_model", "Qwen/Qwen3-ForcedAligner-0.6B"
            ),
            "asr_hypothesis": hypothesis,
            "asr_detected_language": asr_out.get("detected_language"),
            "forced_alignment": asr_out.get("forced_alignment", []),
            "forced_alignment_status": "ok"
            if asr_out.get("forced_alignment")
            else "empty",
            "wer_reference_norm": normalize_for_wer(reference, language=language),
            "wer_hypothesis_norm": normalize_for_wer(hypothesis, language=language),
        }
    except Exception as e:  # noqa: BLE001
        return {
            "wer": None,
            "wer_status": "error",
            "wer_detail": str(e),
        }
