"""Local Qwen3-ASR + ForcedAligner backend (singleton)."""

from __future__ import annotations

import threading
from typing import Any

from metrics.wer_utils import lang_to_asr

_lock = threading.Lock()
_asr_model = None
_asr_cfg_key: tuple[Any, ...] | None = None


def _cfg_key(wer_cfg: dict[str, Any]) -> tuple[Any, ...]:
    return (
        wer_cfg.get("asr_model", "Qwen/Qwen3-ASR-0.6B"),
        wer_cfg.get("forced_aligner_model", "Qwen/Qwen3-ForcedAligner-0.6B"),
        wer_cfg.get("device", "cuda:0"),
        wer_cfg.get("dtype", "bfloat16"),
        int(wer_cfg.get("max_new_tokens", 256)),
    )


def get_asr_model(wer_cfg: dict[str, Any]):
    global _asr_model, _asr_cfg_key
    key = _cfg_key(wer_cfg)
    with _lock:
        if _asr_model is not None and _asr_cfg_key == key:
            return _asr_model

        import torch
        from qwen_asr.inference.qwen3_asr import Qwen3ASRModel

        dtype_name = str(wer_cfg.get("dtype", "bfloat16")).lower()
        dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
        device = str(wer_cfg.get("device", "cuda:0"))
        asr_model = wer_cfg.get("asr_model", "Qwen/Qwen3-ASR-0.6B")
        aligner = wer_cfg.get("forced_aligner_model", "Qwen/Qwen3-ForcedAligner-0.6B")

        model = Qwen3ASRModel.from_pretrained(
            asr_model,
            dtype=dtype,
            device_map=device,
            forced_aligner=aligner,
            forced_aligner_kwargs=dict(dtype=dtype, device_map=device),
            max_inference_batch_size=int(wer_cfg.get("max_batch_size", 1)),
            max_new_tokens=int(wer_cfg.get("max_new_tokens", 256)),
        )
        _asr_model = model
        _asr_cfg_key = key
        return _asr_model


def transcribe(
    audio_path: str,
    *,
    language: str,
    wer_cfg: dict[str, Any],
    alignment_text: str | None = None,
) -> dict[str, Any]:
    asr_lang = lang_to_asr(language)
    model = get_asr_model(wer_cfg)
    results = model.transcribe(
        audio=audio_path,
        language=asr_lang,
        return_time_stamps=False,
    )
    if not results:
        return {"hypothesis": "", "detected_language": "", "forced_alignment": []}
    r = results[0]
    alignment = []
    if alignment_text:
        aligned = model.forced_aligner.align(
            audio=audio_path,
            text=alignment_text,
            language=asr_lang,
        )
        if aligned:
            alignment = [
                {
                    "text": str(item.text),
                    "start_time": float(item.start_time),
                    "end_time": float(item.end_time),
                }
                for item in aligned[0]
            ]
    return {
        "hypothesis": str(getattr(r, "text", "") or "").strip(),
        "detected_language": str(getattr(r, "language", "") or "").strip(),
        "forced_alignment": alignment,
    }
