#!/usr/bin/env python3
"""Build eval manifests from test_audio/{model}/{utt}.wav + mint-bench2-data.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from utils import write_jsonl

PIPELINE_ROOT = Path(__file__).resolve().parent
DEFAULT_BENCH = Path("/mnt/node06_tmpdata1/data/Mint_bench2.0/our-bench/mint-bench2-data.jsonl")
DEFAULT_AUDIO_ROOT = Path("/mnt/node04_tmpdata2/Mint_bench2.0/test_audio")
DEFAULT_OUT_ROOT = Path("/mnt/node04_tmpdata2/Mint_bench2.0/test_result")


def load_json_stream(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    idx = 0
    n = len(text)
    rows: list[dict[str, Any]] = []
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        obj, end = decoder.raw_decode(text, idx)
        if not isinstance(obj, dict):
            raise ValueError(f"expected object at pos {idx} in {path}")
        rows.append(obj)
        idx = end
    return rows


def norm_lang(raw: str) -> str:
    s = (raw or "").strip().lower()
    if s in {"zh", "cn", "chinese", "中文"}:
        return "zh"
    if s in {"en", "eng", "english", "英文"}:
        return "en"
    return s or "zh"


def backend_name(model_dir: str) -> str:
    mapping = {
        "Harness-TTS": "harness_tts",
        "SoulX-Podcast": "soulx_podcast",
        "VibeVoice": "vibevoice",
    }
    return mapping.get(model_dir, model_dir.lower().replace("-", "_"))


def record_from_sample(
    *,
    sample: dict[str, Any],
    wav_path: Path,
    backend: str,
) -> dict[str, Any] | None:
    utt = str(sample.get("utt") or wav_path.stem).strip()
    characters = sample.get("character") or {}
    content = sample.get("content") or []
    if not content:
        return None

    order: list[str] = []
    turns: list[tuple[str, str]] = []
    ctx_parts = [f"[background] {str(sample.get('context_background') or '').strip()}"]
    for turn in content:
        name = str((turn.get("speaker_info") or {}).get("speaker_name") or "").strip()
        text = str(turn.get("text") or "").strip()
        if not text:
            continue
        if name and name not in order:
            order.append(name)
        turns.append((name, text))
        turn_ctx = str(turn.get("context") or "").strip()
        if turn_ctx:
            ctx_parts.append(f"[turn {turn.get('turn_num')}] {turn_ctx}")

    if not turns:
        return None

    name_to_sid = {name: i + 1 for i, name in enumerate(order)}
    generate_text: list[list[str]] = []
    ref_audio_paths: dict[str, str] = {}
    for name, text in turns:
        sid = name_to_sid.get(name)
        tag = f"S{sid}" if sid else "S1"
        generate_text.append([tag, text])
        info = characters.get(name) or {}
        wav = str(info.get("wav_path") or "").strip()
        if wav and tag not in ref_audio_paths:
            ref_audio_paths[tag] = wav

    if not ref_audio_paths:
        first = next(iter(characters.values()), {}) or {}
        wav = str(first.get("wav_path") or "").strip()
        if wav:
            ref_audio_paths["S1"] = wav

    ref_audio_path = next(iter(ref_audio_paths.values()), "")
    if not ref_audio_path:
        return None

    lang = norm_lang(str(sample.get("language") or ""))
    return {
        "id": utt,
        "backend": backend,
        "language": lang,
        "task": str(sample.get("eval_type") or "mint-bench2"),
        "dialogue_id": utt,
        "turn_id": None,
        "audio_path": str(wav_path),
        "ref_audio_path": ref_audio_path,
        "ref_audio_paths": ref_audio_paths,
        "target_text": " ".join(t for _, t in turns),
        "context_info": "\n".join(p for p in ctx_parts if p.strip()),
        "generate_text": json.dumps(generate_text, ensure_ascii=False),
        "meta_path": "",
        "prompt_path": "",
        "n_speakers": len(order),
        "n_turns": len(turns),
    }


def collect_model(audio_dir: Path, bench_index: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    backend = backend_name(audio_dir.name)
    rows: list[dict[str, Any]] = []
    missing_bench = 0
    for wav in sorted(audio_dir.glob("*.wav")):
        if wav.name.endswith(".partial.wav"):
            continue
        sample = bench_index.get(wav.stem)
        if sample is None:
            missing_bench += 1
            continue
        rec = record_from_sample(sample=sample, wav_path=wav, backend=backend)
        if rec is not None:
            rows.append(rec)
    if missing_bench:
        print(f"[warn] {audio_dir.name}: {missing_bench} wavs not in benchmark", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--model", action="append", default=[], help="Model folder name; default: all")
    args = parser.parse_args()

    bench = load_json_stream(args.benchmark)
    index = {str(s.get("utt") or "").strip(): s for s in bench}
    models = args.model or [
        p.name for p in sorted(args.audio_root.iterdir()) if p.is_dir() and any(p.glob("*.wav"))
    ]
    for name in models:
        audio_dir = args.audio_root / name
        if not audio_dir.is_dir():
            print(f"[skip] missing {audio_dir}", flush=True)
            continue
        rows = collect_model(audio_dir, index)
        out = args.out_root / name / "manifest.jsonl"
        write_jsonl(out, rows)
        print(f"wrote {len(rows)} -> {out}", flush=True)


if __name__ == "__main__":
    main()
