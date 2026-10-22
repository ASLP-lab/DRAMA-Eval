#!/usr/bin/env python3
"""Main-speaker voice consistency (anchor = first >=2s turn).

Rules:
  1) Main speaker = max cumulative aligned duration.
  2) Keep only turns with duration >= MIN_SEC (default 2.0).
  3) Anchor = earliest kept turn; compare every later kept turn to anchor.
  4) Report mean / std of those pair similarities.

Reads existing eval results.jsonl (needs generate_text + forced_alignment).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PIPELINE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE))

from metrics.sim import _aligned_speaker_spans  # noqa: E402
from metrics.sim_backend import (  # noqa: E402
    TARGET_SR,
    _embedding_waveform,
    _load_mono_16k,
    cosine_sim,
)

MIN_SEC_DEFAULT = 2.0


def load_config() -> dict[str, Any]:
    cfg_path = PIPELINE / "config.yaml"
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}


def turn_spans(
    record: dict[str, Any],
) -> dict[str, list[tuple[float, float]]]:
    """speaker -> list of (start, end) per turn, in temporal order."""
    spans = _aligned_speaker_spans(record)
    out: dict[str, list[tuple[float, float]]] = {}
    for speaker, turns in spans.items():
        out[speaker] = [
            (float(items[0]["start_time"]), float(items[-1]["end_time"]))
            for items in turns
            if items
        ]
    return out


def pick_main_speaker(spans: dict[str, list[tuple[float, float]]]) -> str:
    totals = {
        sp: sum(max(0.0, e - s) for s, e in segs) for sp, segs in spans.items()
    }
    if not totals:
        raise ValueError("no speaker spans")
    return max(totals, key=totals.get)


def score_main_consistency(
    record: dict[str, Any],
    *,
    sim_cfg: dict[str, Any],
    min_sec: float = MIN_SEC_DEFAULT,
) -> dict[str, Any]:
    audio_path = str(record.get("audio_path") or "")
    if not audio_path or not Path(audio_path).is_file():
        return {
            "main_sim_status": "error",
            "main_sim_detail": "missing audio_path",
        }

    try:
        spans = turn_spans(record)
    except Exception as e:  # noqa: BLE001
        return {
            "main_sim_status": "error",
            "main_sim_detail": f"alignment:{e}",
        }

    if not spans:
        return {
            "main_sim_status": "skipped",
            "main_sim_detail": "empty speaker spans",
        }

    main = pick_main_speaker(spans)
    dur_tot = {
        sp: round(sum(max(0.0, e - s) for s, e in segs), 4)
        for sp, segs in spans.items()
    }
    kept = [(s, e) for s, e in spans[main] if (e - s) >= min_sec]
    if len(kept) < 2:
        return {
            "main_sim_status": "skipped",
            "main_sim_detail": f"need >=2 segments >= {min_sec}s for main={main}",
            "main_speaker": main,
            "main_speaker_duration": dur_tot.get(main),
            "speaker_durations": dur_tot,
            "main_n_turns_raw": len(spans[main]),
            "main_n_turns_kept": len(kept),
            "main_sim_mean": None,
            "main_sim_std": None,
            "main_sim_n_pairs": 0,
        }

    wav = _load_mono_16k(audio_path)
    embs = []
    kept_meta = []
    for s, e in kept:
        a0 = max(0, int(s * TARGET_SR))
        a1 = min(wav.size, int(e * TARGET_SR))
        if a1 <= a0:
            continue
        embs.append(_embedding_waveform(wav[a0:a1], sim_cfg))
        kept_meta.append({"start": round(s, 3), "end": round(e, 3), "dur": round(e - s, 3)})

    if len(embs) < 2:
        return {
            "main_sim_status": "skipped",
            "main_sim_detail": "embedding failed for kept segments",
            "main_speaker": main,
            "main_speaker_duration": dur_tot.get(main),
            "speaker_durations": dur_tot,
            "main_n_turns_kept": len(kept_meta),
            "main_sim_mean": None,
            "main_sim_std": None,
            "main_sim_n_pairs": 0,
        }

    anchor = embs[0]
    pair_sims = [round(cosine_sim(anchor, e), 4) for e in embs[1:]]
    mean = float(np.mean(pair_sims))
    std = float(np.std(pair_sims, ddof=1)) if len(pair_sims) >= 2 else None

    return {
        "main_sim_status": "ok",
        "main_speaker": main,
        "main_speaker_duration": dur_tot.get(main),
        "speaker_durations": dur_tot,
        "main_n_turns_raw": len(spans[main]),
        "main_n_turns_kept": len(kept_meta),
        "main_anchor": kept_meta[0],
        "main_segments_kept": kept_meta,
        "main_sim_pairs": pair_sims,
        "main_sim_mean": round(mean, 4),
        "main_sim_std": round(std, 4) if std is not None else None,
        "main_sim_n_pairs": len(pair_sims),
        "main_sim_min_sec": min_sec,
        "main_sim_backend": "seed_tts_wavlm_large_main_anchor",
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if r.get("main_sim_status") == "ok"]
    means = [float(r["main_sim_mean"]) for r in ok if r.get("main_sim_mean") is not None]
    stds = [float(r["main_sim_std"]) for r in ok if isinstance(r.get("main_sim_std"), (int, float))]
    n_pairs = [int(r["main_sim_n_pairs"]) for r in ok]
    return {
        "total": len(rows),
        "ok": len(ok),
        "skipped": sum(1 for r in rows if r.get("main_sim_status") == "skipped"),
        "error": sum(1 for r in rows if r.get("main_sim_status") == "error"),
        "mean_of_means": round(float(np.mean(means)), 4) if means else None,
        "std_of_means": round(float(np.std(means, ddof=1)), 4) if len(means) >= 2 else None,
        "median_of_means": round(float(np.median(means)), 4) if means else None,
        "p10_of_means": round(float(np.percentile(means, 10)), 4) if means else None,
        "p90_of_means": round(float(np.percentile(means, 90)), 4) if means else None,
        "mean_of_pair_stds": round(float(np.mean(stds)), 4) if stds else None,
        "avg_n_pairs": round(float(np.mean(n_pairs)), 3) if n_pairs else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="Input results.jsonl with forced_alignment")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-sec", type=float, default=MIN_SEC_DEFAULT)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config()
    sim_cfg = dict(cfg.get("sim") or {})
    sim_cfg["device"] = "cuda:0"  # respect CUDA_VISIBLE_DEVICES

    rows = [json.loads(l) for l in args.results.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.num_shards > 1:
        rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard_id]
    if args.limit > 0:
        rows = rows[: args.limit]

    print(
        f"loaded shard={args.shard_id}/{args.num_shards} n={len(rows)} from {args.results}",
        flush=True,
    )

    existing: dict[str, dict[str, Any]] = {}
    if args.resume and args.out.is_file():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            rid = str(rec.get("id") or "")
            if rid and rec.get("main_sim_status") == "ok":
                existing[rid] = rec
        print(f"resume: {len(existing)} done", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_rows: list[dict[str, Any]] = []
    t0 = time.time()
    for i, row in enumerate(rows, start=1):
        rid = str(row.get("id") or "")
        if rid in existing:
            out_rows.append(existing[rid])
            continue
        scored = {
            "id": rid,
            "backend": row.get("backend"),
            "language": row.get("language"),
            "audio_path": row.get("audio_path"),
            "n_speakers": row.get("n_speakers"),
            "n_turns": row.get("n_turns"),
        }
        try:
            patch = score_main_consistency(row, sim_cfg=sim_cfg, min_sec=args.min_sec)
        except Exception as e:  # noqa: BLE001
            err = str(e)
            if "out of memory" in err.lower():
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass
                patch = {
                    "main_sim_status": "error",
                    "main_sim_detail": f"oom:{err[:200]}",
                }
            else:
                patch = {
                    "main_sim_status": "error",
                    "main_sim_detail": err[:400],
                }
        scored.update(patch)
        out_rows.append(scored)
        bits = []
        if scored.get("main_sim_mean") is not None:
            bits.append(f"mean={scored['main_sim_mean']}")
            bits.append(f"std={scored.get('main_sim_std')}")
            bits.append(f"pairs={scored.get('main_sim_n_pairs')}")
            bits.append(f"spk={scored.get('main_speaker')}")
        print(
            f"[{i}/{len(rows)}] id={rid} status={scored.get('main_sim_status')} "
            + " ".join(bits),
            flush=True,
        )
        if i % 10 == 0 or i == len(rows):
            args.out.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows) + "\n",
                encoding="utf-8",
            )

    args.out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows) + "\n",
        encoding="utf-8",
    )
    summary = summarize(out_rows)
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"done in {time.time()-t0:.1f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
