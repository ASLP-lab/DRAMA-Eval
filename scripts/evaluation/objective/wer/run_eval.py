#!/usr/bin/env python3
"""WER/CER eval on a manifest (Qwen3-ASR + ForcedAligner).

Example:
  python run_eval.py --manifest ../en/MODEL/manifest.jsonl --metrics wer \\
    --out ../en/MODEL/results.shard_0.jsonl --shard-id 0 --num-shards 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import yaml

from metrics.wer import score_wer
from utils import read_jsonl, write_jsonl

PIPELINE_ROOT = Path(__file__).resolve().parent

METRIC_FNS = {
    "wer": score_wer,
}

METRIC_STATUS = {
    "wer": "wer_status",
}


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def check_record(record: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    for key in ("audio_path", "target_text", "context_info", "ref_audio_path"):
        val = str(record.get(key) or "").strip()
        if not val:
            issues.append(f"missing_{key}")
        elif key.endswith("_path") and not Path(val).is_file():
            issues.append(f"file_not_found:{key}={val}")

    audio = Path(str(record.get("audio_path") or ""))
    duration_sec = None
    if audio.is_file():
        try:
            import soundfile as sf

            duration_sec = float(sf.info(str(audio)).duration)
        except Exception as e:  # noqa: BLE001
            issues.append(f"audio_read_error:{e}")

    return {
        **record,
        "check_ok": len(issues) == 0,
        "check_issues": issues,
        "duration_sec": duration_sec,
    }


def run_metrics(
    record: dict[str, Any],
    *,
    metrics: list[str],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    out = dict(record)
    for name in metrics:
        fn = METRIC_FNS.get(name)
        if fn is None:
            out[f"{name}_status"] = "unknown_metric"
            continue
        try:
            patch = fn(out, cfg=cfg)
            out.update(patch)
        except NotImplementedError as e:
            out[f"{name}_status"] = "not_implemented"
            out[f"{name}_detail"] = str(e)
        except Exception as e:  # noqa: BLE001
            out[f"{name}_status"] = "error"
            out[f"{name}_detail"] = str(e)
    return out


def _metrics_done(row: dict[str, Any], metrics: list[str]) -> bool:
    for name in metrics:
        status_key = METRIC_STATUS.get(name)
        if status_key is None:
            continue
        if row.get(status_key) != "ok":
            return False
    return True


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    check_ok = sum(1 for r in rows if r.get("check_ok"))
    out: dict[str, Any] = {
        "total": n,
        "check_ok": check_ok,
        "check_failed": n - check_ok,
        "metrics_present": sorted({k for r in rows for k in r if k == "wer"}),
    }
    vals = [float(r["wer"]) for r in rows if isinstance(r.get("wer"), (int, float))]
    ok = sum(1 for r in rows if r.get("wer_status") == "ok")
    if ok or vals:
        mean = round(sum(vals) / len(vals), 4) if vals else None
        out["wer"] = {"total": n, "ok": ok, "mean": mean}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PIPELINE_ROOT / "config.yaml")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--metrics", default="wer", help="Only 'wer' is supported here")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    rows = read_jsonl(args.manifest)
    print(f"loaded {len(rows)} records from {args.manifest}", flush=True)

    checked = [check_record(r) for r in rows]
    if args.num_shards > 1:
        if not (0 <= args.shard_id < args.num_shards):
            raise SystemExit(f"bad shard-id={args.shard_id} num-shards={args.num_shards}")
        checked = [r for i, r in enumerate(checked) if i % args.num_shards == args.shard_id]
        print(
            f"shard={args.shard_id}/{args.num_shards} this_shard={len(checked)}",
            flush=True,
        )
    if args.check_only:
        write_jsonl(args.out, checked)
        summary = summarize(checked)
        args.out.with_suffix(".summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if summary["check_failed"]:
            raise SystemExit(1)
        return

    metric_list = [m.strip() for m in args.metrics.split(",") if m.strip()]
    for m in metric_list:
        if m not in METRIC_FNS:
            raise SystemExit(f"unsupported metric={m}; only wer is available in wer/code")

    existing: dict[str, dict[str, Any]] = {}
    if args.resume:
        resume_paths = [args.out]
        if args.resume_from is not None:
            resume_paths.append(args.resume_from)
        for path in resume_paths:
            if not path.is_file():
                continue
            for rec in read_jsonl(path):
                rid = str(rec.get("id") or "")
                if rid:
                    existing[rid] = rec
        print(f"resume: {len(existing)} completed rows", flush=True)

    results: list[dict[str, Any]] = []
    skipped_exist = 0
    for i, row in enumerate(checked, start=1):
        rid = str(row.get("id") or "")
        prev = existing.get(rid)
        if prev is not None and _metrics_done(prev, metric_list):
            results.append(prev)
            skipped_exist += 1
        else:
            base = dict(prev) if prev is not None else dict(row)
            for key, value in row.items():
                if key.endswith("_status") or key.endswith("_detail"):
                    continue
                if key == "wer":
                    continue
                base[key] = value
            scored = run_metrics(base, metrics=metric_list, cfg=cfg)
            results.append(scored)
            bits = []
            if scored.get("wer") is not None:
                bits.append(f"wer={scored['wer']}")
            print(
                f"[{i}/{len(checked)}] id={rid} " + (" ".join(bits) or "scored"),
                flush=True,
            )
        if i % 10 == 0 or i == len(checked):
            write_jsonl(args.out, results)

    if skipped_exist:
        print(f"skip_existing={skipped_exist}", flush=True)
    write_jsonl(args.out, results)
    summary = summarize(results)
    args.out.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"results -> {args.out}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
