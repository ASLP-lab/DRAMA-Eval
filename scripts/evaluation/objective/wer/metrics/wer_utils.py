"""Text normalization and WER computation for zh/en."""

from __future__ import annotations

import re
import unicodedata


def lang_to_asr(language: str) -> str | None:
    lang = (language or "").strip().lower()
    if lang.startswith("zh") or lang in {"cn", "chinese", "中文"}:
        return "Chinese"
    if lang.startswith("en") or lang in {"english", "英文"}:
        return "English"
    return None


_PUNCT_RE = re.compile(r"[^\w\s\u4e00-\u9fff]", re.UNICODE)


def _strip_punct(text: str) -> str:
    return _PUNCT_RE.sub("", text)


def normalize_for_wer(text: str, *, language: str) -> str:
    text = unicodedata.normalize("NFKC", (text or "").strip())
    lang = (language or "").lower()
    if lang.startswith("zh") or lang in {"cn", "chinese", "中文"}:
        # CJK char-level: drop whitespace and punctuation
        chars = [ch for ch in text if not ch.isspace() and not _PUNCT_RE.match(ch)]
        return "".join(chars)
    # English word-level
    text = text.lower()
    text = _strip_punct(text)
    return " ".join(text.split())


def _edit_distance(ref: list[str], hyp: list[str]) -> int:
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev, dp[j] = dp[j], cur
    return dp[m]


def compute_wer(reference: str, hypothesis: str, *, language: str) -> float:
    ref_norm = normalize_for_wer(reference, language=language)
    hyp_norm = normalize_for_wer(hypothesis, language=language)
    if not ref_norm:
        return 0.0 if not hyp_norm else 1.0

    lang = (language or "").lower()
    if lang.startswith("zh") or lang in {"cn", "chinese", "中文"}:
        ref_tokens = list(ref_norm)
        hyp_tokens = list(hyp_norm)
    else:
        ref_tokens = ref_norm.split()
        hyp_tokens = hyp_norm.split()

    dist = _edit_distance(ref_tokens, hyp_tokens)
    return dist / max(len(ref_tokens), 1)
