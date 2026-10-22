"""Seed-TTS eval SIM: WavLM-large SV finetune + cosine similarity.

Uses the official UniSpeech speaker-verification wrapper from seed-tts-eval:
  WavLM-large SSL features -> ECAPA-TDNN -> cosine(test, reference)
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np

_lock = threading.Lock()
_model = None
_device = None
_sim_cfg_key: tuple[Any, ...] | None = None

TARGET_SR = 16000
MIN_SAMPLES = 16000  # ~1s; WavLM CNN front-end needs enough samples

DEFAULT_UNISPEECH = (
    "/mnt/node06_tmpdata0/hyzhang/code/seed-tts-eval/"
    "thirdparty/UniSpeech/downstreams/speaker_verification"
)
DEFAULT_CHECKPOINT = (
    "/mnt/node06_tmpdata0/hyzhang/code/seed-tts-eval/"
    "wavlm_large_finetune/wavlm_large_finetune.pth"
)
DEFAULT_SSL_CKPT = (
    "/mnt/node06_tmpdata0/hyzhang/code/seed-tts-eval/wavlm/wavlm_large.pt"
)


def _cfg_key(sim_cfg: dict[str, Any]) -> tuple[Any, ...]:
    return (
        sim_cfg.get("checkpoint", DEFAULT_CHECKPOINT),
        sim_cfg.get("ssl_ckpt", DEFAULT_SSL_CKPT),
        sim_cfg.get("device", "cuda:0"),
        sim_cfg.get("model", "wavlm_large"),
    )


def _patch_wavlm_loader(ssl_ckpt: str) -> None:
    """Force UniSpeech to load local wavlm_large.pt via cached s3prl hub."""
    import models.ecapa_tdnn as ecapa

    def _load_wavlm(feat_type: str):
        hub_cache = Path.home() / ".cache" / "torch" / "hub" / "s3prl_s3prl_main"
        if hub_cache.is_dir() and str(hub_cache) not in sys.path:
            sys.path.insert(0, str(hub_cache))
        import s3prl.hub  # noqa: WPS433

        if feat_type == "wavlm_large" and ssl_ckpt and Path(ssl_ckpt).is_file():
            print(f"[INFO] loading wavlm_large from {ssl_ckpt}", flush=True)
            return s3prl.hub.wavlm_local(ckpt=ssl_ckpt)
        load_fn = getattr(s3prl.hub, feat_type, None)
        if load_fn is None:
            raise ValueError(f"unknown feat_type={feat_type}")
        return load_fn()

    ecapa._load_wavlm = _load_wavlm


def _get_model(sim_cfg: dict[str, Any]):
    global _model, _device, _sim_cfg_key
    key = _cfg_key(sim_cfg)
    with _lock:
        if _model is not None and _sim_cfg_key == key:
            return _model, _device

        unispeech_dir = str(sim_cfg.get("unispeech_dir", DEFAULT_UNISPEECH))
        if unispeech_dir not in sys.path:
            sys.path.insert(0, unispeech_dir)

        import torch
        from models.ecapa_tdnn import ECAPA_TDNN_SMALL

        ssl_ckpt = str(sim_cfg.get("ssl_ckpt", DEFAULT_SSL_CKPT))
        checkpoint = str(sim_cfg.get("checkpoint", DEFAULT_CHECKPOINT))
        device = str(sim_cfg.get("device", "cuda:0"))
        model_name = str(sim_cfg.get("model", "wavlm_large"))

        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"missing seed-tts WavLM SV ckpt: {checkpoint}")
        if model_name != "wavlm_large":
            raise ValueError(f"unsupported sim.model={model_name}; expected wavlm_large")

        _patch_wavlm_loader(ssl_ckpt)
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large")
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["model"], strict=False)
        model = model.to(device)
        model.eval()
        _model = model
        _device = device
        _sim_cfg_key = key
        return _model, _device


def _load_mono_16k(path: str) -> np.ndarray:
    import torch
    import torchaudio

    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    arr = wav.squeeze(0).numpy().astype(np.float32)
    if arr.size < MIN_SAMPLES:
        arr = np.concatenate([arr, np.zeros(MIN_SAMPLES - arr.size, dtype=np.float32)])
    return arr


def _embedding_waveform(wav: np.ndarray, sim_cfg: dict[str, Any]) -> np.ndarray:
    import torch

    model, device = _get_model(sim_cfg)
    if wav.size < MIN_SAMPLES:
        wav = np.pad(wav, (0, MIN_SAMPLES - wav.size))
    tensor = torch.from_numpy(wav).float().unsqueeze(0).to(device)
    with torch.inference_mode():
        emb = model(tensor)
    return emb.squeeze(0).detach().cpu().numpy()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
