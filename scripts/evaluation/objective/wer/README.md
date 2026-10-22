# WER / CER Evaluation

Whole-dialogue intelligibility metric for multi-turn TTS.

- **English** → word error rate (WER)
- **Chinese** → character error rate (CER; same code path, character tokenization)

Uses **Qwen3-ASR** for transcription and **Qwen3-ForcedAligner** for token-level timestamps (needed by main-sim).

## Requirements

- Python 3.10+
- CUDA GPU recommended
- Packages: `torch`, `transformers` (with Qwen3-ASR), `soundfile`, `PyYAML`, plus the ASR model’s own deps
- Hugging Face models (downloaded once, then can run offline):
  - `Qwen/Qwen3-ASR-0.6B`
  - `Qwen/Qwen3-ForcedAligner-0.6B`

Edit `config.yaml` if you use different model IDs or devices.

## Layout

```
wer/code/
├── build_test_audio_manifest.py   # wav + benchmark → manifest.jsonl
├── run_eval.py                    # WER/CER runner (shardable)
├── config.yaml
├── utils.py
└── metrics/
    ├── wer.py
    ├── wer_backend.py
    └── wer_utils.py
```

## Input

1. **Benchmark jsonl** — each sample needs at least:
   - `utt` (matches `{utt}.wav` stem)
   - `language` (`zh` / `en`)
   - `character` (with `wav_path` for reference speakers)
   - `content` (turns with `text`, `speaker_info`, optional `context`)

2. **Audio directory** — flat wavs:

```
<audio_root>/<model_name>/0001.wav
<audio_root>/<model_name>/0002.wav
...
```

## Usage

### 1. Build manifest

```bash
cd wer/code

python build_test_audio_manifest.py \
  --benchmark /path/to/bench.jsonl \
  --audio-root /path/to/audio_root \
  --out-root /path/to/out \
  --model YourModel
```

Writes `/path/to/out/YourModel/manifest.jsonl`.

### 2. Run WER / CER

Single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -u run_eval.py \
  --manifest /path/to/out/YourModel/manifest.jsonl \
  --metrics wer \
  --out /path/to/out/YourModel/results.jsonl
```

Multi-GPU (example: 8 shards):

```bash
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python -u run_eval.py \
    --manifest /path/to/out/YourModel/manifest.jsonl \
    --metrics wer \
    --out /path/to/out/YourModel/results.shard_${i}.jsonl \
    --shard-id $i --num-shards 8 &
done
wait
# then merge shard jsonl files into results.jsonl by id
```

Useful flags:

| flag | meaning |
|---|---|
| `--config config.yaml` | ASR / aligner settings |
| `--resume` | skip ids already `wer_status=ok` in `--out` |
| `--resume-from FILE` | also treat that jsonl as completed |
| `--check-only` | validate paths only, no ASR |
| `--shard-id` / `--num-shards` | data parallel over GPUs |

### 3. Outputs

- `results.jsonl` — per-sample fields include `wer`, `asr_hypothesis`, `forced_alignment`, …
- `results.summary.json` — aggregate (`total`, `ok`, `wer.mean`, …)

`forced_alignment` + `generate_text` in the results are the usual inputs for **main-sim**.

## Config (`config.yaml`)

```yaml
wer:
  mode: local
  asr_model: Qwen/Qwen3-ASR-0.6B
  forced_aligner_model: Qwen/Qwen3-ForcedAligner-0.6B
  device: cuda:0          # respects CUDA_VISIBLE_DEVICES
  dtype: bfloat16
  max_new_tokens: 256
  max_batch_size: 1
```

Offline tip: set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` after models are cached.
