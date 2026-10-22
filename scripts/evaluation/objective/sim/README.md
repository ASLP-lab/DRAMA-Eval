# Main-Sim Evaluation

Main-speaker **cross-turn voice consistency** for multi-turn / multi-speaker TTS.

## Metric definition

1. Pick the **main speaker** = speaker with the largest total aligned duration  
2. Keep only that speaker’s turns with duration **≥ `min-sec`** (default `2.0`)  
3. Use the **earliest kept turn** as the anchor embedding  
4. Compare every later kept turn to the anchor (cosine similarity)  
5. Report per-dialogue mean (and std); summary is the mean of those means  

A sample is **skipped** if the main speaker has fewer than **2** turns ≥ `min-sec` (cannot form an anchor–pair).

Speaker embeddings: **Seed-TTS WavLM-large** speaker verification (ECAPA-TDNN).

## Requirements

- Python 3.10+
- CUDA GPU recommended
- Packages: `torch`, `torchaudio`, `numpy`, `PyYAML`, plus UniSpeech / s3prl stack used by Seed-TTS eval
- Local weights (set paths in `config.yaml`):
  - WavLM-large finetuned SV checkpoint
  - WavLM-large SSL checkpoint
  - UniSpeech `speaker_verification` code directory (provides `models.ecapa_tdnn`)

Typical upstream: [seed-tts-eval](https://github.com/BytedanceSpeech/seed-tts-eval) (or equivalent WavLM SV export).

## Layout

```
sim/code/
├── run_main_sim.py       # main-sim runner (shardable)
├── config.yaml
└── metrics/
    ├── sim.py            # turn spans from forced_alignment
    └── sim_backend.py    # embedding + cosine
```

## Input

A jsonl of records that already contain:

| field | role |
|---|---|
| `id` | sample id |
| `audio_path` | path to the synthesized dialogue wav |
| `generate_text` | turn list `[["S1","..."],["S2","..."], ...]` (json string or list) |
| `forced_alignment` | token-level `{text, start_time, end_time}` list |

The usual source is **WER/CER `results.jsonl`** (same pipeline’s ASR + ForcedAligner).

## Setup

```bash
cp config.example.yaml config.yaml
# then edit checkpoint / ssl_ckpt / unispeech_dir
```

```yaml
sim:
  enabled: true
  backend: seed_tts
  model: wavlm_large
  checkpoint: /path/to/wavlm_large_finetune.pth
  ssl_ckpt: /path/to/wavlm_large.pt
  unispeech_dir: /path/to/UniSpeech/downstreams/speaker_verification
  device: cuda:0          # respects CUDA_VISIBLE_DEVICES
```

## Usage

### Single GPU

```bash
cd sim/code

CUDA_VISIBLE_DEVICES=0 python -u run_main_sim.py \
  --results /path/to/wer_results.jsonl \
  --out /path/to/main_sim.jsonl \
  --min-sec 2.0
```

### Multi-GPU (example: 8 shards)

```bash
cd sim/code

for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python -u run_main_sim.py \
    --results /path/to/wer_results.jsonl \
    --out /path/to/main_sim.shard_${i}.jsonl \
    --min-sec 2.0 \
    --shard-id $i --num-shards 8 &
done
wait
# merge shard jsonl files into main_sim.jsonl by id
```

Useful flags:

| flag | meaning |
|---|---|
| `--min-sec` | minimum turn duration to keep (default `2.0`) |
| `--resume` | skip ids already `main_sim_status=ok` in `--out` |
| `--shard-id` / `--num-shards` | data parallel over GPUs |
| `--limit N` | debug: only first N rows of this shard |

## Outputs

- `main_sim.jsonl` — per sample: `main_sim_status`, `main_sim_mean`, `main_sim_std`, `main_speaker`, …
- `main_sim.summary.json` — aggregate, e.g.:

```json
{
  "total": 537,
  "ok": 523,
  "skipped": 14,
  "error": 0,
  "mean_of_means": 0.6982,
  "std_of_means": 0.1244,
  "median_of_means": 0.7256
}
```

Report **`mean_of_means`** as the leaderboard score (average over successfully scored dialogues).
