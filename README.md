


<p align="center">
  <img src="https://github.com/user-attachments/assets/6992d6ba-923e-4b10-9f3f-df41c64080f5" alt="DRAMA-Eval Logo" width="250"/>
</p>

<h1 align="center">DRAMA-Eval: A Benchmark for Context-Aware, Multi-Turn, Multi-Speaker Speech Generation</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2610.31796">
    <img src="https://img.shields.io/badge/Paper-Arxiv-red.svg" alt="Paper">
  </a>
  <a href="https://huggingface.co/datasets/ASLP-lab/DRAMA-Eval">
    <img src="https://img.shields.io/badge/Dataset-HuggingFace-yellow.svg" alt="Dataset">
  </a>
  <a href="https://huggingface.co/datasets/ASLP-lab/DRAMA-Eval-Results">
    <img src="https://img.shields.io/badge/Results-HuggingFace-orange.svg" alt="Results">
  </a>
  <a href="https://ASLP-lab.github.io/DRAMA-Eval-Demo">
    <img src="https://img.shields.io/badge/Demo-Webpage-blue.svg" alt="Demo">
  </a>
</p>

**DRAMA-Eval** is a comprehensive bilingual (English & Chinese) benchmark designed to evaluate context-aware, multi-turn, and multi-speaker speech generation systems. By utilizing structured script contexts grounded in real-world audio, it offers a holistic evaluation framework covering objective, local, and global aspects of generation quality.

<a href="https://huggingface.co/datasets/ASLP-lab/DRAMA-Eval/resolve/main/assets/video.mp4">👉 <strong>Click Here to Watch the DRAMA-Eval Demo Video 🎥</strong> <sup>1</sup></a>


## 📖 Introduction

Recent advances in speech generation are rapidly shifting toward context-aware, multi-turn, and multi-speaker synthesis. While these capabilities are in high demand for applications such as audiobook and audio drama production, evaluation methodologies have not kept pace with these advances. 

DRAMA-Eval bridges this fundamental gap by providing a unified evaluation protocol that seamlessly integrates:

- **Objective Metrics**: Textual fidelity (WER/CER) and Speaker Consistency (WavLM-based Cosine Similarity).
- **Local LALM Metrics (B-Level)**: Instruction Following (B1), Prosody & Fluency (B2), and Local Expressiveness (B3) via an LLM-as-a-Judge approach.
- **Global LALM Metrics (C-Level)**: Interaction Naturalness (C1), Global Flow (C2), and Scene Immersion (C3) through macro-level LLM evaluation.

## 🗂️ Repository Structure

To adhere to the best practices of open-source academic research, **code and data are fully decoupled**. This GitHub repository remains extremely lightweight, hosting only evaluation pipelines and lightweight metadata, while all heavy audio files are hosted on Hugging Face.

```text
DRAMA-Eval/
├── data/                       # Script annotations (JSONL) and metadata
│   ├── en/                     # English evaluation scripts (en-data.jsonl)
│   ├── zh/                     # Chinese evaluation scripts (zh-data.jsonl)
│   └── prompt_wav/             # Reference audio metadata (selected_metadata.jsonl)
├── scripts/                    
│   ├── data_preparation/       # Dataset initialization tools (linking HF data)
│   └── evaluation/             # Complete evaluation pipeline
│       ├── objective/          # Code for WER/CER and Speaker Similarity
│       └── lalm/               # Code for LLM-as-a-Judge (Local & Global AB tests)
├── Leaderboard.md              # Auto-generated benchmark results
└── README.md
```

## 🚀 Quick Start

### 1. Environment Setup & Data Initialization

Clone this repository and download the corresponding heavy audio assets (Ground Truth & Reference Audios) from our Hugging Face Dataset. *Note: Ensure `git-lfs` is installed before cloning the Hugging Face repository.*

```bash
# 1. Clone the GitHub code repository
git clone https://github.com/ASLP-lab/DRAMA-Eval.git
cd DRAMA-Eval

# 2. Clone the Hugging Face audio dataset (Requires Git LFS)
git lfs install
git clone https://huggingface.co/datasets/ASLP-lab/DRAMA-Eval ../DRAMA-Eval-HF

# 3. Initialize the local dataset (syncing audio files into the data/ directory)
python scripts/data_preparation/init_dataset.py --hf_dir ../DRAMA-Eval-HF
```

### 2. Run the Evaluation Pipeline

Our automated evaluation pipeline connects objective timestamping with LLM-based judging. **Note: You must run the Objective WER evaluation first to generate the forced-alignment timestamps required by the Local LLM-Judge.**

#### Step 2.1: Objective Evaluation (WER/CER & Alignment)

This step uses Qwen3-ASR to evaluate textual fidelity and Qwen3-ForcedAligner to extract turn-level timestamps.

```bash
# 1. Build test audio manifest
python scripts/evaluation/objective/wer/build_test_audio_manifest.py \
    --benchmark data/en/en-data.jsonl \
    --audio-root /path/to/generated_audio_dir \
    --out-root results/objective/wer/ \
    --model ModelName

# 2. Run WER/CER evaluation & Extract Timestamps
# This generates `results.jsonl` which contains the "forced_alignment" field.
python scripts/evaluation/objective/wer/run_eval.py \
    --manifest results/objective/wer/ModelName/manifest.jsonl \
    --out results/objective/wer/ModelName/en/results.jsonl

# 3. Run Speaker Consistency evaluation
python scripts/evaluation/objective/sim/run_main_sim.py \
    --results results/objective/wer/ModelName/en/results.jsonl \
    --out results/objective/sim/ModelName/en/main_sim.jsonl
```

#### Step 2.2: Subjective Evaluation (LLM-as-a-Judge)

Set your API Key (e.g., Gemini 3.8 Flash) and run the A/B test pipeline against the ground-truth audio.

```bash
# Local Evaluation (B1-B3)
# Note: `--jsonl` uses the output from Step 2.1 which contains audio timestamps
python scripts/evaluation/lalm/run_abtest_B.py \
    --jsonl results/objective/wer/ModelName/en/results.jsonl \
    --gt_jsonl data/en/en-data.jsonl \
    --audio_dir_model /path/to/generated_wavs \
    --audio_dir_gt data/en/audios \
    --output results/subjective/local/ModelName/en/run_1.jsonl \
    --api_key "YOUR_API_KEY"

# Global Evaluation (C1-C3)
python scripts/evaluation/lalm/run_abtest_C.py \
    --jsonl data/en/en-data.jsonl \
    --gt_jsonl data/en/en-data.jsonl \
    --audio_dir_model /path/to/generated_wavs \
    --audio_dir_gt data/en/audios \
    --output results/subjective/global/ModelName/en/run_1.jsonl \
    --api_key "YOUR_API_KEY"
```

### 3. Generate the Leaderboard

Once the evaluations are complete, you can aggregate all results to automatically generate a Markdown leaderboard:

```bash
python scripts/evaluation/generate_leaderboard.py
```

## 🏆 Evaluated Models

We have comprehensively benchmarked 12 leading open-source and commercial models across 4 distinctive paradigms:

1. **Audio-Visual Models**: MiniMax-H3
2. **Context-Aware Multi-Speaker TTS**: SeedAudio, StepAudio-3-Gen
3. **Multi-Speaker Dialogue TTS**: MOSS-TTSD, SoulX-Podcast, VibeVoice, FireRedTTS2, Higgs-Audio-V2
4. **Instruction-Controllable TTS**: Qwen3-TTS-VD, VoxCPM2, FireRedTTS3, MiMo-Audio-7B

*For detailed performance metrics, generated audio samples, and comprehensive evaluation logs, please visit our Hugging Face Results Page: ASLP-lab/DRAMA-Eval-Results [<sup>5</sup>](https://huggingface.co/datasets/ASLP-lab/DRAMA-Eval-Results).*

## 📑 Citation

If you find our dataset, codebase, or evaluation pipeline useful for your research, please consider citing our work:

```bibtex
@article{drama_eval_2026,
  title={DRAMA-Eval: A Benchmark for Context-Aware, Multi-Turn, Multi-Speaker Speech Generation},
  author={Jingbin Hu, Zihang Yang, Xiaoge Peng, Haoyu Zhang, Ziyu Zhang and others},
  journal={arXiv preprint},
  year={2026}
}
```
