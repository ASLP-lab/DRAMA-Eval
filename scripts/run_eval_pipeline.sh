#!/bin/bash
# 用法: bash run_eval_pipeline.sh <ModelName> <ModelAudioDir> <API_KEY>
# 示例: bash run_eval_pipeline.sh SeedAudio /path/to/seedaudio_wavs sk-xxxxxx

MODEL_NAME=$1
AUDIO_DIR=$2
API_KEY=$3

if [ -z "$MODEL_NAME" ] || [ -z "$AUDIO_DIR" ] || [ -z "$API_KEY" ]; then
    echo "Usage: bash run_eval_pipeline.sh <ModelName> <ModelAudioDir> <API_KEY>"
    exit 1
fi

echo "==================================================="
echo " 开始评测模型: $MODEL_NAME"
echo "==================================================="

# 基础路径配置
BASE_DIR="$(pwd)"
EN_JSONL="${BASE_DIR}/data/en/en-data.jsonl"
ZH_JSONL="${BASE_DIR}/data/zh/zh-data.jsonl"
GT_EN_AUDIO="${BASE_DIR}/data/en/audios"
GT_ZH_AUDIO="${BASE_DIR}/data/zh/audios"

# ================= 1. 客观指标 (WER & SIM) =================
for LANG in en zh; do
    echo ">>> [1/3] 运行客观指标 ($LANG) ..."
    OUT_DIR="${BASE_DIR}/results/objective/wer/${MODEL_NAME}/${LANG}"
    SIM_OUT_DIR="${BASE_DIR}/results/objective/sim/${MODEL_NAME}/${LANG}"
    mkdir -p $OUT_DIR $SIM_OUT_DIR
    
    # 假设使用对应的 JSONL
    BENCH_JSONL="${BASE_DIR}/data/${LANG}/${LANG}-data.jsonl"

    # 生成 Manifest (以 WER 代码中的工具为例)
    python scripts/evaluation/objective/wer/build_test_audio_manifest.py \
        --benchmark $BENCH_JSONL --audio-root $AUDIO_DIR --out-root $OUT_DIR --model $MODEL_NAME
    
    # 跑 WER
    python scripts/evaluation/objective/wer/run_eval.py \
        --manifest $OUT_DIR/$MODEL_NAME/manifest.jsonl \
        --out $OUT_DIR/results.jsonl

    # 跑 SIM (依赖 WER 产出的 forced_alignment)
    python scripts/evaluation/objective/sim/run_main_sim.py \
        --results $OUT_DIR/results.jsonl \
        --out $SIM_OUT_DIR/main_sim.jsonl
done

# ================= 2. 主观指标 (LLM-Judge) =================
for LANG in en zh; do
    echo ">>> [2/3] 运行主观指标 ($LANG) ..."
    BENCH_JSONL="${BASE_DIR}/data/${LANG}/${LANG}-data.jsonl"
    GT_AUDIO="${BASE_DIR}/data/${LANG}/audios"
    
    LOC_OUT="${BASE_DIR}/results/lalm/local/${MODEL_NAME}/${LANG}/run_1.jsonl"
    GLO_OUT="${BASE_DIR}/results/lalm/global/${MODEL_NAME}/${LANG}/run_1.jsonl"

    # Local 评估
    python scripts/evaluation/lalm/run_abtest_B.py \
        --jsonl $BENCH_JSONL --gt_jsonl $BENCH_JSONL \
        --audio_dir_model $AUDIO_DIR --audio_dir_gt $GT_AUDIO \
        --output $LOC_OUT --api_key $API_KEY

    # Global 评估
    python scripts/evaluation/lalm/run_abtest_C.py \
        --jsonl $BENCH_JSONL --gt_jsonl $BENCH_JSONL \
        --audio_dir_model $AUDIO_DIR --audio_dir_gt $GT_AUDIO \
        --output $GLO_OUT --api_key $API_KEY
done

# ================= 3. 更新榜单 =================
echo ">>> [3/3] 汇总结果并更新 Leaderboard ..."
python scripts/evaluation/generate_leaderboard.py

echo "评测完成！请查看 DRAMA-Eval/Leaderboard.md"
