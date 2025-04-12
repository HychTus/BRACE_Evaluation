#!/bin/bash

export BASE_DIR="/mnt/public/data/lh/chy"
source "${BASE_DIR}/.bashrc"

LOG_PATH="${BASE_DIR}/BRACE_Eval/logs"
META_PATH="${BASE_DIR}/BRACE_Eval/BRACE"
AUDIO_PATH="${BASE_DIR}/data/BRACE"

# audio path example: /mnt/public/data/lh/chy/data/BRACE/Hallu/Clotho/audio
# meta data example: /mnt/public/data/lh/chy/evaluation/metadata/AudioCaps_Hallu_v2.json

# ---------- Debug ----------
# EXP_NAME='test_clap'
# MODEL_NAME='MS_CLAP_2023'
# DATA_NAME='Clotho'
# DATA_TYPE='Hallu'
# DATA_VERSION='v2s'
# CUDA='0'
# LOG_PATH="${BASE_DIR}/BRACE_Eval/logs"

# ---------- Script ----------
MODEL_NAME=$1
DATA_NAME=$2
DATA_TYPE=$3
DATA_VERSION=$4
CUDA=$5

set_cuda "$CUDA"
# activate "$MODEL_NAME"

python -m src.eval_clap.eval \
    --log_base_dir "$LOG_PATH" \
    --meta_path "${META_PATH}/${DATA_NAME}_${DATA_TYPE}_${DATA_VERSION}.json" \
    --meta_type "$DATA_TYPE" \
    --audio_base_dir "${AUDIO_PATH}/${DATA_TYPE}/${DATA_NAME}/audio" \
    --model_name "$MODEL_NAME" \
    # --toy_dataset \
    # --debug \
    # --exp_name "$EXP_NAME" \

# 为什么在引用 $ 变量时只能使用双引号，不能使用单引号？
# 为什么使用 $ 时有些时候要用 ${} 而有些时候不需要？ 
# ${} 能够界定变量名边界并且支持更复杂的功能