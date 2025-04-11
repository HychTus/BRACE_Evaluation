#!/bin/bash

export BASE_DIR="/mnt/public/data/lh/chy"
source "${BASE_DIR}/.bashrc"

LOG_PATH="${BASE_DIR}/BRACE_Eval/logs"
META_PATH="${BASE_DIR}/BRACE_Eval/BRACE"
AUDIO_PATH="${BASE_DIR}/data/BRACE"

# audio path example: /mnt/public/data/lh/chy/data/BRACE/Hallu/Clotho/audio
# meta data example: /mnt/public/data/lh/chy/evaluation/metadata/AudioCaps_Hallu_v2.json

# ---------- Debug ----------
# EXP_NAME='test_pre'
# MODEL_NAME='AF2-3B'
# DATA_NAME='AudioCaps'
# DATA_TYPE='Main'
# DATA_VERSION='v2'
# PROMPT='simple_without_tie'
# CUDA='0'
# LOG_PATH="${BASE_DIR}/BRACE_Eval/logs/temp"

# ---------- Script ----------
MODEL_NAME=$1
DATA_NAME=$2
DATA_TYPE=$3
DATA_VERSION=$4
PROMPT=$5
CUDA=$6

# 为什么需要重新定义函数才能生效？使用的不同的 bash 进行运行？
# 直接运行的话使用的是 sh 而不是 bash，所以先前的配置不生效（但是开头应该限制了解析的方式？）
# bash 运行的话似乎就打开了新的终端，所以配置也会失效

set_cuda "$CUDA"
# activate "$MODEL_NAME"

python -m src.eval_lalm.pre \
    --log_base_dir "$LOG_PATH" \
    --meta_path "${META_PATH}/${DATA_NAME}_${DATA_TYPE}_${DATA_VERSION}.json" \
    --meta_type "$DATA_TYPE" \
    --audio_base_dir "${AUDIO_PATH}/${DATA_TYPE}/${DATA_NAME}/audio" \
    --model_name "$MODEL_NAME" \
    --single_inference \
    --prompt_template_type "$PROMPT" \

# --exp_name "$EXP_NAME" \
# --toy_dataset \
# --debug \
# --resume \

# 为什么在引用 $ 变量时只能使用双引号，不能使用单引号？
# 为什么使用 $ 时有些时候要用 ${} 而有些时候不需要？ 
# ${} 能够界定变量名边界并且支持更复杂的功能