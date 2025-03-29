#!/bin/bash

LOG_PATH='/mnt/public/data/lh/chy/evaluation/logs'
META_PATH='/mnt/public/data/lh/chy/evaluation/metadata'
AUDIO_PATH='/mnt/public/data/lh/chy/data/Brace'
# /mnt/public/data/lh/chy/data/Brace/Hallu/Clotho/audio

activate() {
    if [ -z "$1" ]; then
        echo "Usage: activate <name>"
        return 1
    fi
    local env_path="/mnt/public/data/lh/chy/envs/$1/bin/activate"
    if [ -f "$env_path" ]; then
        source "$env_path"
    else
        echo "Error: Environment '$1' not found at $env_path"
        return 1
    fi
}

set_cuda() {
    if [ -z "$1" ]; then
        echo "Usage: set_cuda <device>"
        return 1
    fi
    export CUDA_VISIBLE_DEVICES="$1"
}

EXP_NAME=''
MODEL_NAME='LTU'
DATA_NAME='Clotho'
DATA_VERSION='v2'
DATA_TYPE='Hallu'
# /mnt/public/data/lh/chy/evaluation/metadata/AudioCaps_Hallu_v2.json

# 为什么需要重新定义函数才能生效？使用的不同的 bash 进行运行？
# 直接运行的话使用的是 sh 而不是 bash，所以先前的配置不生效（但是开头应该限制了解析的方式？）
# bash 运行的话似乎就打开了新的终端，所以配置也会失效
# export CUDA_VISIBLE_DEVICES='3'
set_cuda '3' # 注意编号是从 0~3
activate "$MODEL_NAME"

python -m evaluation.eval_llm_pre \
    --log_base_dir "${LOG_PATH}" \
    --meta_path "${META_PATH}/${DATA_NAME}_${DATA_TYPE}_${DATA_VERSION}.json" \
    --meta_type "$DATA_TYPE" \
    --audio_base_dir "${AUDIO_PATH}/$DATA_TYPE/$DATA_NAME/audio" \
    --model_name "$MODEL_NAME" \
    --single_inference \
    # --toy_dataset \
    # --debug \
    # --exp_name "" \

# 为什么只能使用双引号，不能使用单引号？