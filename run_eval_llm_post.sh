#!/bin/bash

MODEL_NAME='Qwen2.5-14B-Instruct'
META_PATH='/mnt/public/data/lh/chy/evaluation/res'
LOG_PATH='/mnt/public/data/lh/chy/evaluation/logs'
META_NAME=$1


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

activate vllm
set_cuda $2

python eval_llm_post.py \
    --meta_path "${META_PATH}/${META_NAME}.json" \
    --model_name "${MODEL_NAME}" \
    --log_base_dir "${LOG_PATH}" \