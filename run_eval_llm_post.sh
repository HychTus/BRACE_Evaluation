#!/bin/bash

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
set_cuda 0

python process.py \
    --result_path '/mnt/public/data/lh/chy/evaluation/res/AudioCaps_Hallu_v1_GAMA.json' \
    --model_name 'Qwen2.5-14B-Instruct'