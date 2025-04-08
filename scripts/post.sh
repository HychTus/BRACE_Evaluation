#!/bin/bash

# export 才能生效，并且才能够通过 os.environ.get 在程序中获取到
# 直接设置的都是临时变量，只在当前上下文中有效
export BASE_DIR="/mnt/public/data/lh/chy"
MODEL_NAME="Qwen2.5-14B-Instruct"
LOG_PATH="$BASE_DIR/BRACE_Eval/logs"
TARGET=$1


activate() {
    if [ -z "$1" ]; then
        echo "Usage: activate <name>"
        return 1
    fi
    local env_path="$BASE_DIR/envs/$1/bin/activate"
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

# activate vllm
set_cuda $2

# python -m evaluation.eval_llm_post \
#     --target /mnt/public/data/lh/chy/evaluation/history_res/test_process.json \
#     --task_type "meta" \
#     --log_base_dir "${LOG_PATH}" \
#     --model_name "${MODEL_NAME}" \
#     --prompt_template_type summary_answer \

python -m src.eval_llm_post \
    --target "$TARGET" \
    --log_base_dir "${LOG_PATH}" \
    --model_name "${MODEL_NAME}" \
    --prompt_template_type summary_answer \