#!/bin/bash

# export 才能生效，并且才能够通过 os.environ.get 在程序中获取到
# 直接设置的都是临时变量，只在当前上下文中有效
export BASE_DIR="/mnt/public/data/lh/chy"
source "${BASE_DIR}/.bashrc"

MODEL_NAME="Qwen2.5-14B-Instruct"
LOG_PATH="${BASE_DIR}/BRACE_Eval/logs"
TARGET=$1

# 在 source bashrc 后就不需要重新定义函数了
# activate vllm
set_cuda $2

python -m src.eval_lalm.post \
    --target "$TARGET" \
    --log_base_dir "$LOG_PATH" \
    --model_name "$MODEL_NAME" \
    --prompt_template_type "final_version" \