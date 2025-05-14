#!/bin/bash

MODEL_NAME="Qwen/Qwen2.5-14B-Instruct"
LOG_PATH="logs"
TARGET=$1

python -m src.eval_lalm.post \
    --target "$TARGET" \
    --log_base_dir "$LOG_PATH" \
    --model_name "$MODEL_NAME" \
    --prompt_template_type "summary_v0" \