#!/bin/bash

LOG_PATH="logs"
META_PATH="BRACE/meta"
AUDIO_PATH="BRACE/audio"

MODEL_NAME=$1
DATA_NAME=$2
DATA_TYPE=$3
PROMPT=$4

python -m src.eval_lalm.pre \
    --log_base_dir "$LOG_PATH" \
    --meta_path "${META_PATH}/${DATA_NAME}_${DATA_TYPE}.json" \
    --meta_type "$DATA_TYPE" \
    --audio_base_dir "${AUDIO_PATH}/${DATA_NAME}_${DATA_TYPE}/audio" \
    --model_name "$MODEL_NAME" \
    --single_inference \
    --prompt_template_type "$PROMPT" \
    # --ref_num 1 \
    # --toy_dataset \
    # --exp_name "$EXP_NAME" \
    # --resume \
    # --debug \