#!/bin/bash

LOG_PATH="logs"
META_PATH="BRACE/meta"
AUDIO_PATH="BRACE/audio"

MODEL_NAME=$1
DATA_NAME=$2
DATA_TYPE=$3

python -m src.eval_clap.eval \
    --log_base_dir "$LOG_PATH" \
    --meta_path "${META_PATH}/${DATA_NAME}_${DATA_TYPE}.json" \
    --meta_type "$DATA_TYPE" \
    --audio_base_dir "${AUDIO_PATH}/${DATA_NAME}_${DATA_TYPE}" \
    --model_name "$MODEL_NAME" \
    --ref_num 0 \
    # --toy_dataset \
    # --debug \
    # --exp_name "$EXP_NAME" \
