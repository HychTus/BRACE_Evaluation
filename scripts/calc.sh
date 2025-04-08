#!/bin/bash

export BASE_DIR="/mnt/public/data/lh/chy"
LOG_PATH="${BASE_DIR}/BRACE_Eval/logs"
TARGET=$1

python -m src.calc_metrics \
    --target "$TARGET" \
    --log_base_dir "${LOG_PATH}" \