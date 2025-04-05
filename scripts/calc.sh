#!/bin/bash

LOG_PATH='/mnt/data/lh/chy/BRACE_Eval/logs'
TARGET=$1

python -m src.calc_metrics \
    --target "$TARGET" \
    --log_base_dir "${LOG_PATH}" \