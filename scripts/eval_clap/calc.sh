#!/bin/bash

export BASE_DIR="/mnt/public/data/lh/chy"
source "${BASE_DIR}/.bashrc"

LOG_PATH="${BASE_DIR}/BRACE_Eval/logs"
TARGET=$1
# TARGTE="" # for test

python -m src.eval_clap.calc \
    --target "$TARGET" \
    --log_base_dir "$LOG_PATH" \