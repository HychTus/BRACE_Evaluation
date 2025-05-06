#!/bin/bash

LOG_PATH="logs"
TARGET=$1

python -m src.eval_clap.calc \
    --target "$TARGET" \
    --log_base_dir "$LOG_PATH" \