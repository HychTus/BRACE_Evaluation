#!/bin/bash
export CUDA_VISIBLE_DEVICES='3'
python -m evaluation.eval_llm \
    --meta_path '/mnt/public/data/lh/chy/evaluation/metadata/Clotho_Hallu_v2.json' \
    --meta_type 'hallucination' \
    --audio_base_path '/mnt/public/data/lh/chy/data/Brace/Hallu/Clotho/audio' \
    --model_name 'GAMA' \
    --logs '/mnt/public/data/lh/chy/evaluation/logs' \
    --single_inference \