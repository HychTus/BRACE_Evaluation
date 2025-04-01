#!/bin/bash

# AudioCaps_Hallu_v1_GAMA  0
# AudioCaps_Hallu_v1_LTU   1
# AudioCaps_Hallu_v2_GAMA  2
# AudioCaps_Hallu_v2_LTU   3

# Clotho_Hallu_v1_GAMA     0
# Clotho_Hallu_v1_LTU      1
# Clotho_Hallu_v2_GAMA     2
# Clotho_Hallu_v2_LTU      3

# AudioCaps_Main_v0_GAMA   0
# AudioCaps_Main_v0_LTU    1
# Clotho_Main_v0_GAMA      2
# Clotho_Main_v0_LTU       3

if [ $# -ne 1 ]; then
    echo "Usage: $0 <task_id>"
    echo "Task IDs:"
    echo "  0: Run all tasks labeled as 0"
    echo "  1: Run all tasks labeled as 1"
    echo "  2: Run all tasks labeled as 2"
    echo "  3: Run all tasks labeled as 3"
    exit 1
fi

task_id=$1

case $task_id in
    0)
        ./run_eval_llm_post.sh AudioCaps_Hallu_v1_GAMA 0
        ./run_eval_llm_post.sh Clotho_Hallu_v1_GAMA 0
        ./run_eval_llm_post.sh AudioCaps_Main_v0_GAMA 0
        ;;
    1)
        ./run_eval_llm_post.sh AudioCaps_Hallu_v1_LTU 1
        ./run_eval_llm_post.sh Clotho_Hallu_v1_LTU 1
        ./run_eval_llm_post.sh AudioCaps_Main_v0_LTU 1
        ;;
    2)
        ./run_eval_llm_post.sh AudioCaps_Hallu_v2_GAMA 2
        ./run_eval_llm_post.sh Clotho_Hallu_v2_GAMA 2
        ./run_eval_llm_post.sh Clotho_Main_v0_GAMA 2
        ;;
    3)
        ./run_eval_llm_post.sh AudioCaps_Hallu_v2_LTU 3
        ./run_eval_llm_post.sh Clotho_Hallu_v2_LTU 3
        ./run_eval_llm_post.sh Clotho_Main_v0_LTU 3
        ;;
    *)
        echo "Invalid task ID"; exit 1
        ;;
esac
