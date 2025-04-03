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
        ./evaluation/run_eval_llm_post.sh AudioCaps_Hallu_v1s_GAMA_complex_with_tie-2025_04_01-04_56_11 0
        ./evaluation/run_eval_llm_post.sh AudioCaps_Hallu_v1s_GAMA_simple_with_tie-2025_03_31-22_36_36 0
        ./evaluation/run_eval_llm_post.sh AudioCaps_Hallu_v1s_LTU_complex_with_tie-2025_04_01-04_33_26 0
        ./evaluation/run_eval_llm_post.sh Clotho_Hallu_v1s_GAMA_complex_with_tie-2025_04_01-07_27_55 0
        ./evaluation/run_eval_llm_post.sh Clotho_Hallu_v1s_GAMA_simple_with_tie-2025_03_31-22_36_39 0
        ./evaluation/run_eval_llm_post.sh Clotho_Hallu_v1s_LTU_complex_with_tie-2025_04_01-06_05_39 0
        ;;
    1)
        ./evaluation/run_eval_llm_post.sh AudioCaps_Hallu_v1s_LTU_simple_with_tie-2025_03_31-22_36_50 1
        ./evaluation/run_eval_llm_post.sh AudioCaps_Hallu_v2s_GAMA_complex_with_tie-2025_04_01-04_57_07 1
        ./evaluation/run_eval_llm_post.sh AudioCaps_Hallu_v2s_GAMA_simple_with_tie-2025_03_31-22_36_42 1
        ./evaluation/run_eval_llm_post.sh Clotho_Hallu_v1s_LTU_simple_with_tie-2025_03_31-22_36_52 1
        ./evaluation/run_eval_llm_post.sh Clotho_Hallu_v2s_GAMA_complex_with_tie-2025_04_01-07_31_09 1
        ./evaluation/run_eval_llm_post.sh Clotho_Hallu_v2s_GAMA_simple_with_tie-2025_03_31-22_36_45 1
        ;;
    2)
        ./evaluation/run_eval_llm_post.sh AudioCaps_Hallu_v2s_LTU_complex_with_tie-2025_04_01-04_18_44 2
        ./evaluation/run_eval_llm_post.sh AudioCaps_Hallu_v2s_LTU_simple_with_tie-2025_03_31-22_36_57 2
        ./evaluation/run_eval_llm_post.sh AudioCaps_Main_v0_GAMA_complex_with_tie-2025_04_01-04_03_31 2
        ./evaluation/run_eval_llm_post.sh Clotho_Hallu_v2s_LTU_complex_with_tie-2025_04_01-05_53_00 2
        ./evaluation/run_eval_llm_post.sh Clotho_Hallu_v2s_LTU_simple_with_tie-2025_03_31-22_37_10 2
        ./evaluation/run_eval_llm_post.sh Clotho_Main_v0_GAMA_complex_with_tie-2025_04_01-04_14_13 2
        ;;
    3)
        ./evaluation/run_eval_llm_post.sh AudioCaps_Main_v0_GAMA_simple_with_tie-2025_04_01-02_24_09 3
        ./evaluation/run_eval_llm_post.sh AudioCaps_Main_v0_LTU_complex_with_tie-2025_04_01-09_37_38 3
        ./evaluation/run_eval_llm_post.sh AudioCaps_Main_v0_LTU_simple_with_tie-2025_04_01-06_25_45 3
        ./evaluation/run_eval_llm_post.sh Clotho_Main_v0_GAMA_simple_with_tie-2025_04_01-02_25_47 3
        ./evaluation/run_eval_llm_post.sh Clotho_Main_v0_LTU_complex_with_tie-2025_04_01-08_23_47 3
        ./evaluation/run_eval_llm_post.sh Clotho_Main_v0_LTU_simple_with_tie-2025_04_01-06_24_15 3
        ;;
    *)
        echo "Invalid task ID"; exit 1
        ;;
esac
