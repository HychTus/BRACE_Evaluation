#!/bin/bash

# 所以每张卡实际只分配了一个任务？容易爆显存啊？
case $1 in
    0)
        # case 0
        ./evaluation/run_eval_llm_pre.sh GAMA AudioCaps Hallu v1s simple_with_tie 0
        ./evaluation/run_eval_llm_pre.sh GAMA AudioCaps Main v0 simple_with_tie 0
        ./evaluation/run_eval_llm_pre.sh GAMA AudioCaps Main v0 complex_with_tie 0
        ;;
    1)
        # case 1
        ./evaluation/run_eval_llm_pre.sh GAMA Clotho Hallu v1s simple_with_tie 0
        ./evaluation/run_eval_llm_pre.sh GAMA AudioCaps Hallu v1s complex_with_tie 0
        ./evaluation/run_eval_llm_pre.sh GAMA Clotho Hallu v1s complex_with_tie 0
        ;;
    2)
        # case 2
        ./evaluation/run_eval_llm_pre.sh GAMA AudioCaps Hallu v2s simple_with_tie 1
        ./evaluation/run_eval_llm_pre.sh GAMA Clotho Main v0 simple_with_tie 1
        ./evaluation/run_eval_llm_pre.sh GAMA Clotho Main v0 complex_with_tie 1
        ;;
    3)
        # case 3
        ./evaluation/run_eval_llm_pre.sh GAMA Clotho Hallu v2s simple_with_tie 1
        ./evaluation/run_eval_llm_pre.sh GAMA AudioCaps Hallu v2s complex_with_tie 1
        ./evaluation/run_eval_llm_pre.sh GAMA Clotho Hallu v2s complex_with_tie 1
        ;;
    4)
        # case 4
        ./evaluation/run_eval_llm_pre.sh LTU AudioCaps Hallu v1s simple_with_tie 2
        ./evaluation/run_eval_llm_pre.sh LTU AudioCaps Main v0 simple_with_tie 2
        ./evaluation/run_eval_llm_pre.sh LTU AudioCaps Main v0 complex_with_tie 2
        ;;
    5)
        # case 5
        ./evaluation/run_eval_llm_pre.sh LTU Clotho Hallu v1s simple_with_tie 2
        ./evaluation/run_eval_llm_pre.sh LTU AudioCaps Hallu v1s complex_with_tie 2
        ./evaluation/run_eval_llm_pre.sh LTU Clotho Hallu v1s complex_with_tie 2
        ;;
    6)
        ./evaluation/run_eval_llm_pre.sh LTU AudioCaps Hallu v2s simple_with_tie 3
        ./evaluation/run_eval_llm_pre.sh LTU Clotho Main v0 simple_with_tie 3
        ./evaluation/run_eval_llm_pre.sh LTU Clotho Main v0 complex_with_tie 3
        ;;
    7)
        ./evaluation/run_eval_llm_pre.sh LTU Clotho Hallu v2s simple_with_tie 3
        ./evaluation/run_eval_llm_pre.sh LTU AudioCaps Hallu v2s complex_with_tie 3
        ./evaluation/run_eval_llm_pre.sh LTU Clotho Hallu v2s complex_with_tie 3
        ;;
    *)
        echo "Invalid input. Please provide a number between 0 and 3."
        exit 1
        ;;
esac
