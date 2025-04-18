#!/bin/bash
task_id=$1

if [ -z "$task_id" ]; then
    echo "Usage: $0 <task_id>"
    exit 1
fi

case $task_id in
    0)
        echo "Running task 0"
        ./scripts/eval_clap/eval.sh MS_CLAP_2023 AudioCaps Main v2 0
        ./scripts/eval_clap/eval.sh MS_CLAP_2023 Clotho Main v2 0
        ./scripts/eval_clap/eval.sh MS_CLAP_2023 AudioCaps Hallu v1s 0
        ./scripts/eval_clap/eval.sh MS_CLAP_2023 AudioCaps Hallu v2s 0
        ./scripts/eval_clap/eval.sh MS_CLAP_2023 Clotho Hallu v1s 0
        ./scripts/eval_clap/eval.sh MS_CLAP_2023 Clotho Hallu v2s 0
        ;;
    1)
        echo "Running task 1"
        ./scripts/eval_clap/eval.sh MS_CLAP_2022 AudioCaps Main v2 1
        ./scripts/eval_clap/eval.sh MS_CLAP_2022 Clotho Main v2 1
        ./scripts/eval_clap/eval.sh MS_CLAP_2022 AudioCaps Hallu v1s 1
        ./scripts/eval_clap/eval.sh MS_CLAP_2022 AudioCaps Hallu v2s 1
        ./scripts/eval_clap/eval.sh MS_CLAP_2022 Clotho Hallu v1s 1
        ./scripts/eval_clap/eval.sh MS_CLAP_2022 Clotho Hallu v2s 1
        ;;
    2)
        echo "Running task 2"
        ./scripts/eval_clap/eval.sh M2D_CLAP AudioCaps Main v2 2
        ./scripts/eval_clap/eval.sh M2D_CLAP Clotho Main v2 2
        ./scripts/eval_clap/eval.sh M2D_CLAP AudioCaps Hallu v1s 2
        ./scripts/eval_clap/eval.sh M2D_CLAP AudioCaps Hallu v2s 2
        ./scripts/eval_clap/eval.sh M2D_CLAP Clotho Hallu v1s 2
        ./scripts/eval_clap/eval.sh M2D_CLAP Clotho Hallu v2s 2
        ;;
    3)
        echo "Running task 3"
        ./scripts/eval_clap/eval.sh LAION_CLAP AudioCaps Main v2 3
        ./scripts/eval_clap/eval.sh LAION_CLAP Clotho Main v2 3
        ./scripts/eval_clap/eval.sh LAION_CLAP AudioCaps Hallu v1s 3
        ./scripts/eval_clap/eval.sh LAION_CLAP AudioCaps Hallu v2s 3
        ./scripts/eval_clap/eval.sh LAION_CLAP Clotho Hallu v1s 3
        ./scripts/eval_clap/eval.sh LAION_CLAP Clotho Hallu v2s 3
        ;;
    *)
        echo "Invalid task ID: $task_id"
        exit 1
        ;;
esac







