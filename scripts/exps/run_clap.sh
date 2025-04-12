MODEL_NAME=$1
DATA_NAME=$2
DATA_TYPE=$3
DATA_VERSION=$4
CUDA=$5

./scripts/eval_clap/eval.sh MS_CLAP_2023 AudioCaps Main v2
./scripts/eval_clap/eval.sh MS_CLAP_2023 Clotho Main v2

./scripts/eval_clap/eval.sh MS_CLAP_2023 AudioCaps Hallu v1s
./scripts/eval_clap/eval.sh MS_CLAP_2023 AudioCaps Hallu v2s

./scripts/eval_clap/eval.sh MS_CLAP_2023 Clotho Hallu v1s
./scripts/eval_clap/eval.sh MS_CLAP_2023 Clotho Hallu v2s


./scripts/eval_clap/eval.sh MS_CLAP_2022 AudioCaps Main v2
./scripts/eval_clap/eval.sh MS_CLAP_2022 Clotho Main v2

./scripts/eval_clap/eval.sh MS_CLAP_2022 AudioCaps Hallu v1s
./scripts/eval_clap/eval.sh MS_CLAP_2022 AudioCaps Hallu v2s

./scripts/eval_clap/eval.sh MS_CLAP_2022 Clotho Hallu v1s
./scripts/eval_clap/eval.sh MS_CLAP_2022 Clotho Hallu v2s

./scripts/eval_clap/eval.sh M2D_CLAP AudioCaps Main v2
./scripts/eval_clap/eval.sh M2D_CLAP Clotho Main v2
./scripts/eval_clap/eval.sh M2D_CLAP AudioCaps Hallu v1s
./scripts/eval_clap/eval.sh M2D_CLAP AudioCaps Hallu v2s
./scripts/eval_clap/eval.sh M2D_CLAP Clotho Hallu v1s
./scripts/eval_clap/eval.sh M2D_CLAP Clotho Hallu v2s

./scripts/eval_clap/eval.sh LAION_CLAP AudioCaps Main v2
./scripts/eval_clap/eval.sh LAION_CLAP Clotho Main v2
./scripts/eval_clap/eval.sh LAION_CLAP AudioCaps Hallu v1s
./scripts/eval_clap/eval.sh LAION_CLAP AudioCaps Hallu v2s
./scripts/eval_clap/eval.sh LAION_CLAP Clotho Hallu v1s
./scripts/eval_clap/eval.sh LAION_CLAP Clotho Hallu v2s