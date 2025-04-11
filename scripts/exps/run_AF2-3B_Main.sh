MODEL_NAME=$1
DATA_NAME=$2
DATA_TYPE=$3
DATA_VERSION=$4
PROMPT=$5
CUDA=$6

./scripts/eval_lalm/pre.sh AF2-3B AudioCaps Main v2 simple_without_tie 0
./scripts/eval_lalm/pre.sh AF2-3B AudioCaps Main v2 simple_with_tie 1
./scripts/eval_lalm/pre.sh AF2-3B AudioCaps Main v2 complex_without_tie 2
./scripts/eval_lalm/pre.sh AF2-3B AudioCaps Main v2 complex_with_tie 3 

./scripts/eval_lalm/pre.sh AF2-3B Clotho Main v2 simple_without_tie 0
./scripts/eval_lalm/pre.sh AF2-3B Clotho Main v2 simple_with_tie 1
./scripts/eval_lalm/pre.sh AF2-3B Clotho Main v2 complex_without_tie 2
./scripts/eval_lalm/pre.sh AF2-3B Clotho Main v2 complex_with_tie 3