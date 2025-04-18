MODEL_NAME=$1
DATA_NAME=$2
DATA_TYPE=$3
DATA_VERSION=$4
PROMPT=$5
CUDA=$6

# 我觉得还是应该分开跑会比较合适
# Hallu 还是应该分到6张卡上跑

pre_prompt_template = {
    "naive_nontie": naive_nontie,
    "naive_tie": naive_tie,
    "simple_nontie": simple_nontie,
    "simple_tie": simple_tie,
    "complex_nontie": complex_nontie,
    "complex_tie": complex_tie,
}

./scripts/eval_lalm/pre.sh AF2-3B AudioCaps Hallu v2s naive_nontie 2
./scripts/eval_lalm/pre.sh AF2-3B AudioCaps Hallu v2s naive_tie 3
./scripts/eval_lalm/pre.sh AF2-3B AudioCaps Hallu v2s simple_nontie 4
./scripts/eval_lalm/pre.sh AF2-3B AudioCaps Hallu v2s simple_tie 5
./scripts/eval_lalm/pre.sh AF2-3B AudioCaps Hallu v2s complex_nontie 6 
./scripts/eval_lalm/pre.sh AF2-3B AudioCaps Hallu v2s complex_tie 7

./scripts/eval_lalm/pre.sh AF2-3B Clotho Hallu v2s naive_nontie 2
./scripts/eval_lalm/pre.sh AF2-3B Clotho Hallu v2s naive_tie 3
./scripts/eval_lalm/pre.sh AF2-3B Clotho Hallu v2s simple_nontie 4
./scripts/eval_lalm/pre.sh AF2-3B Clotho Hallu v2s simple_tie 5
./scripts/eval_lalm/pre.sh AF2-3B Clotho Hallu v2s complex_nontie 6 
./scripts/eval_lalm/pre.sh AF2-3B Clotho Hallu v2s complex_tie 7