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

./scripts/eval_lalm/pre.sh LTU AudioCaps Main v2 all 0
./scripts/eval_lalm/pre.sh LTU Clotho Main v2 all 1

./scripts/eval_lalm/pre.sh LTU AudioCaps Hallu v2s naive_nontie_ref 2
./scripts/eval_lalm/pre.sh LTU AudioCaps Hallu v2s naive_tie_ref 3
./scripts/eval_lalm/pre.sh LTU AudioCaps Hallu v2s simple_nontie_ref 4
./scripts/eval_lalm/pre.sh LTU AudioCaps Hallu v2s simple_tie_ref 5
./scripts/eval_lalm/pre.sh LTU AudioCaps Hallu v2s complex_nontie_ref 6 
./scripts/eval_lalm/pre.sh LTU AudioCaps Hallu v2s complex_tie_ref 7

./scripts/eval_lalm/pre.sh LTU Clotho Hallu v2s naive_nontie_ref 2
./scripts/eval_lalm/pre.sh LTU Clotho Hallu v2s naive_tie_ref 3
./scripts/eval_lalm/pre.sh LTU Clotho Hallu v2s simple_nontie_ref 4
./scripts/eval_lalm/pre.sh LTU Clotho Hallu v2s simple_tie_ref 5
./scripts/eval_lalm/pre.sh LTU Clotho Hallu v2s complex_nontie_ref 6 
./scripts/eval_lalm/pre.sh LTU Clotho Hallu v2s complex_tie_ref 7


./scripts/eval_lalm/pre.sh GAMA Clotho Hallu v2s complex_tie 0 Clotho_Hallu_v2s_GAMA_complex_tie-2025_04_19-21_12_27
./scripts/eval_lalm/pre.sh GAMA Clotho Hallu v2s simple_nontie 2 Clotho_Hallu_v2s_GAMA_simple_nontie-2025_04_19-21_11_54
./scripts/eval_lalm/pre.sh GAMA Clotho Hallu v2s simple_tie 3 Clotho_Hallu_v2s_GAMA_simple_tie-2025_04_19-21_10_34

/mnt/public/data/lh/chy/BRACE_Eval/logs/Clotho_Hallu_v2s_GAMA_complex_tie-2025_04_19-21_12_27
/mnt/public/data/lh/chy/BRACE_Eval/logs/Clotho_Hallu_v2s_GAMA_simple_nontie-2025_04_19-21_11_54
/mnt/public/data/lh/chy/BRACE_Eval/logs/Clotho_Hallu_v2s_GAMA_simple_tie-2025_04_19-21_10_34

./scripts/eval_lalm/pre.sh LTU AudioCaps Main v2 all 1
./scripts/eval_lalm/pre.sh LTU Clotho Main v2 all 1

./scripts/eval_lalm/pre.sh LTU AudioCaps Hallu v2s naive_nontie 2
./scripts/eval_lalm/pre.sh LTU AudioCaps Hallu v2s naive_tie 3
./scripts/eval_lalm/pre.sh LTU AudioCaps Hallu v2s simple_nontie 4
./scripts/eval_lalm/pre.sh LTU AudioCaps Hallu v2s simple_tie 5
./scripts/eval_lalm/pre.sh LTU AudioCaps Hallu v2s complex_nontie 6 
./scripts/eval_lalm/pre.sh LTU AudioCaps Hallu v2s complex_tie 7

./scripts/eval_lalm/pre.sh LTU Clotho Hallu v2s naive_nontie 2
./scripts/eval_lalm/pre.sh LTU Clotho Hallu v2s naive_tie 3
./scripts/eval_lalm/pre.sh LTU Clotho Hallu v2s simple_nontie 4
./scripts/eval_lalm/pre.sh LTU Clotho Hallu v2s simple_tie 5
./scripts/eval_lalm/pre.sh LTU Clotho Hallu v2s complex_nontie 6 
./scripts/eval_lalm/pre.sh LTU Clotho Hallu v2s complex_tie 7