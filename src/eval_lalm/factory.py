from .model import GAMA, LTU, LTU_AS, AF2
from ..utils import MODEL_DIR, LALM_DIR


# 虽然只有一个函数，但是按照标准格式还是使用了 factory 模式
def create_model(model_name):
    if model_name == 'GAMA':
        return GAMA(
            base_model_path=f"{LALM_DIR}/GAMA/models/Llama-2-7b-chat-hf-qformer",
            eval_mdl_path=f"{LALM_DIR}/GAMA/models/checkpoint-2500/pytorch_model.bin",
            base_dir=f"{LALM_DIR}/GAMA",
            use_fp16=False,
        )
    elif model_name == 'LTU':
        return LTU(
            base_model_path=f"{LALM_DIR}/LTU/pretrained_mdls/vicuna_ltu",
            eval_mdl_path=f"{LALM_DIR}/LTU/pretrained_mdls/ltu_ori_paper.bin",
            base_dir=f"{LALM_DIR}/LTU/src/ltu",
            use_fp16=False,
        )
    elif model_name == 'LTU_AS':
        return LTU_AS(
            base_model_path=f"{LALM_DIR}/LTU/pretrained_mdls/vicuna_ltuas",
            eval_mdl_path=f"{LALM_DIR}/LTU/pretrained_mdls/ltuas_long_noqa_a6.bin",
            base_dir=f"{LALM_DIR}/LTU/src/ltu_as",
            use_fp16=False,
        )
    elif 'AF2' in model_name:
        # AF2 base model Qwen2.5 保存在 MODEL_PATH 而不是 LALM_DIR
        model_size = model_name.split('-')[1]
        return AF2(
            base_model_path=f'{MODEL_DIR}/Qwen2.5-{model_size}',
            eval_mdl_path=f'{LALM_DIR}/AF2/models/audio-flamingo-2-{model_size}',
            base_dir=f'{LALM_DIR}/AF2',
            config_path=f'{LALM_DIR}/AF2/inference/configs/inference.yaml',
        )
    else:
        raise ValueError(f"Model {model_name} not supported!")