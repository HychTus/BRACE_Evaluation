from .model import GAMA, LTU, LTU_AS

# 虽然只有一个函数，但是按照标准格式还是使用了 factory 模式
def create_model(model_name):
    if model_name == 'GAMA':
        return GAMA(
            base_model_path='/mnt/public/data/lh/chy/GAMA/models/Llama-2-7b-chat-hf-qformer',
            eval_mdl_path='/mnt/public/data/lh/chy/GAMA/models/checkpoint-2500/pytorch_model.bin',
            GAMA_dir='/mnt/public/data/lh/chy/GAMA'
        )
    elif model_name == 'LTU':
        return LTU(
            base_model_path='/mnt/public/data/lh/chy/LTU/pretrained_mdls/vicuna_ltu',
            eval_mdl_path='/mnt/public/data/lh/chy/LTU/pretrained_mdls/ltu_ori_paper.bin',
            LTU_dir='/mnt/public/data/lh/chy/LTU/src/ltu',
            use_fp16=False,
        )
    elif model_name == 'LTU-AS':
        raise NotImplementedError("LTU-AS not implemented yet!")
    else:
        raise ValueError(f"Model {model_name} not supported!")