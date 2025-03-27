from .model import GAMA

def create_model(model_name):
    if model_name == "GAMA":
        return GAMA(
            base_model_path='/mnt/public/data/lh/chy/GAMA/models/Llama-2-7b-chat-hf-qformer',
            eval_mdl_path='/mnt/public/data/lh/chy/GAMA/models/checkpoint-2500/pytorch_model.bin',
            GAMA_dir='/mnt/public/data/lh/chy/GAMA'
        )
    else:
        raise ValueError(f"Model {model_name} not supported!")