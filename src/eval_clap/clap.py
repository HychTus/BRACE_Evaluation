import torch
import logging
import numpy as np

from ..utils import CLAP_DIR


# NOTE: CLAP 内部分别实现 get_emb 和 get_emb_batch，由于外部控制了，这里不用控制
# 既然已经写了我也不好更改，还是加一个 batch_size 参数吧，可以在 factory.py/create_model 里设置

# NOTE: 不同的 CLAP backend 内部使用的 encode 函数不同，所以需要单独重写处理
# NOTE: 部分 CLAP backend 没有设置 no_grad，所以需要手动设置


class BASE_CLAP():
    def __init__(self):
        pass

    def get_text_embs(self, texts):
        raise NotImplementedError
    
    def get_audio_embs(self, audio_full, audio_clip):
        raise NotImplementedError


class MS_CLAP_2023(BASE_CLAP):
    def __init__(self, batch_size=128, use_local_model=False):
        if use_local_model:
            from msclap import CLAP
            model_fp = f"{CLAP_DIR}/MS_CLAP/CLAP_weights_2023.pth"
            self.model = CLAP(model_fp=model_fp, version='2023', use_cuda=True)
        else:
            from msclap import CLAP
            self.model = CLAP(version='2023', use_cuda=True) # 都使用 cuda

        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.duration = self.model.args.duration
        self.sampling_rate = self.model.args.sampling_rate
        
    def _tokenize_text(self, text):
        if 'gpt' in self.model.args.text_model:
            text = text + ' <|endoftext|>' 
        tok = self.model.tokenizer.encode_plus(
            text=text, 
            add_special_tokens=True, 
            max_length=self.model.args.text_len, 
            padding='max_length', 
            return_tensors="pt",
            truncation=True
        )
        return tok

    def tokenize_text(self, text):
        # tokenize_text 还被用于计算 tf-idf
        tokens = self._tokenize_text(text)['input_ids'].reshape(-1).tolist()
        end_token = max(tokens)
        end_token_index = tokens.index(end_token)
        return tokens[:end_token_index]

    def get_text_embs(self, texts):
        batch_size = self.batch_size
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch_embs = self.model.get_text_embeddings(texts[i: i+batch_size]).cpu()
            all_embs.append(batch_embs)
        all_embs = torch.cat(all_embs, dim=0)
        return all_embs

    def get_audio_embs(self, audio_full, audio_clip):
        # audio_clip List[List[torch.Tensor(N, )]]
        batch_size = self.batch_size
        tensors = []
        idx_range = []
        for t in audio_clip:
            idx_range.append((len(tensors), len(tensors)+len(t)))
            tensors.extend(t)

        tensors = torch.stack(tensors, dim=0) 
        all_embs = []
        for i in range(0, len(tensors), batch_size):
            batch_tensors = tensors[i: i+batch_size].to(self.device)
            batch_tensors = batch_tensors.reshape(batch_tensors.shape[0], 1, batch_tensors.shape[1])
            batch_embs = self.model._get_audio_embeddings(batch_tensors).cpu()
            all_embs.append(batch_embs)
        all_embs = torch.cat(all_embs, dim=0)

        full_embs = []
        clip_embs = []
        for start_idx, end_idx in idx_range:
            clip_embs.append(all_embs[start_idx: end_idx])
            global_emb = torch.mean(clip_embs[-1], dim=0, keepdim=False)
            full_embs.append(global_emb)
        
        return full_embs, clip_embs


class MS_CLAP_2022(MS_CLAP_2023):
    def __init__(self, batch_size=128, use_local_model=True):
        if use_local_model:
            from msclap import CLAP
            model_fp = f"{CLAP_DIR}/MS_CLAP/CLAP_weights_2022.pth"
            self.model = CLAP(model_fp=model_fp, version='2022', use_cuda=True)
        else:
            from msclap import CLAP
            self.model = CLAP(version='2022', use_cuda=True)

        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.duration = self.model.args.duration
        self.sampling_rate = self.model.args.sampling_rate
    
    def tokenize_text(self, text):
        tok = self._tokenize_text(text)
        input_ids = tok['input_ids'].reshape(-1).tolist()
        attention_mask = tok['attention_mask'].reshape(-1).tolist()
        
        end_index = len(attention_mask) - 1
        while end_index >= 0 and attention_mask[end_index] == 0:
            end_index -= 1
        
        return input_ids[:end_index+1] # 此时要包含所有内容


class M2D_CLAP(MS_CLAP_2022):
    def __init__(self, batch_size=128, use_local_model=True):
        if use_local_model:
            from CLAP.M2D_CLAP.m2d.examples.portable_m2d import PortableM2D
            model_fp = f"{CLAP_DIR}/M2D_CLAP/m2d_clap_vit_base-80x608p16x16-240128/checkpoint-300.pth"
            model = PortableM2D(weight_file=model_fp, flat_features=True)
        else:
            raise NotImplementedError("M2D model is not available for now.")

        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.duration = 10 
        self.sampling_rate = 16000

    def _tokenize_text(self, text):
        max_length = 512
        tok = self.model.text_encoder.tokenizer(
            text, 
            max_length=max_length, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        )
        return tok

    def get_audio_embs(self, audio_full, audio_clip):
        # audio_clip List[List[torch.Tensor(N, )]]
        batch_size = self.batch_size
        tensors = []
        idx_range = []
        for t in audio_clip:
            idx_range.append((len(tensors), len(tensors)+len(t)))
            tensors.extend(t)

        tensors = torch.stack(tensors, dim=0)
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(tensors), batch_size):
                batch_tensors = tensors[i: i+batch_size].to(self.device)
                batch_tensors = batch_tensors.reshape(batch_tensors.shape[0], 1, batch_tensors.shape[1])
                batch_embs = self.model.encode_clap_audio(batch_tensors).cpu()
                all_embs.append(batch_embs)
        all_embs = torch.cat(all_embs, dim=0)

        full_embs = []
        clip_embs = []
        for start_idx, end_idx in idx_range:
            clip_embs.append(all_embs[start_idx: end_idx])
            global_emb = torch.mean(clip_embs[-1], dim=0, keepdim=False)
            full_embs.append(global_emb)
        
        return full_embs, clip_embs

    def get_text_embs(self, texts):
        batch_size = self.batch_size
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_embs = self.model.encode_clap_text(
                    texts[i: i+batch_size],
                    truncate=True # 需要进行截断
                ).cpu()
                all_embs.append(batch_embs)
            all_embs = torch.cat(all_embs, dim=0)
        return all_embs


class LAION_CLAP(M2D_CLAP):
    def __init__(self, batch_size=128):
        import laion_clap
        model = laion_clap.CLAP_Module(enable_fusion=False)
        model.load_ckpt()

        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.duration = 10
        self.sampling_rate = 48000

    def int16_to_float32(self, x):
        return (x / 32767.0).astype(np.float32)
    
    def float32_to_int16(self, x):
        x = np.clip(x, a_min=-1., a_max=1.)
        return (x * 32767.).astype(np.int16)

    def get_audio_embs(self, audio_full, audio_clip):
        # audio_clip List[List[torch.Tensor(N, )]]
        batch_size = self.batch_size
        tensors = []
        idx_range = []
        for t in audio_clip:
            idx_range.append((len(tensors), len(tensors)+len(t)))
            tensors.extend(t)

        tensors = torch.stack(tensors, dim=0)
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(tensors), batch_size):
                batch_tensors = tensors[i: i+batch_size]
                batch_tensors = torch.from_numpy(self.int16_to_float32(self.float32_to_int16(batch_tensors.numpy()))).float()
                batch_tensors = batch_tensors.to(self.device)
                batch_embs = self.model.get_audio_embedding_from_data(x=batch_tensors, use_tensor=True)
                all_embs.append(torch.tensor(batch_embs).float())
        all_embs = torch.cat(all_embs, dim=0)

        full_embs = []
        clip_embs = []
        for start_idx, end_idx in idx_range:
            clip_embs.append(all_embs[start_idx: end_idx])
            global_emb = torch.mean(clip_embs[-1], dim=0, keepdim=False)
            full_embs.append(global_emb)
        
        return full_embs, clip_embs
    
    def get_text_embs(self, texts):
        batch_size = self.batch_size
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_embs = self.model.get_text_embedding(texts[i: i+batch_size])
                all_embs.append(torch.tensor(batch_embs).float())
            all_embs = torch.cat(all_embs, dim=0)
        return all_embs


class AF2_CLAP(BASE_CLAP):
    def __init__(
            self,
            batch_size=128,
            use_local_model=False,
            use_cuda=True
        ):
        from LALM.AF2.inference.src.factory import CLAP
        model = CLAP(
            clap_config = {
                'method': 'nvclap-large',
                'audio_embed_dim': 2048,
                'checkpoint': 'clap_ckpt/epoch_15.pt',
                'window_length': 10.0,  # seconds
                'window_overlap': 0.0,  # seconds
                'max_num_window': 9,  # 1.5 minutes
                'max_num_fewshot': 1,  # number of fewshot samples (including the final one)
                'finetune': True,
            }
        )
        model.load_ckpt()

        self.batch_size = batch_size
        self.device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.duration = 10
        self.sampling_rate = 48000



if __name__ == "__main__":
    pass