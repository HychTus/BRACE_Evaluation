import torch
import logging
import numpy as np

from ..utils import CLAP_DIR


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
            self.model = CLAP(version='2023', use_cuda=True)

        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.duration = self.model.args.duration
        self.sampling_rate = self.model.args.sampling_rate

    def load_audio(self, audio_path, resample=True):
        import random
        import torchaudio
        import torchaudio.transforms as T

        resample_rate = self.sampling_rate
        duration = self.duration
        audio_time_series, sample_rate = torchaudio.load(audio_path)
        
        if resample and resample_rate != sample_rate:
            resampler = T.Resample(sample_rate, resample_rate)
            audio_time_series = resampler(audio_time_series)
            sample_rate = resample_rate
        
        audio = audio_time_series.reshape(-1)

        if duration*sample_rate >= audio.shape[0]:
            repeat_factor = int(np.ceil((duration*sample_rate) / audio.shape[0]))
            audio = audio.repeat(repeat_factor)
            audio = audio[0: duration*sample_rate]
        else:
            start_index = random.randrange(audio.shape[0] - duration*sample_rate)
            audio = audio[start_index:start_index + duration*sample_rate]
        tensor = torch.tensor(audio, dtype=torch.float32).reshape(-1)
        return tensor

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
    
    def get_audio_embs_simple(self, files):
        batch_size = self.batch_size
        all_embs = []
        for i in range(0, len(files), batch_size):
            batch_files = files[i: i+batch_size]
            batch_embs = self.model.get_audio_embeddings(batch_files).cpu()
            all_embs.append(batch_embs)
        all_embs = torch.cat(all_embs, dim=0)
        return all_embs


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
        
        return input_ids[:end_index+1]


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
                    truncate=True
                ).cpu()
                all_embs.append(batch_embs)
            all_embs = torch.cat(all_embs, dim=0)
        return all_embs

    def get_audio_embs_simple(self, files):
        batch_size = self.batch_size
        all_embs = []

        tensors = [self.load_audio(file) for file in files]        
        tensors = torch.stack(tensors, dim=0)

        with torch.no_grad():
            for i in range(0, len(files), batch_size):
                batch_tensors = tensors[i: i+batch_size].to(self.device)
                batch_tensors = batch_tensors.reshape(batch_tensors.shape[0], 1, batch_tensors.shape[1])
                batch_embs = self.model.encode_clap_audio(batch_tensors).cpu()
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

    def get_audio_embs_simple(self, files):
        batch_size = self.batch_size
        
        tensors = [self.load_audio(file) for file in files]        
        tensors = torch.stack(tensors, dim=0)

        all_embs = []
        with torch.no_grad():
            for i in range(0, len(files), batch_size):
                batch_tensors = tensors[i: i+batch_size]
                batch_tensors = torch.from_numpy(self.int16_to_float32(self.float32_to_int16(batch_tensors.numpy()))).float()
                batch_tensors = batch_tensors.to(self.device)
                batch_embs = self.model.get_audio_embedding_from_data(x=batch_tensors, use_tensor=True)
                all_embs.append(torch.tensor(batch_embs).float())
            all_embs = torch.cat(all_embs, dim=0)
        return all_embs


if __name__ == "__main__":
    pass