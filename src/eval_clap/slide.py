import os
import re
import pickle
import random
import torch
import numpy as np
from tqdm import tqdm

import torchaudio
import torchaudio.transforms as T

class SLIDE_CLAP:    
    def __init__(self, hop_size, window_size, clap, batch_size=256):
        self.clap = clap
        self.hop_size = hop_size
        self.window_size = window_size
        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def read_audio(self, audio_path, resample=True):
        audio_time_series, sample_rate = torchaudio.load(audio_path)
        
        resample_rate = self.clap.sampling_rate
        if resample and resample_rate != sample_rate:
            resampler = T.Resample(sample_rate, resample_rate)
            audio_time_series = resampler(audio_time_series)
        return audio_time_series, resample_rate

    def divide_audio(self, audio_path):
        full_audio, sample_rate = self.read_audio(audio_path, resample=True)
        full_audio = full_audio.reshape(-1)
        audio_clips = []

        start_index = 0
        while True:
            end_index = start_index + (int)(self.window_size*sample_rate)
            audio_clip = full_audio[start_index:end_index]
            audio_clips.append(audio_clip)
            if end_index >= full_audio.shape[0]:
                break
            start_index += (int)(self.hop_size*sample_rate)
        return full_audio, audio_clips # Tensor, List of Tensor

    def preprocess_audio(self, audio):
        sample_rate = self.clap.sampling_rate
        duration = self.clap.duration

        if duration*sample_rate >= audio.shape[0]:
            repeat_factor = int(np.ceil((duration*sample_rate) / audio.shape[0]))
            audio = audio.repeat(repeat_factor)
            audio = audio[0: duration*sample_rate]
        else:
            start_index = random.randrange(audio.shape[0] - duration*sample_rate)
            audio = audio[start_index:start_index + duration*sample_rate]
        tensor = torch.tensor(audio, dtype=torch.float32).reshape(-1)
        return tensor

    def tokenize_text(self, text):
        return self.clap.tokenize_text(text)

    def encode_audio(self, files):
        origin_files = files.copy()
        files = sorted(set(files))
        idx_map = {file: idx for idx, file in enumerate(files)}

        processed_full = []
        processed_clip = []
        for idx, file in enumerate(files):
            full, clips = self.divide_audio(file)
            clips = [self.preprocess_audio(clip) for clip in clips]
            # full shape/clip shape: Tensor(N,)
            processed_full.append(full)
            processed_clip.append(clips)

        # full: List[Tensor(N,)], clips: List[List[Tensor(N,)]]
        full_embs, _ = self.clap.get_audio_embs(processed_full, processed_clip)
        all_embs = [full_embs[idx_map[file]].reshape(-1) for file in origin_files]
        return all_embs

    def encode_text(self, texts):
        origin_texts = texts.copy()
        texts = sorted(set(texts))
        idx_map = {text: idx for idx, text in enumerate(texts)}
        
        embs = self.clap.get_text_embs(texts)
        all_embs = [embs[idx_map[text]].reshape(-1) for text in origin_texts]
        return all_embs


    def _calc_match_score(self, src_embs, dst_embs):
        src_embs = torch.stack(src_embs, dim=0).to(self.device)
        dst_embs = torch.stack(dst_embs, dim=0).to(self.device)        
        src_embs = src_embs / torch.norm(src_embs, dim=-1, keepdim=True)
        dst_embs = dst_embs / torch.norm(dst_embs, dim=-1, keepdim=True)
        sim = (src_embs * dst_embs).sum(dim=1)
        sim = torch.max(sim, torch.tensor(0.0).to(self.device)).cpu()
        return sim.tolist()

    def _score_batch(self, captions, audios):
        caption_embs = self.encode_text(captions)
        audio_embs = self.encode_audio(audios)
        results = self._calc_match_score(caption_embs, audio_embs)
        return results

    def score(self, captions, audios):
        results = [] # list of score
        for idx in tqdm(range(0, len(captions), self.batch_size), desc="Scoring captions with audios"):
            batch_captions = captions[idx: idx+self.batch_size]
            batch_audios = audios[idx: idx+self.batch_size]
            batch_results = self._score_batch(batch_captions, batch_audios)
            results.extend(batch_results)
        results = np.array(results, dtype=np.float32)
        return results


if __name__ == "__main__":
    pass
