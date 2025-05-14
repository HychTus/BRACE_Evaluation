import random
import torch
import numpy as np
from tqdm import tqdm

import torchaudio
import torchaudio.transforms as T

class SLIDE_CLAP:    
    def __init__(self, hop_size, window_size, clap, batch_size=512):
        self.clap = clap
        self.hop_size = hop_size
        self.window_size = window_size
        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        assert hasattr(self.clap, 'sampling_rate'), "CLAP object must have a 'sampling_rate' attribute."
        assert hasattr(self.clap, 'duration'), "CLAP object must have a 'duration' attribute."

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
            # [start_index, end_index-1]
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

    def encode_audio(self, audio_paths):
        origin_audio_paths = audio_paths.copy()
        audio_paths = sorted(set(audio_paths))
        idx_map = {audio_path: idx for idx, audio_path in enumerate(audio_paths)}

        processed_full = []
        processed_clip = []
        for idx, audio_path in enumerate(audio_paths):
            full, clips = self.divide_audio(audio_path)
            clips = [self.preprocess_audio(clip) for clip in clips]
            # full shape/clip shape: Tensor(N,)
            processed_full.append(full)
            processed_clip.append(clips)

        # full: List[Tensor(N,)], clips: List[List[Tensor(N,)]]
        full_embs, _ = self.clap.get_audio_embs(processed_full, processed_clip)
        all_embs = [full_embs[idx_map[audio_path]].reshape(-1) for audio_path in origin_audio_paths]
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

    def score(self, captions, audios):
        results = [] # list of score
        for idx in tqdm(range(0, len(captions), self.batch_size), desc="SLIDE CLAP scoring captions and audios"):
            batch_captions = captions[idx: idx+self.batch_size]
            batch_audios = audios[idx: idx+self.batch_size]

            caption_embs = self.encode_text(batch_captions)
            audio_embs = self.encode_audio(batch_audios)
            
            batch_results = self._calc_match_score(caption_embs, audio_embs)
            results.extend(batch_results)
        results = np.array(results, dtype=np.float32)
        return results

    def score_ref(self, captions, refs):
        results = []
        max_len = max([len(ref) for ref in refs])
        for idx in tqdm(range(0, len(captions), self.batch_size), desc="SLIDE CLAP scoring captions and refs"):
            idx_range = []
            batch_captions, batch_refs = [], []
            for i in range(idx, min(idx+self.batch_size, len(captions))):
                idx_range.append((len(batch_refs), len(batch_refs) + len(refs[i])))
                batch_captions.extend([captions[i]] * len(refs[i]))
                batch_refs.extend(refs[i])
                
            caption_embs = self.encode_text(batch_captions)
            ref_embs = self.encode_text(batch_refs)
            flatten_results = self._calc_match_score(caption_embs, ref_embs)

            for start, end in idx_range:
                curr_result = flatten_results[start:end]
                curr_result += [np.nan] * (max_len - len(curr_result))
                results.append(curr_result)
        results = np.array(results, dtype=np.float32)
        return results


class SIMPLE_CLAP(SLIDE_CLAP):
    def encode_audio(self, audio_paths):
        origin_audio_paths = audio_paths.copy()
        audio_paths = sorted(set(audio_paths))
        idx_map = {audio_path: idx for idx, audio_path in enumerate(audio_paths)}
        embs = self.clap.get_audio_embs_simple(audio_paths)
        all_embs = [embs[idx_map[audio_path]] for audio_path in origin_audio_paths]
        return all_embs


if __name__ == "__main__":
    pass
