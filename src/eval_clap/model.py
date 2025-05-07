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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' # 不考虑无 cuda

        # NOTE: CLAP 必须保证记录了 duration 和 sampling_rate（在定义 CLAP 时指定了）
        # window size 只代表裁剪的方式，不代表 CLAP 对于 audio clip 如何处理
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
        full_audio = full_audio.reshape(-1) # 双声道音频转为单声道
        audio_clips = []

        start_index = 0
        while True:
            # [start_index, end_index-1]
            end_index = start_index + (int)(self.window_size*sample_rate)
            audio_clip = full_audio[start_index:end_index]
            audio_clips.append(audio_clip)
            # slide window 中的内容刚好覆盖完全部区间，或者越界时停止
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
            # randrange 的范围为 [0, n)
            start_index = random.randrange(audio.shape[0] - duration*sample_rate)
            audio = audio[start_index:start_index + duration*sample_rate]
        tensor = torch.tensor(audio, dtype=torch.float32).reshape(-1)
        return tensor

    def tokenize_text(self, text):
        # NOTE: 统计 token 数量来考虑是否要筛去对应数据
        return self.clap.tokenize_text(text)

    def encode_audio(self, audio_paths):
        origin_audio_paths = audio_paths.copy()
        audio_paths = sorted(set(audio_paths))
        idx_map = {audio_path: idx for idx, audio_path in enumerate(audio_paths)}

        processed_full = []
        processed_clip = []
        for idx, audio_path in enumerate(audio_paths):
            # BUG: 这里不进行 process 问题也应该不大？得要看使用的 CLAP 那个层级的 API
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
        # CLAP encode 返回结果为 List[Tensor(D)]
        # torch.stack 会增加一个维度，而 torch.cat 会在对应的维度上拼接（单个叠加和 batch 拼接）
        src_embs = torch.stack(src_embs, dim=0).to(self.device)
        dst_embs = torch.stack(dst_embs, dim=0).to(self.device)        
        src_embs = src_embs / torch.norm(src_embs, dim=-1, keepdim=True)
        dst_embs = dst_embs / torch.norm(dst_embs, dim=-1, keepdim=True)

        # NOTE: 是向量点乘而不是矩阵乘法计算 similarity matrix
        sim = (src_embs * dst_embs).sum(dim=1)
        sim = torch.max(sim, torch.tensor(0.0).to(self.device)).cpu()
        return sim.tolist()

    def score(self, captions, audios):
        # NOTE: 唯一生效的 batch_size 为 self.batch_size
        # 同时对于 audio encode/text encode/calc_match_score 生效（可能会导致 GPU 被 CPU 阻塞）

        results = [] # list of score
        for idx in tqdm(range(0, len(captions), self.batch_size), desc="SLIDE CLAP scoring captions and audios"):
            batch_captions = captions[idx: idx+self.batch_size]
            batch_audios = audios[idx: idx+self.batch_size]

            caption_embs = self.encode_text(batch_captions)
            audio_embs = self.encode_audio(batch_audios)
            
            # _calc_match_score 返回的 batch_results 为 list
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
                # 如果超出了 index 会导致 refs[i] 访问错误
                idx_range.append((len(batch_refs), len(batch_refs) + len(refs[i])))
                batch_captions.extend([captions[i]] * len(refs[i])) # 需要进行复制
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
    # SIMPLE_CLAP 继承 SLIDE_CLAP，实际只需要对于 encode_audio 进行修改
    # 如何尽量少的进行修改？CLAP 这部分的逻辑我不确定有没有问题
    # 原本的 get_audio_embs_simple 处理的是 files 而不是 audio clips
    # 因为是人工进行分割然后处理，所以需要深入处理一些内部的逻辑

    def encode_audio(self, audio_paths):
        origin_audio_paths = audio_paths.copy()
        audio_paths = sorted(set(audio_paths))
        idx_map = {audio_path: idx for idx, audio_path in enumerate(audio_paths)}
        embs = self.clap.get_audio_embs_simple(audio_paths)
        all_embs = [embs[idx_map[audio_path]] for audio_path in origin_audio_paths]
        return all_embs


if __name__ == "__main__":
    pass
