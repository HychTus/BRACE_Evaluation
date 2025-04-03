# Copyright (c) 2025 NVIDIA CORPORATION. 
#   Licensed under the MIT license.

import os
import yaml
import json
import argparse

import torch
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

from src.factory import create_model_and_transforms
from utils import Dict2Class, get_autocast, get_cast_dtype


def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.) # np.clip 限制 x 的范围在 [-1, 1]
    return (x * 32767.).astype(np.int16)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_num_windows(T, sr, clap_config):
    # 将 clap_config 中的参数转换为以采样点为单位
    window_length  = int(float(clap_config["window_length"]) * sr)
    window_overlap = int(float(clap_config["window_overlap"]) * sr)
    max_num_window = int(clap_config["max_num_window"])

    num_windows = 1
    if T <= window_length:
        # 如果音频长度小于等于 window_length，则只需要一个窗口
        num_windows = 1
        full_length = window_length
    elif T >= (max_num_window * window_length - (max_num_window - 1) * window_overlap):
        # 如果音频长度大于等于可能覆盖的最大长度，则使用 max_num_window
        num_windows = max_num_window
        full_length = (max_num_window * window_length - (max_num_window - 1) * window_overlap)
    else:
        # 否则计算需要的 num_windows
        num_windows = 1 + int(np.ceil((T - window_length) / float(window_length - window_overlap)))
        full_length = num_windows * window_length - (num_windows - 1) * window_overlap
    
    # num_windows 和 full_length 用于后续数据处理
    # 应该会先 padding 到 full_length 再进行分割
    return num_windows, full_length


def read_audio(file_path, target_sr, duration, start, clap_config):
    # BUG: 没有选择使用 torchaudio，而是使用 soundfile 和 pydub

    if file_path.endswith('.mp3'):
        audio = AudioSegment.from_file(file_path)
        # AudioSegment 对象单位为毫秒 ms
        if len(audio) > (start + duration) * 1000:
            audio = audio[start * 1000:(start + duration) * 1000]

        if audio.frame_rate != target_sr:
            # set_frame_rate 转换为目标采样率
            audio = audio.set_frame_rate(target_sr)

        if audio.channels > 1:
            # set_channels 转换为单声道
            audio = audio.set_channels(1)
        
        # 音频处理中幅度值存储时有 sample_width，所以要转换为浮点数范围 [-1, 1]
        # 首先转换为采样点，然后根据 sample_width 转换为 [-1, 1] 范围
        data = np.array(audio.get_array_of_samples())
        if audio.sample_width == 2:
            data = data.astype(np.float32) / np.iinfo(np.int16).max
        elif audio.sample_width == 4:
            data = data.astype(np.float32) / np.iinfo(np.int32).max
        else:
            raise ValueError("Unsupported bit depth: {}".format(audio.sample_width))

    else:
        with sf.SoundFile(file_path) as audio:
            original_sr = audio.samplerate
            channels = audio.channels

            max_frames = int((start + duration) * original_sr)

            audio.seek(int(start * original_sr)) # seek 定位文件开始读取位置
            frames_to_read = min(max_frames, len(audio))
            data = audio.read(frames_to_read)

            if data.max() > 1 or data.min() < -1:
                # 对于数据的范围进行归一化处理 BUG: 是否需要根据格式确定？
                data = data / max(abs(data.max()), abs(data.min()))
        
        if original_sr != target_sr:
            # 使用 librosa 进行 resample
            if channels == 1:
                data = librosa.resample(data.flatten(), orig_sr=original_sr, target_sr=target_sr)
            else:
                # 先进行 resample 再获取单声道
                data = librosa.resample(data.T, orig_sr=original_sr, target_sr=target_sr)[0]
        else:
            if channels != 1:
                data = data.T[0]
    
    # BUG: 分类归一化处理，对于不同的 audio 使用不同的归一化处理真的没问题？
    if data.min() >= 0:
        data = 2 * data / abs(data.max()) - 1.0
    else:
        data = data / max(abs(data.max()), abs(data.min()))
    
    # 最终返回一维的数组（类型应该是 ndarray）
    assert len(data.shape) == 1, data.shape
    return data

def load_audio(audio_path, clap_config):
    # NOTE: audio encode 全部由 CLAP 进行，所以使用 clap_config

    sr = 16000 # 使用 16kHz 采样率
    window_length  = int(float(clap_config["window_length"]) * sr)
    window_overlap = int(float(clap_config["window_overlap"]) * sr)
    max_num_window = int(clap_config["max_num_window"])
    duration = max_num_window * (clap_config["window_length"] - clap_config["window_overlap"]) + clap_config["window_overlap"]

    # 直接读取需要的最大长度 BUG: 是否进行了填充？为了便于进行 batch inference？
    # 还是说不会进行填充，只是读取到最大长度？
    audio_data = read_audio(audio_path, sr, duration, 0.0, clap_config) # hard code audio start to 0.0
    T = len(audio_data)
    num_windows, full_length = get_num_windows(T, sr, clap_config)

    # pads to the nearest multiple of window_length
    # pad 后保证能够恰好划分成对应数量的 window
    if full_length > T:
        audio_data = np.append(audio_data, np.zeros(full_length - T))

    audio_data = audio_data.reshape(1, -1)
    # 这里作用是将整数部分高于 16 位的部分进行截断
    # BUG: 防止 float32 转 int16 时溢出（怎么会担心这个？）
    # from_numpy 从 ndarray 转换为 tensor，.float() 转换为 float32
    audio_data_tensor = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float()

    audio_clips = []
    audio_embed_mask = torch.ones(num_windows)
    for i in range(num_windows):
        # 遍历并且将数据保存到 list 中，由于处理时不会修改原始数据，所以可以使用引用
        start = i * (window_length - window_overlap)
        audio_data_tensor_this = audio_data_tensor[:, start:start+window_length]
        audio_clips.append(audio_data_tensor_this)

    # BUG: 不是很明白这里的作用，限制完全没用啊
    # audio_embed_mask 可能是在 batch 的情况下使用的？
    # audio_embed_mask 初始大小就是 num_windows，限制到 max_num_window 没用啊
    if len(audio_clips) < max_num_window:
        audio_clips = audio_clips[:max_num_window]
        audio_embed_mask = audio_embed_mask[:max_num_window]

    # torch.cat 将 list 中的 tensor 进行拼接得到二维 tensor
    audio_clips = torch.cat(audio_clips)
    
    return audio_clips, audio_embed_mask

def predict(filepath, question, clap_config, inference_kwargs):

    audio_clips, audio_embed_mask = load_audio(filepath, clap_config)
    # device_id=0 为全局变量；non_blocking 非阻塞传输，使用时会自动阻塞
    # BUG: 没有使用 AMP，cast_dtype 的意义是什么？
    audio_clips = audio_clips.to(device_id, dtype=cast_dtype, non_blocking=True)
    audio_embed_mask = audio_embed_mask.to(device_id, dtype=cast_dtype, non_blocking=True)

    text_prompt = str(question).lower()

    # 开头添加 <audio> ，结尾添加 sep_token 构建完整的输入
    sample = f"<audio>{text_prompt.strip()}{tokenizer.sep_token}"

    # padding="longest" 填充到最长的长度
    text = tokenizer(
        sample,
        max_length=512,
        padding="longest",
        truncation="only_first",
        return_tensors="pt"
    )

    input_ids = text["input_ids"].to(device_id, non_blocking=True)

    prompt = input_ids

    with torch.no_grad():
        # unsqueeze(0) 还需要增加维度变成 batch 形式？
        # BUG: 不是很合理，audio_clips 本身已经是二维的了
        # 这个 mask 不会是将 text 和 audio clip 通过 id 进行对应的吧？
        output = model.generate(
            audio_x=audio_clips.unsqueeze(0),
            audio_x_mask=audio_embed_mask.unsqueeze(0),
            lang_x=prompt,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=256, # 生成的最大长度为 256
            **inference_kwargs,
            # temperature=0.0
        )[0]
    
    # 通过 sep_token 来获取生成的部分
    # 去除结束符（eos_token）、填充符（pad_token）以及其他不必要的标记 <|endofchunk|>
    output_decoded = tokenizer.decode(output).split(tokenizer.sep_token)[-1].replace(tokenizer.eos_token, '').replace(tokenizer.pad_token, '').replace('<|endofchunk|>', '')

    print('Prompt: ', question)
    print('Audio Flamingo 2: ', output_decoded)

    return output_decoded


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, help="Path to input JSON file")
    parsed_args = parser.parse_args()

    # TODO: 确认模型下载的位置，对比检查如何进行使用的
    snapshot_download(repo_id="nvidia/audio-flamingo-2", local_dir="./", token="YOUR_HF_TOKEN")

    # inference/configs/inference.yaml
    # yaml.FullLoader 解析器会将 YAML 文件中的数据转换为 Python 对象
    config = yaml.load(open("configs/inference.yaml"), Loader=yaml.FullLoader)

    data_config = config['data_config']
    model_config = config['model_config']
    clap_config = config['clap_config']
    args = Dict2Class(config['train_config'])

    # 在 factory 中根据 config 来创建完整的模型，args 来自 train_config
    model, tokenizer = create_model_and_transforms(
        **model_config,
        clap_config=clap_config, 
        use_local_files=args.offline,
        gradient_checkpointing=args.gradient_checkpointing,
        freeze_lm_embeddings=args.freeze_lm_embeddings,
    )

    device_id = 0
    model = model.to(device_id)
    model.eval()

    # TODO: 根据分析模型应该下载在 inference 中
    # inference.yaml clap_config checkpoint: clap_ckpt/epoch_15.pt
    # 使用的实际是相同的 CLAP，有点重复下载了
    # BUG: 这个模型大小真的没有问题吗？0.5B/1.5B/3B 的模型怎么可能这么大？

    # Load metadata
    with open("safe_ckpt/metadata.json", "r") as f:
        metadata = json.load(f)

    # Reconstruct the full state_dict
    state_dict = {}

    # Load each SafeTensors chunk
    for chunk_name in metadata:
        # 通过 safetensors.torch.load_file 来加载每个 chunk，update 到 state_dict 中
        chunk_path = f"safe_ckpt/{chunk_name}.safetensors"
        chunk_tensors = load_file(chunk_path)

        # Merge tensors into state_dict
        state_dict.update(chunk_tensors)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, False)

    # BUG: autocast 似乎也没有用到？
    autocast = get_autocast(
        args.precision, cache_enabled=(not args.fsdp)
    )
    cast_dtype = get_cast_dtype(args.precision)

    data = []
    with open(parsed_args.input, "r", encoding="utf-8") as file:
        # 注意是使用普通的模式打开的数据，读取每行内容，然后使用 json.loads 解析
        # json.loads 实际是将字符串转换为 Python 对象
        for line in file:
            data.append(json.loads(line.strip()))

    # inference 的参数设置
    inference_kwargs = {
        "do_sample": True,
        "top_k": 30,
        "top_p": 0.95,
        "num_return_sequences": 1
    }

    # 实际上还是逐个进行 predict 的
    for item in data:
        predict(item['path'], item['prompt'], clap_config, inference_kwargs)