import re
import torch
import logging
import torchaudio
import numpy as np

from abc import ABC, abstractmethod
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import GenerationConfig
from peft import LoraConfig, get_peft_model

# GAMA
from LALM.GAMA.utils.prompter import Prompter as GAMA_Prompter

# LTU & LTU_AS
# TODO: 在使用 LTU 时再激活这部分依赖
# import whisper_at
# import skimage.measure
# from whisper.model import Whisper, ModelDimensions
# 通过 as 来区分相同名的 Prompter 类，注意 LTU 的路径
from LALM.LTU.src.ltu.utils.prompter import Prompter as LTU_Prompter
from LALM.LTU.src.ltu_as.utils.prompter import Prompter as LTU_AS_Prompter

# AF2
import os
import json
import yaml
import torch
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from safetensors.torch import load_file

from LALM.AF2.inference.src.factory import create_model_and_transforms
from LALM.AF2.inference.utils import Dict2Class, get_autocast, get_cast_dtype


class BaseModel(ABC):
    @abstractmethod # 必须要实现的抽象方法
    def inference_single(self, audio_path: str, prompt: str):
        pass


class GAMA(BaseModel):
    model = None
    tokenizer = None
    prompter = None
    device = None  # 将 device 也作为类属性
    
    def __init__(self, base_model_path, eval_mdl_path, base_dir, use_fp16=False):
        # 使用类属性检查模型和其他组件是否已加载
        if GAMA.model is None:
            self.load_model(
                base_model_path=base_model_path,
                eval_mdl_path=eval_mdl_path,
                base_dir=base_dir,
                use_fp16=use_fp16
            )
        else:
            # 如果模型已经加载，直接设置到实例中
            self.model = GAMA.model
            self.tokenizer = GAMA.tokenizer
            self.prompter = GAMA.prompter
            self.device = GAMA.device

    def load_model(self, base_model_path, eval_mdl_path, base_dir, use_fp16):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prompter = GAMA_Prompter('alpaca_short', base_dir=base_dir)
        self.tokenizer = LlamaTokenizer.from_pretrained(base_model_path)
        self.model = LlamaForCausalLM.from_pretrained(
            base_model_path, 
            device_map="auto",
            torch_dtype=torch.float32
        )

        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, config)
        state_dict = torch.load(eval_mdl_path, map_location='cpu')
        self.model.load_state_dict(state_dict, strict=False)

        self.model.is_parallelizable = True
        self.model.model_parallel = True

        self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2

        self.model.eval()

        # 保存为类属性，以确保所有实例共享相同的模型和组件
        GAMA.model = self.model
        GAMA.tokenizer = self.tokenizer
        GAMA.prompter = self.prompter
        GAMA.device = self.device

    def load_audio(self, filename):
        waveform, sr = torchaudio.load(filename)
        audio_info = f"Original input audio length {waveform.shape[1] / sr:.2f} seconds, number of channels: {waveform.shape[0]}, sampling rate: {sr}."

        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform=waveform, orig_freq=sr, new_freq=16000)
            sr = 16000
        waveform = waveform - waveform.mean()
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr,
                                                  use_energy=False, window_type='hanning',
                                                  num_mel_bins=128, dither=0.0, frame_shift=10)
        target_length = 1024
        n_frames = fbank.shape[0]
        p = target_length - n_frames
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]
        # normalize the fbank
        fbank = (fbank + 5.081) / 4.4849
        return fbank, audio_info

    def inference_single(self, audio_path, prompt):
        prompt = self.prompter.generate_prompt(prompt, None)
        logging.debug(f"Prompt: {prompt}")
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        # 加载音频并进行处理
        cur_audio_input = None
        audio_info = "No audio provided"
        if audio_path != 'empty':
            cur_audio_input, audio_info = self.load_audio(audio_path)
            cur_audio_input = cur_audio_input.unsqueeze(0).to(self.device)

        logging.debug(f"Audio info: {audio_info}")

        generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.1,
            top_p=0.95,
            max_new_tokens=400,
            bos_token_id=self.model.config.bos_token_id,
            eos_token_id=self.model.config.eos_token_id,
            pad_token_id=self.model.config.pad_token_id,
            num_return_sequences=1
        )

        # 生成输出
        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids.to(self.device),
                audio_input=cur_audio_input,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=400,
            )

        # 解码输出并去除<s>和</s>标记
        s = generation_output.sequences[0]
        output = self.tokenizer.decode(s)[len(prompt) + 6:-4]  # 去除 <s> 和 </s> 部分
        return output


class LTU(BaseModel):
    model = None
    tokenizer = None
    prompter = None
    device = None
    use_fp16 = None

    def __init__(self, base_model_path, eval_mdl_path, base_dir, use_fp16=False):
        if LTU.model is None:
            self.load_model(
                base_model_path=base_model_path,
                eval_mdl_path=eval_mdl_path,
                base_dir=base_dir,
                use_fp16=use_fp16
            )
        else:
            self.model = LTU.model
            self.tokenizer = LTU.tokenizer
            self.prompter = LTU.prompter
            self.device = LTU.device
            self.use_fp16 = LTU.use_fp16

    def load_model(self, base_model_path, eval_mdl_path, base_dir, use_fp16):
        # NOTE: 注意将 LTU 中使用的相对路径都进行替换
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prompter = LTU_Prompter('alpaca_short', base_dir=base_dir)
        self.tokenizer = LlamaTokenizer.from_pretrained(base_model_path)

        if self.device == "cuda":
            # LTU 使用的是 float16，而 GAMA 使用的是 float32，内存占用会更小
            self.model = LlamaForCausalLM.from_pretrained(
                base_model_path, 
                device_map="auto", 
                torch_dtype=torch.float16 if use_fp16 else torch.float32,
            )
        else:
            self.model = LlamaForCausalLM.from_pretrained(base_model_path, device_map="auto")

        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05, # LTU dropout 是 0.05，GAMA 是 0.0
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, config)
        state_dict = torch.load(eval_mdl_path, map_location='cpu')
        self.model.load_state_dict(state_dict, strict=False)

        # 先前 GAMA 遗忘了这部分代码
        self.model.is_parallelizable = True
        self.model.model_parallel = True

        self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2
        
        # self.model.to(self.device)
        # 由于在 from_pretrained 时已经设置了 device_map，所以不需要再调用 to(self.device)
        self.model.eval()

        LTU.model = self.model
        LTU.tokenizer = self.tokenizer
        LTU.prompter = self.prompter
        LTU.device = self.device
        LTU.use_fp16 = use_fp16

    def load_audio(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
        audio_info = 'Original input audio length {:.2f} seconds, number of channels: {:d}, sampling rate: {:d}.'.format(waveform.shape[1]/sample_rate, waveform.shape[0], sample_rate)
        
        # NOTE: 不同的模型的 load audio 还是有区别，所以在 class 中实现
        # 比如 GAMA 在构建 Mel Spectrogram 时使用的是双声道，LTU 使用的是单声道
        # BUG: 加上 10s 长度的限制，会产生和 CLAP 类似的问题
        if waveform.shape[0] != 1:
            waveform = waveform[0].unsqueeze(0)
            audio_info += ' Only the first channel is used.'
        
        if sample_rate == 16000:
            pass
        else:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=16000)
            sample_rate = 16000
            audio_info += ' Resample to 16000Hz.'
        waveform = waveform - waveform.mean()
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sample_rate,
                                                use_energy=False, window_type='hanning',
                                                num_mel_bins=128, dither=0.0, frame_shift=10)
        target_length = 1024
        n_frames = fbank.shape[0]
        p = target_length - n_frames
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]
        # normalize the fbank
        fbank = (fbank + 5.081) / 4.4849
        return fbank, audio_info
    
    def inference_single(self, audio_path: str, prompt: str):
        prompt = self.prompter.generate_prompt(prompt, None)
        logging.debug(f"Prompt: {prompt}")
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        # 加载音频并进行处理
        cur_audio_input = None
        audio_info = "No audio provided"
        # NOTE: 由于标注了 audio_path 为 str 类型，所以使用 empty 而不是 None
        if audio_path != "empty":
            cur_audio_input, audio_info = self.load_audio(audio_path)
            cur_audio_input = cur_audio_input.unsqueeze(0)
            if self.device == "cuda":
                if self.use_fp16:
                    cur_audio_input = cur_audio_input.half()
                cur_audio_input = cur_audio_input.to(self.device)

        logging.debug(f"Audio info: {audio_info}")

        generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.1,
            top_p=0.95,
            max_new_tokens=400,
            bos_token_id=self.model.config.bos_token_id,
            eos_token_id=self.model.config.eos_token_id,
            pad_token_id=self.model.config.pad_token_id,
            num_return_sequences=1
        )

        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids.to(self.device),
                audio_input=cur_audio_input,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=400,
            )

        # 解码输出并去除<s>和</s>标记
        s = generation_output.sequences[0]
        output = self.tokenizer.decode(s)[len(prompt) + 6:-4]  # 去除 <s> 和 </s> 部分
        return output
    

class LTU_AS(BaseModel):
    model = None
    tokenizer = None
    prompter = None
    device = None
    whisper_text_model = None
    whisper_feat_model = None
    text_cache = {}
    use_fp16 = None

    def __init__(self, base_model_path, eval_mdl_path, base_dir, use_fp16=False):
        if LTU_AS.model is None:
            self.load_model(
                base_model_path=base_model_path,
                eval_mdl_path=eval_mdl_path,
                base_dir=base_dir,
                use_fp16=use_fp16
            )
        else:
            self.model = LTU_AS.model
            self.tokenizer = LTU_AS.tokenizer
            self.prompter = LTU_AS.prompter
            self.device = LTU_AS.device
            self.whisper_text_model = LTU_AS.whisper_text_model
            self.whisper_feat_model = LTU_AS.whisper_feat_model
            self.text_cache = LTU_AS.text_cache
            self.use_fp16 = LTU_AS.use_fp16

    def load_model(self, base_model_path, eval_mdl_path, base_dir, use_fp16):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prompter = LTU_AS_Prompter("alpaca_short", base_dir=base_dir)
        self.tokenizer = LlamaTokenizer.from_pretrained(base_model_path)

        if self.device == "cuda":
            self.model = LlamaForCausalLM.from_pretrained(
                base_model_path, 
                device_map="auto", 
                torch_dtype=torch.float16 if use_fp16 else torch.float32,
            )
        else:
            self.model = LlamaForCausalLM.from_pretrained(base_model_path, device_map="auto")

        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, config)

        state_dict = torch.load(eval_mdl_path, map_location='cpu')
        self.model.load_state_dict(state_dict, strict=False)

        self.model.is_parallelizable = True
        self.model.model_parallel = True

        self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2

        self.model.eval()

        LTU_AS.model = self.model
        LTU_AS.tokenizer = self.tokenizer
        LTU_AS.prompter = self.prompter
        LTU_AS.device = self.device
        LTU_AS.whisper_text_model = self.load_whisper_text_model()
        LTU_AS.whisper_feat_model = self.load_whisper_feat_model()
        LTU_AS.text_cache = {}
        LTU_AS.use_fp16 = use_fp16

    def load_whisper_text_model(self):
        # return load_model("large-v2", device="cuda:1")
        # BUG: 这里 load 到相同的模型上是否有问题？
        return whisper_at.load_model("large-v2", device=self.device)

    def load_whisper_feat_model(self):
        # BUG: 这里使用的是绝对路径，而不是传入的路径
        mdl_size = 'large-v1'
        checkpoint_path = f'/mnt/public/data/lh/chy/LTU/pretrained_mdls/{mdl_size}.pt'
        # BUG: 这里本来的 map_location 是 'cuda:0'，改成 'cpu' 也行？
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        dims = ModelDimensions(**checkpoint["dims"])
        whisper_feat_model = Whisper(dims)
        whisper_feat_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        # BUG: 这里从 cuda:0 改为 self.device
        whisper_feat_model.to(self.device)
        return whisper_feat_model

    def load_audio_trans(self, filename):
        if filename not in self.text_cache:
            result = self.whisper_text_model.transcribe(filename)
            text = self.remove_thanks_for_watching(result["text"].lstrip())
            self.text_cache[filename] = text
        else:
            text = self.text_cache[filename]
            logging.debug('Using ASR cache')

        _, audio_feat = self.whisper_feat_model.transcribe_audio(filename)
        audio_feat = audio_feat[0]
        audio_feat = torch.permute(audio_feat, (2, 0, 1)).detach().cpu().numpy()
        audio_feat = skimage.measure.block_reduce(audio_feat, (1, 20, 1), np.mean)
        audio_feat = audio_feat[1:]  # skip the first layer
        audio_feat = torch.FloatTensor(audio_feat)
        return audio_feat, text

    def remove_thanks_for_watching(self, text):
        variations = [
            "thanks for watching", "Thanks for watching", "THANKS FOR WATCHING",
            "thanks for watching.", "Thanks for watching.", "THANKS FOR WATCHING.",
            "thanks for watching!", "Thanks for watching!", "THANKS FOR WATCHING!",
            "thank you for watching", "Thank you for watching", "THANK YOU FOR WATCHING",
            "thank you for watching.", "Thank you for watching.", "THANK YOU FOR WATCHING.",
            "thank you for watching!", "Thank you for watching!", "THANK YOU FOR WATCHING!"
        ]
        variations = sorted(variations, key=len, reverse=True)
        pattern = "|".join(re.escape(var) for var in variations)
        result = re.sub(pattern, "", text)
        return result

    def trim_string(self, a):
        separator = "### Response:\n"
        trimmed_string = a.partition(separator)[-1]
        trimmed_string = trimmed_string.strip()
        return trimmed_string

    def inference_single(self, audio_path, question):
        prompt = self.prompter.generate_prompt(question, None)
        logging.debug(f"Prompt: {prompt}")
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        cur_audio_input = None
        audio_info = "No audio provided"
        if audio_path != None:
            cur_audio_input, audio_info = self.load_audio_trans(audio_path)
            cur_audio_input = cur_audio_input.unsqueeze(0)
            if self.device == "cuda":
                if self.use_fp16:
                    cur_audio_input = cur_audio_input.half()
                cur_audio_input = cur_audio_input.to(self.device)
        logging.debug(f"Audio info: {audio_info}")

        # NOTE: temperature 竟然设得这么低，而且还设置了 top_k
        generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.1,
            top_p=0.95,
            top_k=500,
            repetition_penalty=1.1,
            max_new_tokens=500,
            bos_token_id=self.model.config.bos_token_id,
            eos_token_id=self.model.config.eos_token_id,
            pad_token_id=self.model.config.pad_token_id,
            num_return_sequences=1
        )

        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                audio_input=cur_audio_input,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=500,
            )

        s = generation_output.sequences[0]
        output = self.tokenizer.decode(s)[5: -4]
        logging.debug(f"Output: {output}")
        output = self.trim_string(output)
        logging.debug(f"Trimmed output: {output}")
        return output


class AF2(BaseModel):
    model = None
    tokenizer = None
    device = None
    clap_config = None
    autocast = None
    cast_dtype = None

    def __init__(self, base_model_path, eval_mdl_path, base_dir, config_path):
        if AF2.model is None:
            self.load_model(
                base_model_path=base_model_path,
                eval_mdl_path=eval_mdl_path,
                base_dir=base_dir,
                config_path=config_path
            )
        else:
            self.model = AF2.model
            self.tokenizer = AF2.tokenizer
            self.device = AF2.device
            self.clap_config = AF2.clap_config
            self.autocast = AF2.autocast
            self.cast_dtype = AF2.cast_dtype

    def load_model(self, base_model_path, eval_mdl_path, base_dir, config_path):
        # 1. 加载 inference 配置文件
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        data_config = config['data_config']
        model_config = config['model_config']
        clap_config = config['clap_config']
        args = Dict2Class(config['train_config'])

        model_config['lang_encoder_path'] = base_model_path
        model_config['tokenizer_path'] = base_model_path
        clap_config['checkpoint'] = os.path.join(eval_mdl_path, 'clap_ckpt', 'epoch_15.pt')
        self.clap_config = config['clap_config']

        # 2. 加载模型和 tokenizer
        model, tokenizer = create_model_and_transforms(
            **model_config,
            clap_config=clap_config, 
            use_local_files=args.offline,
            gradient_checkpointing=args.gradient_checkpointing,
            freeze_lm_embeddings=args.freeze_lm_embeddings,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(self.device)
        model.eval()

        metadata_path = os.path.join(eval_mdl_path, 'safe_ckpt', 'metadata.json')
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        state_dict = {}
        for chunk_name in metadata:
            chunk_path = os.path.join(eval_mdl_path, 'safe_ckpt', f'{chunk_name}.safetensors')
            chunk_tensors = load_file(chunk_path)
            state_dict.update(chunk_tensors)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        self.autocast = get_autocast(args.precision, cache_enabled=(not args.fsdp))
        self.cast_dtype = get_cast_dtype(args.precision)

        self.model = model
        self.tokenizer = tokenizer

        AF2.model = self.model
        AF2.tokenizer = self.tokenizer
        AF2.device = self.device
        AF2.clap_config = self.clap_config
        AF2.autocast = self.autocast
        AF2.cast_dtype = self.cast_dtype

    @staticmethod
    def int16_to_float32(x):
        return (x / 32767.0).astype(np.float32)

    @staticmethod
    def float32_to_int16(x):
        x = np.clip(x, a_min=-1., a_max=1.)
        return (x * 32767.).astype(np.int16)

    @staticmethod
    def get_num_windows(T, sr, clap_config):
        window_length  = int(float(clap_config["window_length"]) * sr)
        window_overlap = int(float(clap_config["window_overlap"]) * sr)
        max_num_window = int(clap_config["max_num_window"])

        if T <= window_length:
            num_windows = 1
            full_length = window_length
        elif T >= (max_num_window * window_length - (max_num_window - 1) * window_overlap):
            num_windows = max_num_window
            full_length = max_num_window * window_length - (max_num_window - 1) * window_overlap
        else:
            num_windows = 1 + int(np.ceil((T - window_length) / float(window_length - window_overlap)))
            full_length = num_windows * window_length - (num_windows - 1) * window_overlap
        return num_windows, full_length

    @staticmethod
    def read_audio(file_path, target_sr, duration, start, clap_config):
        if file_path.endswith('.mp3'):
            audio = AudioSegment.from_file(file_path)
            if len(audio) > (start + duration) * 1000:
                audio = audio[start * 1000:(start + duration) * 1000]
            if audio.frame_rate != target_sr:
                audio = audio.set_frame_rate(target_sr)
            if audio.channels > 1:
                audio = audio.set_channels(1)
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
                audio.seek(int(start * original_sr))
                frames_to_read = min(max_frames, len(audio))
                data = audio.read(frames_to_read)
                if data.max() > 1 or data.min() < -1:
                    data = data / max(abs(data.max()), abs(data.min()))
            if original_sr != target_sr:
                if channels == 1:
                    data = librosa.resample(data.flatten(), orig_sr=original_sr, target_sr=target_sr)
                else:
                    data = librosa.resample(data.T, orig_sr=original_sr, target_sr=target_sr)[0]
            else:
                if channels != 1:
                    data = data.T[0]
        if data.min() >= 0:
            data = 2 * data / abs(data.max()) - 1.0
        else:
            data = data / max(abs(data.max()), abs(data.min()))
        assert len(data.shape) == 1, data.shape
        return data

    def load_audio(self, file_path, clap_config):
        sr = 16000
        window_length  = int(float(clap_config["window_length"]) * sr)
        window_overlap = int(float(clap_config["window_overlap"]) * sr)
        max_num_window = int(clap_config["max_num_window"])
        duration = max_num_window * (clap_config["window_length"] - clap_config["window_overlap"]) + clap_config["window_overlap"]

        audio_data = self.read_audio(file_path, sr, duration, 0.0, clap_config)
        T = len(audio_data)
        num_windows, full_length = self.get_num_windows(T, sr, clap_config)
        if full_length > T:
            audio_data = np.append(audio_data, np.zeros(full_length - T))
        audio_data = audio_data.reshape(1, -1)
        audio_data_tensor = torch.from_numpy(self.int16_to_float32(self.float32_to_int16(audio_data))).float()

        audio_clips = []
        audio_embed_mask = torch.ones(num_windows)
        for i in range(num_windows):
            start_idx = i * (window_length - window_overlap)
            clip = audio_data_tensor[:, start_idx:start_idx+window_length]
            audio_clips.append(clip)
        if len(audio_clips) < max_num_window:
            audio_clips = audio_clips[:max_num_window]
            audio_embed_mask = audio_embed_mask[:max_num_window]
        audio_clips = torch.cat(audio_clips)
        return audio_clips, audio_embed_mask

    def inference_single(self, audio_path, prompt):
        # 处理文本 prompt
        text_prompt = str(prompt).lower()
        sample = f"<audio>{text_prompt.strip()}{self.tokenizer.sep_token}"
        text_inputs = self.tokenizer(
            sample,
            max_length=512,
            padding="longest",
            truncation="only_first",
            return_tensors="pt"
        )
        input_ids = text_inputs["input_ids"].to(self.device, non_blocking=True)

        # 加载音频并处理
        audio_clips, audio_embed_mask = self.load_audio(audio_path, self.clap_config)
        audio_clips = audio_clips.to(self.device, dtype=self.cast_dtype, non_blocking=True)
        audio_embed_mask = audio_embed_mask.to(self.device, dtype=self.cast_dtype, non_blocking=True)

        # 生成输出
        generation_kwargs = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "max_new_tokens": 256,
            "do_sample": False,
            "top_k": 30,
            "top_p": 0.95,
            "num_return_sequences": 1,
            "temperature": 0.0,
        }
        with torch.no_grad():
            output = self.model.generate(
                audio_x=audio_clips.unsqueeze(0),
                audio_x_mask=audio_embed_mask.unsqueeze(0),
                lang_x=input_ids,
                **generation_kwargs
            )[0]
        
        output_decoded = self.tokenizer.decode(output)
        output_decoded = output_decoded.split(self.tokenizer.sep_token)[-1]
        for token in [self.tokenizer.eos_token, self.tokenizer.pad_token, '<|endofchunk|>']:
            output_decoded = output_decoded.replace(token, '')
        
        logging.debug(f"Prompt: {prompt}")
        logging.debug(f"Audio Flamingo 2: {output_decoded}")
        return output_decoded