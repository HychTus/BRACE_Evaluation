import os
import re
import json
import yaml
import torch
import logging
import torchaudio
import numpy as np

from abc import ABC, abstractmethod
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import GenerationConfig
from peft import LoraConfig, get_peft_model


class BaseModel(ABC):
    @abstractmethod
    def inference_single(self, audio_path: str, prompt: str):
        pass


class GAMA(BaseModel):
    def __init__(self, base_model_path, eval_mdl_path, base_dir, use_fp16=False):
        from LALM.GAMA.utils.prompter import Prompter

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prompter = Prompter('alpaca_short', base_dir=base_dir)
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

        cur_audio_input = None
        audio_info = "No audio provided"
        if audio_path != 'empty':
            cur_audio_input, audio_info = self.load_audio(audio_path)
            cur_audio_input = cur_audio_input.unsqueeze(0).to(self.device)

        logging.debug(f"Audio info: {audio_info}")

        generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.01,
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

        s = generation_output.sequences[0]
        output = self.tokenizer.decode(s)[len(prompt) + 6:-4]
        return output


class LTU(BaseModel):
    def __init__(self, base_model_path, eval_mdl_path, base_dir, use_fp16=False):
        from LALM.LTU.src.ltu.utils.prompter import Prompter

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prompter = Prompter('alpaca_short', base_dir=base_dir)
        self.tokenizer = LlamaTokenizer.from_pretrained(base_model_path)
        self.use_fp16 = use_fp16

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

    def load_audio(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
        audio_info = 'Original input audio length {:.2f} seconds, number of channels: {:d}, sampling rate: {:d}.'.format(waveform.shape[1]/sample_rate, waveform.shape[0], sample_rate)
        
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

        cur_audio_input = None
        audio_info = "No audio provided"
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
            temperature=0.01,
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

        s = generation_output.sequences[0]
        output = self.tokenizer.decode(s)[len(prompt) + 6:-4]
        return output
    

class LTU_AS(BaseModel):
    def __init__(self, base_model_path, eval_mdl_path, base_dir, use_fp16=False):
        from LALM.LTU.src.ltu_as.utils.prompter import Prompter

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prompter = Prompter("alpaca_short", base_dir=base_dir)
        self.tokenizer = LlamaTokenizer.from_pretrained(base_model_path)
        self.use_fp16 = use_fp16

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

        self.whisper_text_model = self.load_whisper_text_model()
        self.whisper_feat_model = self.load_whisper_feat_model()
        self.text_cache = {}


    def load_whisper_text_model(self):
        import whisper_at
        return whisper_at.load_model("large-v2", device=self.device)

    def load_whisper_feat_model(self):
        from whisper.model import Whisper, ModelDimensions
        from ..utils import LALM_DIR

        mdl_size = 'large-v1'
        checkpoint_path = f'{LALM_DIR}/LTU/pretrained_mdls/{mdl_size}.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        dims = ModelDimensions(**checkpoint["dims"])
        whisper_feat_model = Whisper(dims)
        whisper_feat_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        whisper_feat_model.to(self.device)
        return whisper_feat_model

    def load_audio_trans(self, filename):
        import skimage.measure

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
    def __init__(self, base_model_path, eval_mdl_path, base_dir, config_path):
        from safetensors.torch import load_file
        from LALM.AF2.inference.src.factory import create_model_and_transforms
        from LALM.AF2.inference.utils import Dict2Class, get_autocast, get_cast_dtype
        
        # Load inference config
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

        # Load model and tokenizer
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
        import librosa
        import soundfile as sf
        from pydub import AudioSegment
        
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

        audio_clips, audio_embed_mask = self.load_audio(audio_path, self.clap_config)
        audio_clips = audio_clips.to(self.device, dtype=self.cast_dtype, non_blocking=True)
        audio_embed_mask = audio_embed_mask.to(self.device, dtype=self.cast_dtype, non_blocking=True)

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