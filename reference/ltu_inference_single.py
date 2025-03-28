import os
import gradio as gr
import torch
import torchaudio
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from utils.prompter import Prompter # TODO: 这里似乎要进行 GAMA 类似的修改，相对路径和绝对路径
import datetime
import time,json

# 从代码来看只能在单卡上运行，获取 device 信息
device = "cuda" if torch.cuda.is_available() else "cpu"

# no matter which check point you use, do not change this section, this loads the llm
# 使用 prompter 模版 alpaca_short
prompter = Prompter('alpaca_short')

# 模型定义在 transformers 的 LlamaTokenizer, LlamaForCausalLM 中
tokenizer = LlamaTokenizer.from_pretrained('../../pretrained_mdls/vicuna_ltu/')
if device == 'cuda':
    # torch_dtype=torch.float16 使用半精度浮点数
    model = LlamaForCausalLM.from_pretrained('../../pretrained_mdls/vicuna_ltu/', device_map="auto", torch_dtype=torch.float16)
else:
    model = LlamaForCausalLM.from_pretrained('../../pretrained_mdls/vicuna_ltu/', device_map="auto")

# TODO: 如何使用 LoRA
# 1. 首先配置 LoraConfig
# 2. 使用 get_peft_model 和 LoraConfig 将 model 包装为可进行 LoRA 微调的模型
# 3. 再 torch.load 读取 state_dict，然后 load_state_dict

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)

# change the path to your checkpoint
state_dict = torch.load('/mnt/public/data/lh/chy/LTU/pretrained_mdls/ltu_ori_paper.bin', map_location='cpu')

# strict=False 表示不严格匹配模型结构
# TODO: LlamaForCausalLM.from_pretrained 应该没有加载完整的模型？是否有问题？
# 此时 load_state_dict 是添加了额外的 LoRA 参数？projector 的参数定义在哪里？
# 抑或者说是手动将 projector 和 LoRA 的参数保存在了 ltu_ori_paper 中？
msg = model.load_state_dict(state_dict, strict=False)

# 开启模型的并行化设置，支持大模型在多 GPU 上运行？
model.is_parallelizable = True
model.model_parallel = True

# TODO: 调整 token_id，不知道为什么要调整？
model.config.pad_token_id = tokenizer.pad_token_id = 0
model.config.bos_token_id = 1
model.config.eos_token_id = 2

model.eval()

# 保存 inference log
eval_log = []
cur_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
log_save_path = './inference_log/'
if os.path.exists(log_save_path) == False:
    os.mkdir(log_save_path)
log_save_path = log_save_path + cur_time + '.json'

# inference 中没有使用这两个参数
SAMPLE_RATE = 16000
AUDIO_LEN = 1.0

def load_audio(audio_path):
    # torchaudio.load 返回 tensor 波形信息，shape 为 [channel, time]
    waveform, sample_rate = torchaudio.load(audio_path)

    # audio_info 记录 load audio 的处理历史
    audio_info = (
        'Original input audio length {:.2f} seconds, '
        'number of channels: {:d}, sampling rate: {:d}.'.format(
            waveform.shape[1] / sample_rate, waveform.shape[0], sample_rate
        )
    )

    if waveform.shape[0] != 1:
        # 不处理多 channel 的情况
        waveform = waveform[0].unsqueeze(0) # (1, time)
        audio_info += ' Only the first channel is used.'
    if sample_rate == 16000:
        pass
    else:
        # sample_rate 需要 resample 到 16kHz
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=16000)
        sample_rate = 16000
        audio_info += ' Resample to 16000Hz.'
    
    # TODO: 使用 torchaudio 计算 fbank
    waveform = waveform - waveform.mean() # 均值处理应该也属于提取 fbank 的流程
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, 
        htk_compat=True,               # 使用 HTK (HMM Tool Kit) 兼容的梅尔滤波器设计
        sample_frequency=sample_rate,
        use_energy=False, 
        window_type='hanning',         # 使用汉宁窗减少频谱泄漏
        num_mel_bins=128,              # 生成 128 维的梅尔频谱
        dither=0.0,                    # 禁用信号预处理中的加噪
        frame_shift=10                 # 帧移为 10ms（默认帧长为 25ms）
    )

    # fbank shape: [n_frames, num_mel_bins]
    # BUG: target_length 限制为 1024，如果不够则 padding，如果超过则截断
    # 1024 对应的实际音频长度约为 10s，必然会造成信息遗失，1024 可能是为了方便 Transformer 处理？
    # LTU/GAMA 使用的是相同的处理方式，为什么只使用部分信息能够取得比 CLAP 更好的结果？长度不是本质问题？
    target_length = 1024
    n_frames = fbank.shape[0]
    p = target_length - n_frames
    if p > 0:
        # ZeroPad2d left, right, top, bottom
        # ZeroPad2d 继承自 nn.Module，实现了 forward 来进行处理
        # nn.Module 重载了 __call__ 方法来调用 forward，实现了模型函数类
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        # tensor 也能使用切片操作
        fbank = fbank[0:target_length, :]
    
    # normalize the fbank（使用的是经验性的 mean 和 std）
    fbank = (fbank + 5.081) / 4.4849
    return fbank, audio_info

def predict(audio_path, question):
    print('audio path, ', audio_path)
    begin_time = time.time()

    instruction = question
    prompt = prompter.generate_prompt(instruction, None)
    # 使用 prompter 工具生成模型需要的结构化提示模板
    # 输出中显示的是 ### Instruction: ... ### Response: TODO: 可能是训练使用的这样的数据？

    print('Input prompt: ', prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    if audio_path != None:
        # load_audio 得到的是 (n_frames, num_mel_bins) 的 fbank
        # 需要转换成 (1, n_frames, num_mel_bins) 的形式处理，unsqueeze(0) 会增加一个维度
        cur_audio_input, audio_info = load_audio(audio_path)
        cur_audio_input = cur_audio_input.unsqueeze(0)
        if torch.cuda.is_available() == False:
            pass
        else:
            # .half() 转换成半精度浮点数
            cur_audio_input = cur_audio_input.half().to(device)
    else:
        # TODO: 是否保持了 pure text 的能力？
        print('go to the none audio loop')
        cur_audio_input = None
        audio_info = 'Audio is not provided, answer pure language question.'

    # model generation_config
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.1,
        top_p=0.95,
        max_new_tokens=400,
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.pad_token_id,
        num_return_sequences=1
    )

    # model 可以使用 streaming 的模式逐个 token 生成
    # Without streaming
    with torch.no_grad():
        # 使用的是 model.generate 而不是 model.forward
        # model.generate 会调用 model.forward，但是会返回更多的信息
        generation_output = model.generate(
            input_ids=input_ids.to(device),
            audio_input=cur_audio_input,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=400,
        )
    
    # TODO: self-attention 的模型到底是如何训练的？
    # 在 input 之后预测 output，batch 中不同的样本开始可能不同，padding 的作用？
    # 解码后去除原始 prompt 和特殊符号，self-attention 机制 prompt 也在输出中
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)[len(prompt)+6:-4] # trim <s> and </s>
    end_time = time.time()
    print(output)

    # 保存 inference log
    cur_res = {'audio_id': audio_path, 'input': instruction, 'output': output}
    eval_log.append(cur_res)
    with open(log_save_path, 'w') as outfile:
        json.dump(eval_log, outfile, indent=1)
    print('eclipse time: ', end_time - begin_time, ' seconds.')
    return audio_info, output

audio_path = "/mnt/public/data/lh/chy/GAMA/sample_audio.wav"
question = "Describe the audio."
_, answer = predict(audio_path, question)
print(answer)


# link = "https://github.com/YuanGongND/ltu"
# text = "[Github]"
# paper_link = "https://arxiv.org/pdf/2305.10790.pdf"
# paper_text = "[Paper]"
# sample_audio_link = "https://drive.google.com/drive/folders/17yeBevX0LIS1ugt0DZDOoJolwxvncMja?usp=sharing"
# sample_audio_text = "[sample audios from AudioSet evaluation set]"
# demo = gr.Interface(fn=predict,
#                     inputs=[gr.Audio(type="filepath"), gr.Textbox(value='What can be inferred from the audio? Why?', label='Edit the textbox to ask your own questions!')],
#                     outputs=[gr.Textbox(label="Audio Meta Information"), gr.Textbox(label="LTU Output")],
#                     cache_examples=True,
#                     title="Quick Demo of Listen, Think, and Understand (LTU)",
#                     description="LTU is a new audio model that bridges audio perception and advanced reasoning, it can answer any open-ended question about the given audio." + f"<a href='{paper_link}'>{paper_text}</a> " + f"<a href='{link}'>{text}</a> <br>" +
#                     "LTU is authored by Yuan Gong, Hongyin Luo, Alexander H. Liu, Leonid Karlinsky, and James Glass (MIT & MIT-IBM Watson AI Lab). <br>" +
#                     "**Note LTU is not an ASR and has limited ability to recognize the speech content, it focuses on general audio perception and understanding.**<br>" +
#                     "Input an audio and ask quesions! Audio will be converted to 16kHz and padded or trim to 10 seconds. Don't have an audio sample on hand? Try some samples from AudioSet evaluation set: " +
#                     f"<a href='{sample_audio_link}'>{sample_audio_text}</a><br>" +
#                     "**Research Demo, Not for Commercial Use (Due to license of LLaMA).**")
# demo.launch(debug=False, share=True)