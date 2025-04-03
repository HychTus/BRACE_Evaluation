import os
from tqdm import tqdm
import json
import pickle
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# 通过 Python 来设置程序的环境变量，可以在程序运行时动态地修改环境变量，而不需要在命令行中设置
# 还可以设置其他变量, 例如 WANDB_API_KEY
# CUDA_VISIBLE_DEVICES
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# LLaMA 3.1 8B 模型路径
# 可以直接将模型复制过来
# 8B 模型的推理需要的显存较大，需要 16G 显存?
model_path = "/cpfs/29f69eb5e2e60f26/user/sft_intern/liyouquan/huggingface_models/Meta-Llama-3.1-8B-Instruct"

# 使用 vLLM 加载模型
# tensor_parallel_size=1 表示使用单卡
# 为什么没有指定使用什么精度的量化?
llm = LLM(model=model_path, tensor_parallel_size=1)
tokenizer = llm.get_tokenizer() # tokenizer 需要单独获取

# 定义Prompt模板
prompt_template = """
Please extract the most relevant keywords from the following question-answer pairs. Focus on identifying distinct words or short phrases that represent key objects, actions, events, or locations in the video. Avoid repetition of similar meanings, and ensure that each keyword provides unique information. Output the keywords directly, separated by commas, without any additional text.

Example 1:
Conversation: 
<video>\nWhat is the man in the video doing? slurping
Key Phrases: man, slurping

Example 2:
Conversation: 
<video>\nWhere is the man located in the video? kitchen
Key Phrases: man, kitchen

Example 3:
Conversation: 
<video>\nWhat initiates the gameplay, and how do the players react at the beginning of the video? whistle
Key Phrases: gameplay, players, beginning, whistle

Now, let's extract the key phrases from the following description:
Conversation: {}
Key Phrases:
"""

# 将句子转化为适合并行处理的格式
def prepare_input_batch(sentences):
    myinput = []
    for sentence in sentences:
        # prompt_template 是一个包含占位符的字符串模板
        # 通过 format 方法将 sentence 填充到模板中
        input_text = prompt_template.format(sentence)
        # input_text = prompt_template + sentence + 'Key Phrases: '
        myinput.append([{'role': 'user', 'content': input_text}])
        # myinput 中保存多个对话的上下文
        # 每个对话的上下文为 List[Dict], Dict 中包含 role 和 content 两个 key
        # 不同的 role 的意义是什么?
    return myinput

# 批量提取关键词
def extract_keywords_batch(sentences):
    # 准备输入的批量格式
    myinput = prepare_input_batch(sentences)

    # 转换为 vLLM 可以处理的格式
    conversations = tokenizer.apply_chat_template(myinput, tokenize=False)

    # 设置采样参数，进行批量生成
    # temperature=0.2 是对于概率分布的缩放
    # p_i = exp(logit_i / temperature) / sum(exp(logit_j / temperature))
    # 当 temperature<1 时，会增加概率最大的 token 的概率，减少概率较小的 token 的概率

    # top_p=0.7 表示在累积概率大于 0.7 时停止采样
    # 只考虑概率最大的 token, 直到累积概率大于 0.7, 然后加权采样
    sampling_params = SamplingParams(
        temperature=0.2,
        top_p=0.7,
        max_tokens=512,
        stop_token_ids=[tokenizer.eos_token_id]
    )
    
    # 使用 vLLM 模型生成结果
 #   outputs = llm.generate(conversations, sampling_params)

    # 直接返回生成的文本

    keywords_list = []
    # desc 是 description 的缩写
    for output in tqdm(llm.generate(conversations, sampling_params), desc="Generating keywords"):
        generated_text = output.outputs[0].text.strip()
        cleaned_text = generated_text.replace("<|start_header_id|>assistant<|end_header_id|>", "").strip()
        # 将生成的文本中的特殊标记去掉, 并去掉首尾的空格
        keywords_list.append(cleaned_text)  # 不再拆分关键词为列表
        # cleaned_text 是什么类型? 为什么直接 append 到 keywords_list 中?
        print(cleaned_text)
    return keywords_list

# 为什么和 model 使用的路径不同? 这里怎么变成 czr 的路径了?
videochatgpt_qa_path = "/cpfs/29f69eb5e2e60f26/user/sft_intern/czr/VideoLLaMA2/datasets/videollava_sft/train_json/videochatgpt_99386_qa_pair_noisy.json"  # 请在此处填入 videochatgpt_qa 的文件路径
with open(videochatgpt_qa_path, 'r', encoding='utf-8') as f:
    videochatgpt_qa = json.load(f)

print(len(videochatgpt_qa))


# batch 的大小应该是在外部设置的