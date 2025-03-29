import os
import sys
import json
import argparse
from vllm import LLM, SamplingParams

model_base_dir = '/mnt/public/data/lh/models'

# prompt_template = (
#     "caption_0: {}\n"
#     "caption_1: {}\n"
#     "prediction: {}\n"
#     "Based on the prediction, determine which caption is better. "
#     "If caption_0/the first caption is better, output '0'; "
#     "If caption_1/the second caption is better, output '1'; "
#     "If the prediction states that both captions are completely indistinguishable in quality, output 'tie'; "
#     "If the prediction is unrelated to determining which caption is better, output 'unknown'. "
#     "You need only output a single word of '0', '1', 'tie', or 'unknown'. "
#     "Do not add any other text or explanation. "
# )

prompt_template = """
prediction: {}  
Based on the prediction, determine which caption is better.  
Output exactly one of the following:  
- '0' if caption_0(the first caption) is better  
- '1' if caption_1(the second caption) is better  
- 'tie' if both captions are indistinguishable in quality  
- 'unknown' if the prediction is unrelated to determining which caption is better

Output only the chosen word, with no additional text or explanation.  
"""


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='Qwen2.5-14B-Instruct')
    args = parser.parse_args(args)
    return args
    

def main(args):
    args = parse_args(args)
    with open(args.result_path, 'r') as f:
        result = json.load(f)

    prompts = []
    for item in result:
        # prompt = prompt_template.format(item['caption_0'], item['caption_1'], item['output'])
        prompt = prompt_template.format(item['output'])
        prompts.append(prompt)
        print(prompt)

    prompts = prompts[:5]
    
    model_path = os.path.join(model_base_dir, args.model_name)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,      # 根据GPU数量调整 （单卡模型并非越多越好）
        gpu_memory_utilization=0.9,  # 设置GPU利用率
        max_num_seqs=256,            # 设置最大并行生成数量
    )
    
    # LLM generate sampling params
    sampling_params = SamplingParams(
        temperature=0.5,
        top_p=0.5,
        max_tokens=5
    )

    # vllm 能够自动调整 batch size 以适应 GPU 内存，所以参数中只需要设置 GPU 利用率
    # TODO: 如何使用 vllm，以及 vllm 背后的原理
    outputs = llm.generate(prompts, sampling_params)

    for item, output in zip(result, outputs):
        print('------------------')
        print(item['caption_0'], item['caption_1'])
        print('------------------')
        print(item['output'])
        print('------------------')
        print(output.outputs[0].text)
        
    return

    for item, output in zip(result, outputs):
        item['prediction'] = output.outputs[0].text

    processed_result_path = args.result_path.replace('.json', '_processed.json')
    with open(processed_result_path, 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


def test_prompt():
    print(prompt_template.format('caption_0', 'caption_1', 'prediction'))


if __name__ == '__main__':
    main(sys.argv[1:])