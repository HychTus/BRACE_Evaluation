import os
import sys
import json
import argparse
from vllm import LLM, SamplingParams

model_base_dir = '/mnt/public/data/lh/models'

prompt_template = (
    "caption_0: {}\n"
    "caption_1: {}\n"
    "prediction: {}\n"
    "Based on the prediction, determine which caption is better."
    "If caption_0/the first caption is better, output '0';"
    "If caption_1/the second caption is better, output '1';"
    "If the prediction states that both captions are completely indistinguishable in quality, output 'tie';"
    "If the prediction is unrelated to determining which caption is better, output 'unknown'."
    "You need only output '0', '1', 'tie', or 'unknown'."
)


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='Qwen2.5-7B-Instruct')
    args = parser.parse_args(args)
    return args
    

def main(args):
    args = parse_args(args)
    with open(args.result_path, 'r') as f:
        result = json.load(f)

    prompts = [prompt_template.format(item['output']) for item in result]
    
    model_path = os.path.join(model_base_dir, args.model_name)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,      # 根据GPU数量调整 （单卡模型并非越多越好）
        gpu_memory_utilization=0.9,  # 设置GPU利用率
        max_num_seqs=256,            # 设置最大并行生成数量
    )
    
    # LLM generate sampling params
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.9,
        max_tokens=256
    )

    # vllm 似乎能够自动调整 batch size 以适应 GPU 内存 TODO
    outputs = llm.generate(prompts, sampling_params)
    for item, output in zip(result, outputs):
        item['prediction'] = output

    processed_result_path = args.result_path.replace('.json', '_processed.json')
    with open(processed_result_path, 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    print(prompt_template.format('caption_0', 'caption_1', 'prediction'))
    
    # main(sys.argv[1:])