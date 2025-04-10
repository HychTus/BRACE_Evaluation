import os
import json
import logging
import argparse

from datetime import datetime
from vllm import LLM, SamplingParams
from .prompt import post_prompt_template

from ..utils import MODEL_DIR as model_base_dir

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate LLM post-pipeline')
    parser.add_argument('--exp_name', type=str, help='Experiment name for tracking')
    parser.add_argument('--target', type=str, required=True, help='Target for evaluation')
    parser.add_argument('--prompt_template_type', type=str, required=True, choices=list(post_prompt_template.keys()), help='Prompt template type')

    parser.add_argument('--log_base_dir', type=str, default='logs', help='Root directory for experiment logs')
    parser.add_argument('--model_name', type=str, default='Qwen2.5-14B-Instruct')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    args.model_base_dir = model_base_dir
    return args


def setup_experiment(args):
    # Set up logging
    if args.exp_name is None:
        date_str = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
        args.exp_name = f'post_{date_str}'

    assert os.path.exists(args.log_base_dir), f"Log base directory {args.log_base_dir} does not exist."
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    os.makedirs(args.log_dir, exist_ok=False)
    
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.log_dir, 'log.txt')),
            logging.StreamHandler()
        ]
    )

    # NOTE: 需要先初始化 logger，但是设置 FileHandler 前需要创建 log_dir
    logging.info(f'Experiment directory: {args.log_dir}')

    # Set up the prompt template
    global prompt_template
    prompt_template = post_prompt_template[args.prompt_template_type]
    args.prompt_template = prompt_template.format(
        caption_0='caption_0',
        caption_1='caption_1',
        answer='answer',
    )
    logging.info(f'Using prompt template: {args.prompt_template}')

    # Set up the target
    # path 可以通过 isfile 和 isdir 来判断是文件还是目录
    if os.path.isfile(args.target):
        args.meta_paths = [args.target]
    elif os.path.isdir(args.target):
        args.meta_paths = [os.path.join(args.target, f) for f in os.listdir(args.target) if f.endswith('.json')]
    else:
        # logging.error 并不会抛出异常，需要手动抛出异常
        logging.error(f"Target {args.target} is neither a file nor a directory")
        raise ValueError(f"Target {args.target} is neither a file nor a directory")

    # NOTE: 通过 basename 而不是人工的方式来获取文件名
    # basename 中会包含文件的后缀名
    for meta_path in args.meta_paths:
        logging.info(f'Processing file: {meta_path}')

    # Save the configuration to a JSON file
    config_path = os.path.join(args.log_dir, 'config.json')
    with open(config_path, 'w') as config_file:
        json.dump(vars(args), config_file, indent=4)


def main():
    args = parse_args()
    setup_experiment(args)

    model_path = os.path.join(model_base_dir, args.model_name)
    logging.info(f'Loading model from {model_path}')
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,      # 根据GPU数量调整 （单卡模型并非越多越好）
        gpu_memory_utilization=0.95,  # 设置GPU利用率
        max_num_seqs=256,            # 设置最大并行生成数量
    )

    # LLM generate sampling params
    # FIXME: 使用该 SamplingParams 能够基本保证输出正常（temperature 的作用？）
    # TODO: 在正常输出之后还有 Assistant: 的输出，是否可以去除？或者只取最前面的内容？
    sampling_params = SamplingParams(
        temperature=0.2,
        top_p=0.7,
        max_tokens=1
    )

    # vllm 能够自动调整 batch size 以适应 GPU 内存，所以参数中只需要设置 GPU 利用率
    # TODO: 如何使用 vllm，以及 vllm 背后的原理
    for meta_path in args.meta_paths:
        logging.info(f'Processing file: {meta_path}')
        with open(meta_path, 'r') as f:
            result = json.load(f)

        prompts = []
        for item in result:
            prompt = prompt_template.format(
                caption_0=item['caption_0'],
                caption_1=item['caption_1'],
                answer=item['output']
            )
            item['post_prompt'] = prompt
            prompts.append(prompt)

        outputs = llm.generate(prompts, sampling_params)
        for item, output in zip(result, outputs):
            item['prediction'] = output.outputs[0].text

        result_path = os.path.join(args.log_dir, os.path.basename(meta_path))
        logging.info(f'Result saved to {result_path}')
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=4)


def test_prompt():
    prompt_template = post_prompt_template['final_version']
    print(prompt_template.format(
        caption_0='caption_0', 
        caption_1='caption_1', 
        answer='answer'
    ))


def test_vllm():
    model_path = os.path.join(model_base_dir, 'Qwen2.5-14B-Instruct')
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,      # 根据GPU数量调整 （单卡模型并非越多越好）
        gpu_memory_utilization=0.95,  # 设置GPU利用率
        max_num_seqs=256,            # 设置最大并行生成数量
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=256
    )
    prompt = 'What is the capital of France?'
    output = llm.generate(prompt, sampling_params)
    print(output[0].outputs[0].text)


if __name__ == '__main__':
    # test_prompt()
    # test_vllm() # 似乎在 Muxi 测试的卡上用不了？
    main()