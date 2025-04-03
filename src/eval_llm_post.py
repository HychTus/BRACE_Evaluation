import os
import json
import logging
import argparse

from datetime import datetime
from vllm import LLM, SamplingParams
from .prompt import prompt_summary_dict

model_base_dir = '/mnt/data/lh/chy/models'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, help='Experiment name for tracking')
    parser.add_argument('--target', type=str, required=True, help='Target for evaluation')
    parser.add_argument('--task_type', type=str, required=True, choices=['pre', 'meta'], help='Task type for evaluation')
    parser.add_argument('--prompt_template_type', type=str, required=True, choices=list(prompt_summary_dict.keys()), help='Prompt template type')

    parser.add_argument('--log_base_dir', type=str, default='logs', help='Root directory for experiment logs')
    parser.add_argument('--model_name', type=str, default='Qwen2.5-14B-Instruct')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    return args


def setup_experiment(args):
    global prompt_template
    prompt_template = prompt_summary_dict[args.prompt_template_type]
    args.prompt_template = prompt_template.format(prediction='prediction')

    if args.task_type == 'pre':
        args.exp_name = args.target
        args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
        assert os.path.exists(args.log_dir), f'Log directory {args.log_dir} does not exist'

        task_name = args.target.split('-')[0]
        args.meta_path = os.path.join(args.log_dir, f'{task_name}.json')
        assert os.path.exists(args.meta_path), f'Meta file {args.meta_path} does not exist'

    else: # args.task_type == 'meta'
        if args.exp_name is None:
            date_str = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
            args.exp_name = f'post_{date_str}'
        args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
        os.makedirs(args.log_dir, exist_ok=True)
        args.meta_path = args.target

    # NOTE: 通过 basename 而不是人工的方式来获取文件名
    meta_filename = os.path.basename(args.meta_path)
    result_filename = f'processed_{meta_filename}'
    args.result_path = os.path.join(args.log_dir, result_filename)

    config_path = os.path.join(args.log_dir, 'post_config.json')
    with open(config_path, 'w') as config_file:
        json.dump(vars(args), config_file, indent=4)


def init_logging(args):
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.log_dir, 'post_log.txt')),
            logging.StreamHandler()
        ]
    )


def main():
    args = parse_args()
    setup_experiment(args)
    init_logging(args)

    logging.info(f'Experiment name: {args.exp_name}')
    logging.info(f'Model name: {args.model_name}')
    logging.info(f'Meta path: {args.meta_path}') # 便于查找文件
    logging.info(f'Result path: {args.result_path}')

    with open(args.meta_path, 'r') as f:
        result = json.load(f)

    prompts = []
    for item in result:
        # 使用 prediction 作为关键字后就不能使用位置传参数了？
        prompt = prompt_template.format(prediction=item['output'])
        item['post_prompt'] = prompt
        prompts.append(prompt)

    model_path = os.path.join(model_base_dir, args.model_name)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,      # 根据GPU数量调整 （单卡模型并非越多越好）
        gpu_memory_utilization=0.8,  # 设置GPU利用率
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
    outputs = llm.generate(prompts, sampling_params)

    # for item, output in zip(result, outputs):
    #     print('Prompt:', prompt_template.format(item['output']))
    #     print('Output:', output.outputs[0].text)
    #     print('---')
    # return

    for item, output in zip(result, outputs):
        item['prediction'] = output.outputs[0].text

    logging.info(f'Result saved to {args.result_path}')
    with open(args.result_path, 'w') as f:
        json.dump(result, f, indent=4)


def test_prompt():
    print(prompt_template.format('caption_0', 'caption_1', 'prediction'))


if __name__ == '__main__':
    main()