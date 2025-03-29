import os
import sys
import time
import json
import logging
import argparse
import subprocess
import numpy as np

from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import f1_score

from .data import BRACE_Dataset
from .infer import inference_single
# from .process import main as process_main 导入也会出现影响


def parse_args():
    # TODO: 使用 sys.argv[1:] 相比直接使用 parser 的优势？能够在代码中手动传参来模拟？
    parser = argparse.ArgumentParser(description='Evaluate LLM pre-pipeline')

    # 注意哪些参数是必须的，哪些是可选的
    # NOTE: 在名称上使用论文中的 Hallu 和 Main
    parser.add_argument('--exp_name', type=str, help='Experiment name for tracking')
    parser.add_argument('--log_base_dir', type=str, default='logs', help='Root directory for experiment logs')
    parser.add_argument('--meta_path', type=str, required=True, help='Path to metadata file')
    parser.add_argument('--meta_type', type=str, required=True, choices=['Main', 'Hallu'], help='Metadata format type')
    parser.add_argument('--audio_base_dir', type=str, required=True, help='Base directory for audio files')
    parser.add_argument('--model_name', type=str, required=True, help='Model name for inference')
    parser.add_argument('--ref_num', type=int, default=0, help='Number of reference samples')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--single_inference', action='store_true', help='Enable single inference mode')
    parser.add_argument('--toy_dataset', action='store_true', help='Use toy dataset for testing')

    return parser.parse_args()


def init_logging(args):
    # log 分别输出到文件和控制台
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.log_dir, 'log.txt')),
            logging.StreamHandler()
        ]
    )


def setup_experiment(args):
    args.dataset_name = args.meta_path.split('/')[-1].split('.')[0]
    args.task_name = f'{args.dataset_name}_{args.model_name}'

    if args.exp_name is None:
        date_str = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
        # task 在前便于进行查找，date_str 作为附加信息进行区分
        args.exp_name = '-'.join([
            f'task_{args.task_name}',
            date_str,
        ])
    # 参数传递的是 log_base_dir，实验对应的目录为 log_dir
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    os.makedirs(args.log_dir, exist_ok=True)

    # Save experiment configuration to config.json
    config_path = os.path.join(args.log_dir, 'config.json')
    with open(config_path, 'w') as config_file:
        json.dump(vars(args), config_file, indent=4)


def generate_prompt(item, ref_num=0):
    # TODO: 暂时先不测试 ref，我记得有只使用 ref 和 caption 的 evaluation method
    if ref_num > 0:
        raise NotImplementedError

    # TODO: 是否要求模型只能输出 caption_0 或者 caption_1？ 
    prompt = (
        "Here are two captions describing the audio content:\n"
        "caption_0: {caption_0}\n"
        "caption_1: {caption_1}\n"
        "Which caption better matches the audio content?"
        # "You only need to output caption_0 or caption_1."
        # "You only need to output '0' or '1' to indicate which caption better matches the audio content.\n"
        # "You don't need to output any other content."
    )

    return prompt.format(
        caption_0=item['caption_0'],
        caption_1=item['caption_1'],
    )


def main():
    start_time = time.time()
    args = parse_args()
    setup_experiment(args)
    init_logging(args)
    
    # FIXME: 由于在 setup_experiment 中调用了 logging，所以没有进行 basicConfig，导致 info 和 debug 都无法输出
    # NOTE: VSCode 输出路径时候可以点击查看是否存在，比较方便
    logging.info(f'Task: {args.task_name}')
    logging.info(f'Experiment name: {args.exp_name}')
    logging.debug(f'Arguments: {args}')

    # create dataset
    # NOTE: 统一文件使用 path，目录使用 dir
    dataset = BRACE_Dataset(
        meta_path=args.meta_path,
        meta_type=args.meta_type,
        audio_base_dir=args.audio_base_dir
    )
    if args.toy_dataset:
        dataset = dataset[:10]

    logging.info(f'Dataset number: {len(dataset)}')

    # inference
    # TODO: 多次尝试是否有区别？对于前后顺序影响的考虑
    logging.info(f'Start inference on {args.dataset_name} with {args.model_name}')
    result = []
    if args.single_inference:
        # single inference
        for item in tqdm(dataset, desc="Processing items"):
            audio_path = item['audio_path']
            prompt = generate_prompt(item, ref_num=args.ref_num)
            output = inference_single(
                audio_path=audio_path, 
                prompt=prompt,
                model_name=args.model_name,
            )
            item['prompt'] = prompt
            item['output'] = output
            result.append(item)
    else:
        # batch inference
        raise NotImplementedError

    result_path = os.path.join(args.log_dir, 'result.json')
    logging.info(f'Result saved to {result_path}')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=4)

    end_time = time.time()
    logging.info(f'Time cost: {end_time - start_time:.2f}s')


if __name__ == '__main__':
    main()