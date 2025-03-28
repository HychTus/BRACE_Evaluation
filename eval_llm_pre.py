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
from .process import main as process_main

def parse_args():
    # TODO: 使用 sys.argv[1:] 相比直接使用 parser 的优势？能够在代码中手动传参来模拟？
    # NOTE: 由于多阶段调用相同的代码，使用的参数不同，导致不能设置 required=True
    parser = argparse.ArgumentParser(description='Multistage evaluation pipeline')

    # 阶段控制参数
    parser.add_argument('--stage', type=str, default='all',
                        choices=['all', 'inference', 'process', 'metric'],
                        help='Pipeline stages to execute: all|inference|process|metric')
    parser.add_argument('--exp_name', type=str, required=True,
                        help='Experiment name for tracking')
    parser.add_argument('--logs_dir', type=str, default='logs',
                        help='Root directory for experiment logs')
    
    # Inference 阶段参数
    parser.add_argument('--meta_path', type=str, help='Path to metadata file')
    parser.add_argument('--meta_type', type=str, help='Metadata format type')
    parser.add_argument('--audio_base_dir', type=str, 
                        help='Base directory for audio files')
    parser.add_argument('--model', type=str, 
                        help='Model name for inference')
    parser.add_argument('--ref_num', type=int, default=0,
                        help='Number of reference samples')
    
    # Debug 参数
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    return parser.parse_args()


def init_logging(log_dir: str, debug: bool = False) -> None:
    """配置日志系统"""
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Path(log_dir)/'pipeline.log'),
            logging.StreamHandler()
        ]
    )


def setup_experiment(logs_dir, args):
    if args.stage == 'all' or args.stage == 'inference':
        # 此时才会设置 dataset_name 和 task_name
        args.dataset_name = args.meta_path.split('/')[-1].split('.')[0]
        args.task_name = f'{args.dataset_name}_{args.model_name}'
    else:
        # 后续阶段时需要保证 exp_name 不为空
        assert args.exp_name is not None

    # NOTE: 注意后续的处理可能只有 result file，没有其他信息
    # 由于要输出 log file，所以最好还是
    
    
    if args.exp_name is None:
        date_str = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
        args.exp_name = '-'.join([
            f'task_{args.task_name}',
            date_str,
        ])

    args.logs_dir = os.path.join(logs_dir, args.exp_name)
    os.makedirs(log_base_path, exist_ok=True)

    # 保存实验配置
    config_path = Path(log_dir) / 'config.json'
    if not config_path.exists():
        with open(config_path, 'w') as f:
            json.dump(vars(args), f, indent=4)


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


def calc_metrics(result):
    # 这部分代码必须要分离出来，可能只执行后面的部分
    error_output = 0
    predictions, answers = [], []
    logging.info(f'Calculate metrics')
    for item in result:
        prediction = item['prediction']

        if prediction == '0' or prediction == '1':
            predictions.append(int(prediction))
            answers.append(item['answer'])
            error_output += 1
        elif prediction == 'tie' or prediction == 'unknown':
            error_output += 1
            logging.debug(f'{prediction.capitalize()} output: {item["output"]}')
            
    logging.info(f'Error output number: {error_output}')
    predictions = np.array(predictions) # NOTE: 注意是否添加了复数 s
    answers = np.array(answers)
    accuracy = np.mean(predictions == answers)
    f1 = f1_score(answers, predictions, average='weighted')
    logging.info(f'Accuracy: {accuracy:.4f}')
    logging.info(f'F1 Score: {f1:.4f}')


def main(args):
    start_time = time.time()
    args = parse_args(args)

    args.task_name = args.meta_path.split('/')[-1].split('.')[0]
    if args.exp_name is None:
        date_str = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
        args.exp_name = '-'.join([
            date_str,
            f'model_{args.model_name}',
            f'task_{args.task_name}',
        ])
    log_base_path = os.path.join(args.logs, args.exp_name)
    os.makedirs(log_base_path, exist_ok=True)

    # log 分别输出到文件和控制台
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_base_path, 'log.txt')),
            logging.StreamHandler()
        ]
    )

    logging.info(f'Model: {args.model_name}')
    logging.info(f'Task: {args.task_name}')
    logging.info(f'Dataset number: {len(dataset)}')

    # create dataset
    dataset = BRACE_Dataset(
        meta_path=args.meta_path,
        meta_type=args.meta_type,
        audio_base_path=args.audio_base_path
    )

    # inference
    # TODO: 多次尝试是否有区别？对于前后顺序影响的考虑
    logging.info(f'Start inference on {args.task_name}')
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

    result_path = os.path.join(log_base_path, 'result.json')
    logging.info(f'Save result to {result_path}')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=4)

    return

    # NOTE: subprocess 通过参数列表来传参，实际运行的是 bash 命令，-c 后面是命令字符串
    # subprocess.run() 是 subprocess.Popen() 的一个高层封装，能够等待子进程结束
    # check=True 会在子进程返回非零状态码时抛出异常（子进程的 logging 和主进程不同步）
    # sub process 会继承环境变量，所以 WORKDIR 仍然是 chy

    vllm_env_path = '/mnt/public/data/lh/chy/envs/vllm/bin/activate'
    code_path = '/mnt/public/data/lh/chy/evaluation/process.py'
    subprocess.run(
        [
            "bash", "-c",
            f"source {vllm_env_path} && python {code_path} --result_path {result_path}"
        ],
        check=True
    )

    # process 相关的逻辑不用从 main 中分离出去，单独运行 process.py 来处理
    with open(result_path, 'r') as f:
        result = json.load(f)

    calc_metrics(result)
    end_time = time.time()
    logging.info(f'Time cost: {end_time - start_time:.2f}s')


if __name__ == '__main__':
    main(sys.argv[1:])