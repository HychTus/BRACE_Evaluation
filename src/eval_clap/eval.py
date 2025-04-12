import os
import time
import json
import logging
import argparse
import numpy as np

from tqdm import tqdm
from datetime import datetime

from ..data import BRACE_Dataset
from .factory import create_model
from sklearn.metrics import f1_score


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate LLM pre-pipeline')
    parser.add_argument('--exp_name', type=str, help='Experiment name for tracking; must be provided if --resume is used')
    parser.add_argument('--log_base_dir', type=str, default='logs', help='Root directory for experiment logs')
    parser.add_argument('--meta_path', type=str, required=True, help='Path to metadata file')
    parser.add_argument('--meta_type', type=str, required=True, choices=['Main', 'Hallu'], help='Metadata format type')
    parser.add_argument('--audio_base_dir', type=str, required=True, help='Base directory for audio files')
    parser.add_argument('--model_name', type=str, required=True, help='CLAP model name')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--toy_dataset', action='store_true', help='Use toy dataset for testing')
    return parser.parse_args()


def setup_experiment(args):
    log_level = logging.DEBUG if args.debug else logging.INFO
    
    args.dataset_name = args.meta_path.split('/')[-1].split('.')[0]
    args.task_name = f'{args.dataset_name}_{args.model_name}'

    if args.exp_name is None:
        date_str = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
        args.exp_name = '-'.join([f'{args.task_name}', date_str])
    
    assert os.path.exists(args.log_base_dir), f"Log base directory {args.log_base_dir} does not exist."
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    os.makedirs(args.log_dir, exist_ok=False)

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.log_dir, 'log.txt')),
            logging.StreamHandler()
        ]
    )

    config_path = os.path.join(args.log_dir, 'config.json')
    with open(config_path, 'w') as config_file:
        json.dump(vars(args), config_file, indent=4)
    logging.info(f"New experiment started: {args.exp_name}")


def main():
    start_time = time.time()
    args = parse_args()
    setup_experiment(args)

    logging.info(f'Task: {args.task_name}')
    logging.info(f'Experiment name: {args.exp_name}')
    logging.debug(f'Arguments: {json.dumps(vars(args), indent=4)}') 

    # Create Dataset
    dataset = BRACE_Dataset(
        meta_path=args.meta_path,
        meta_type=args.meta_type,
        audio_base_dir=args.audio_base_dir
    )
    if args.toy_dataset:
        dataset = dataset[:100]
    logging.info(f'Dataset number: {len(dataset)}')

    # Create Model
    logging.info(f'Start loading model {args.model_name}')
    model = create_model(args.model_name)

    logging.info(f'Start inference on {args.dataset_name} with {args.model_name}')

    src_captions = [item['caption_0'] for item in dataset]
    dst_captions = [item['caption_1'] for item in dataset]
    audios = [item['audio_path'] for item in dataset]
    answers = [item['answer'] for item in dataset]

    # src_score 和 dst_score 的类型是 numpy.ndarray
    src_score = model.score(src_captions, audios)
    dst_score = model.score(dst_captions, audios)
    comparison_array = np.where(src_score > dst_score, 0, 1)

    # Calculate accuracy
    accuracy = np.mean(comparison_array == answers)
    logging.info(f'Accuracy: {accuracy * 100:.4f}')

    # Calculate F1-score
    f1 = f1_score(answers, comparison_array, average='binary')
    logging.info(f'F1-score: {f1 * 100:.4f}')

    result = []
    for index, item in enumerate(dataset):
        # 需要从 np.int64/np.float32 转换为 int/float 后 json 才能序列化
        item['score_0'] = float(src_score[index])
        item['score_1'] = float(dst_score[index])
        item['prediction'] = str(comparison_array[index])
        result.append(item)

    result_path = os.path.join(args.log_dir, f'{args.task_name}.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=4)
    logging.info(f'Final result saved to {result_path}')

    end_time = time.time()
    logging.info(f'Time cost: {end_time - start_time:.2f}s')


if __name__ == '__main__':
    main()