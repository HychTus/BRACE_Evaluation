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
    parser.add_argument('--ref_num', type=int, default=0, help='Number of reference scores to use; 0 means no reference scores')
    return parser.parse_args()


def setup_experiment(args):
    log_level = logging.DEBUG if args.debug else logging.INFO
    
    # task_name 中仅包含 dataset_name 和 model_name
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
        logging.info('Using toy dataset for testing')
        dataset = dataset[:100]
    logging.info(f'Dataset number: {len(dataset)}')

    # Create Model
    logging.info(f'Start loading model {args.model_name}')
    model = create_model(args.model_name)

    logging.info(f'Start inference on {args.dataset_name} with {args.model_name}')
    captions_0 = [item['caption_0'] for item in dataset]
    captions_1 = [item['caption_1'] for item in dataset]
    refs = [item['references'][:args.ref_num] for item in dataset]
    audios = [item['audio_path'] for item in dataset]
    answers = [item['answer'] for item in dataset] # item['answer']: int

    # NOTE: model.score 返回的 score 类型是 numpy.ndarray
    scores_0 = model.score(captions_0, audios)
    scores_1 = model.score(captions_1, audios)

    if args.ref_num > 0:
        ref_scores_0 = model.score_ref(captions_0, refs)
        ref_scores_1 = model.score_ref(captions_1, refs)
        src_scores = (scores_0 + np.max(ref_scores_0, axis=1)) / 2
        dst_scores = (scores_1 + np.max(ref_scores_1, axis=1)) / 2
    else:
        src_scores = scores_0
        dst_scores = scores_1
    predictions = np.where(src_scores > dst_scores, 0, 1)

    # NOTE: 简单计算整体的 accuracy 和 f1-score，详细的计算在 calc.py
    # Calculate accuracy
    accuracy = np.mean(predictions == answers)
    logging.info(f'Accuracy: {accuracy * 100:.4f}')
    # Calculate F1-score
    f1 = f1_score(answers, predictions, average='binary')
    logging.info(f'F1-score: {f1 * 100:.4f}')

    # 将 audio score 和 ref score 都进行保存
    result = []
    for index, item in enumerate(dataset):
        # FIXME: 需要从 np.int64/np.float32 转换为 int/float 后 json 才能序列化
        item['src_score'] = float(src_scores[index])
        item['dst_score'] = float(dst_scores[index])
        item['score_0'] = float(scores_0[index])
        item['score_1'] = float(scores_1[index])
        item['prediction'] = str(predictions[index])
        if args.ref_num > 0:
            item['ref_score_0'] = ref_scores_0[index].tolist()
            item['ref_score_1'] = ref_scores_1[index].tolist()

        result.append(item)

    result_path = os.path.join(args.log_dir, f'{args.task_name}.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=4)
    logging.info(f'Final result saved to {result_path}')

    end_time = time.time()
    logging.info(f'Time cost: {end_time - start_time:.2f}s')


if __name__ == '__main__':
    main()