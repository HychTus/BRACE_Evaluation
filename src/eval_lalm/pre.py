import os
import re
import time
import json
import glob
import shutil
import logging
import argparse

from tqdm import tqdm
from datetime import datetime

from ..data import BRACE_Dataset
from .infer import inference_single, inference_batch
from .prompt import pre_prompt_template
from .factory import create_model


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate LLM pre-pipeline')

    parser.add_argument('--exp_name', type=str, help='Experiment name for tracking; must be provided if --resume is used')
    parser.add_argument('--resume', action='store_true', help='Resume experiment using the provided exp_name')
    parser.add_argument('--log_base_dir', type=str, default='logs', help='Root directory for experiment logs')
    parser.add_argument('--meta_path', type=str, required=True, help='Path to metadata file')
    parser.add_argument('--meta_type', type=str, required=True, choices=['Main', 'Hallu'], help='Metadata format type')
    parser.add_argument('--audio_base_dir', type=str, required=True, help='Base directory for audio files')
    parser.add_argument('--model_name', type=str, required=True, help='Model name for inference')
    parser.add_argument('--ref_num', type=int, default=0, help='Number of reference samples')
    parser.add_argument('--prompt_template_type', type=str, default='naive', help='Prompt template to use')
    parser.add_argument('--single_inference', action='store_true', help='Enable single inference mode')

    # Debug
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--toy_dataset', action='store_true', help='Use toy dataset for testing')
    return parser.parse_args()


def set_prompt_template(args):
    ptype = args.prompt_template_type
    if ptype == 'all':
        args.prompt_template = pre_prompt_template
    else:
        assert pre_prompt_template.get(ptype) is not None, f"Prompt template {ptype} not found."
        args.prompt_template = {ptype: pre_prompt_template[ptype]}


def setup_experiment(args):
    log_level = logging.DEBUG if args.debug else logging.INFO
    
    if args.resume:
        if args.exp_name is None:
            raise ValueError("When using --resume, you must provide --exp_name.")
        args.log_dir = os.path.join(args.log_base_dir, args.exp_name)

        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(args.log_dir, 'log.txt')),
                logging.StreamHandler()
            ]
        )

        config_path = os.path.join(args.log_dir, 'config.json')
        if not os.path.exists(config_path):
            raise ValueError(f"Config file {config_path} not found. Cannot resume.")
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
        
        for key, value in saved_config.items():
            if not hasattr(args, key):
                setattr(args, key, value)
            else:
                if key in ['resume']:
                    continue
                assert getattr(args, key) == value, f"Argument {key} does not match saved config. Please check."
        logging.info(f"Resuming experiment from config: {args.exp_name}")

    else:
        args.dataset_name = args.meta_path.split('/')[-1].split('.')[0]
        args.task_name = f'{args.dataset_name}_{args.model_name}_{args.prompt_template_type}'
        set_prompt_template(args)

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

        # Save experiment configuration to config.json
        config_path = os.path.join(args.log_dir, 'config.json')
        with open(config_path, 'w') as config_file:
            json.dump(vars(args), config_file, indent=4)
        logging.info(f"New experiment started: {args.exp_name}")


def get_resume_result(args):
    result = []
    resume_start = 0
    partial_result_base_dir = os.path.join(args.log_dir, 'partial_results')
    if args.resume:
        if os.path.exists(partial_result_base_dir):
            partial_files = glob.glob(os.path.join(partial_result_base_dir, 'result_partial_*.json'))
            if partial_files:
                indices = []
                for file in partial_files:
                    match = re.search(r'result_partial_(\d+).json', file)
                    if match:
                        indices.append(int(match.group(1)))
                if indices:
                    latest_index = max(indices)
                    partial_file = os.path.join(partial_result_base_dir, f'result_partial_{latest_index}.json')
                    with open(partial_file, 'r') as f:
                        result = json.load(f)
                    resume_start = latest_index
                    logging.info(f"Resuming from partial result {partial_file} (processed {latest_index} items)")
                else:
                    logging.info("No valid partial result files found. Starting from beginning.")
            else:
                logging.info("No partial result files found. Starting from beginning.")
        else:
            logging.info("Partial results directory does not exist. Starting from beginning.")
    return result, resume_start, partial_result_base_dir


def main():
    start_time = time.time()
    args = parse_args()
    setup_experiment(args)

    logging.info(f'Task: {args.task_name}')
    logging.info(f'Experiment name: {args.exp_name}')
    logging.info(f'Prompt template: {json.dumps(args.prompt_template, indent=4)}')

    logging.debug(f'Arguments: {json.dumps(vars(args), indent=4)}') 

    # Create Dataset
    dataset = BRACE_Dataset(
        meta_path=args.meta_path,
        meta_type=args.meta_type,
        audio_base_dir=args.audio_base_dir
    )
    if args.toy_dataset:
        dataset = dataset[:10]
    logging.info(f'Dataset number: {len(dataset)}')

    # Resume Results
    result, resume_start, partial_result_base_dir = get_resume_result(args)
    os.makedirs(partial_result_base_dir, exist_ok=True)
    logging.info(f'Partial results will be saved to {partial_result_base_dir}')

    # Create Model
    logging.info(f'Start loading model {args.model_name}')
    model = create_model(args.model_name)

    # Inference
    logging.info(f'Start inference on {args.dataset_name} with {args.model_name}')
    if not args.single_inference:
        raise NotImplementedError

    for idx, item in enumerate(tqdm(dataset[resume_start:], desc="Processing items"), start=resume_start + 1):
        audio_path = item['audio_path']
        item['prompt'] = {}
        item['output'] = {}

        for (ptype, prompt) in args.prompt_template.items():
            prompt = prompt.format(
                caption_0=item['caption_0'],
                caption_1=item['caption_1'],
                ref=' '.join(item['references'][:args.ref_num]),
            )
            output = inference_single(
                audio_path=audio_path,
                prompt=prompt,
                model=model,
            )
            item['prompt'][ptype] = prompt
            item['output'][ptype] = output

        result.append(item)
        save_interval = 100
        if idx % save_interval == 0:
            partial_result_path = os.path.join(partial_result_base_dir, f'result_partial_{idx}.json')
            with open(partial_result_path, 'w') as f:
                json.dump(result, f, indent=4)
            logging.info(f'Intermediate result saved to {partial_result_path}')

    result_path = os.path.join(args.log_dir, f'{args.task_name}.json')

    shutil.rmtree(partial_result_base_dir)
    logging.info(f"Partial results directory {partial_result_base_dir} has been deleted.")

    with open(result_path, 'w') as f:
        json.dump(result, f, indent=4)
    logging.info(f'Final result saved to {result_path}')

    end_time = time.time()
    logging.info(f'Time cost: {end_time - start_time:.2f}s')


if __name__ == '__main__':
    main()