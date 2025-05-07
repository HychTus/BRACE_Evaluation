import os
import json
import logging
import argparse

from datetime import datetime
from vllm import LLM, SamplingParams
from .prompt import post_prompt_template


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate LLM post-pipeline')
    parser.add_argument('--exp_name', type=str, help='Experiment name for tracking')
    parser.add_argument('--target', type=str, required=True, help='Target for evaluation')
    parser.add_argument('--prompt_template_type', type=str, required=True, choices=list(post_prompt_template.keys()), help='Prompt template type')

    parser.add_argument('--log_base_dir', type=str, default='logs', help='Root directory for experiment logs')
    parser.add_argument('--model_name', type=str, default='Qwen2.5-14B-Instruct')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
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
    logging.info(f'Experiment directory: {args.log_dir}')

    global prompt_template
    prompt_template = post_prompt_template[args.prompt_template_type]
    args.prompt_template = prompt_template.format(
        caption_0='caption_0',
        caption_1='caption_1',
        answer='answer',
    )
    logging.info(f'Using prompt template: {args.prompt_template}')

    if os.path.isfile(args.target):
        args.meta_paths = [args.target]
    elif os.path.isdir(args.target):
        args.meta_paths = [os.path.join(args.target, f) for f in os.listdir(args.target) if f.endswith('.json')]
    else:
        logging.error(f"Target {args.target} is neither a file nor a directory")
        raise ValueError(f"Target {args.target} is neither a file nor a directory")

    for meta_path in args.meta_paths:
        logging.info(f'Target file: {meta_path}')

    # Save the configuration to a JSON file
    config_path = os.path.join(args.log_dir, 'config.json')
    with open(config_path, 'w') as config_file:
        json.dump(vars(args), config_file, indent=4)


def main():
    args = parse_args()
    setup_experiment(args)

    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_num_seqs=256, 
    )

    sampling_params = SamplingParams(
        temperature=0.2,
        top_p=0.7,
        max_tokens=1
    )

    for meta_path in args.meta_paths:
        logging.info(f'Processing file: {meta_path}')
        with open(meta_path, 'r') as f:
            result = json.load(f)

        prompts = []
        idx_map = []
        for idx, item in enumerate(result):
            item['post_prompt'] = {}
            item['prediction'] = {}
            for key in item['output'].keys():
                prompt = prompt_template.format(
                    caption_0=item['caption_0'],
                    caption_1=item['caption_1'],
                    answer=item['output'][key]
                )
                item['post_prompt'][key] = prompt
                prompts.append(prompt)
                idx_map.append((idx, key))

        outputs = llm.generate(prompts, sampling_params)
        for (idx, key), output in zip(idx_map, outputs):
            result[idx]['prediction'][key] = output.outputs[0].text

        result_path = os.path.join(args.log_dir, os.path.basename(meta_path))
        logging.info(f'Result saved to {result_path}')
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=4)


if __name__ == '__main__':
    main()