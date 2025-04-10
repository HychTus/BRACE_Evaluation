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
    # TODO: 使用 sys.argv[1:] 相比直接使用 parser 的优势？能够在代码中手动传参来模拟？
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


def setup_experiment(args):
    # BUG: 标准的 resume 应该只使用 exp_name，并且读取其中的 config.json 恢复
    # 不确定 parser 应该如何适应这种情况，其他参数不传递会有问题
    # 所以还是要求指定基础参数，并且和 config 中进行对比确认，保证实验恢复没问题 

    global prompt_template
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
        
        # 更新 args 中的部分参数，使其与保存的配置保持一致
        # BUG: 暂时不更新，更新会导致 resume attribute 被覆盖；
        # 这里只是恢复其他额外计算的参数 (task_name)，传入的参数没有更改
        for key, value in saved_config.items():
            if not hasattr(args, key):
                setattr(args, key, value)
            else:
                assert getattr(args, key) == value, f"Argument {key} does not match saved config. Please check."
        
        # config 中保存的不是 format string，而是已经填充后的结果，所以要重新计算
        prompt_template = pre_prompt_template[args.prompt_template_type]
        logging.info(f"Resuming experiment from config: {args.exp_name}")

    else:
        args.dataset_name = args.meta_path.split('/')[-1].split('.')[0]
        args.task_name = f'{args.dataset_name}_{args.model_name}_{args.prompt_template_type}'
        prompt_template = pre_prompt_template[args.prompt_template_type]
        args.prompt_template = prompt_template.format(caption_0='caption_0', caption_1='caption_1')

        if args.exp_name is None:
            # 保存时 task 在前便于进行查找，date_str 作为附加信息进行区分
            date_str = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
            args.exp_name = '-'.join([f'{args.task_name}', date_str])
        
        # 参数传递的是 log_base_dir，实验对应的目录为 log_dir
        # 需要进行检查，否则可能出现由于 Muxi 挂载问题在其他位置创建了目录
        assert os.path.exists(args.log_base_dir), f"Log base directory {args.log_base_dir} does not exist."
        args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
        os.makedirs(args.log_dir, exist_ok=False) # 如果已经存在会有问题

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


def generate_prompt(item, ref_num=0):
    # TODO: 暂时先不测试 ref，我记得有只使用 ref 和 caption 的 evaluation method
    if ref_num > 0:
        raise NotImplementedError

    # 也可以通过函数属性实现，反正不想把 args 放到函数中
    global prompt_template
    return prompt_template.format(
        caption_0=item['caption_0'],
        caption_1=item['caption_1'],
    )


def test_prompt():
    print(generate_prompt({
        'caption_0': 'A cat is sitting on a windowsill.',
        'caption_1': 'A dog is barking at a squirrel.'
    }))


def get_resume_result(args):
    # TODO: 对于这部分代码的理解
    # 检查 resume 模式下是否存在部分结果
    result = []
    resume_start = 0
    partial_result_base_dir = os.path.join(args.log_dir, 'partial_results')
    if args.resume:
        if os.path.exists(partial_result_base_dir):
            partial_files = glob.glob(os.path.join(partial_result_base_dir, 'result_partial_*.json'))
            if partial_files:
                # 提取所有 partial 文件中的 index
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

    # FIXME: 由于在 setup_experiment 中调用了 logging，所以没有进行 basicConfig，导致 info 和 debug 都无法输出
    # 必须先生成 log_dir，然后配置 logging，然后再使用 logging 输出信息
    # NOTE: VSCode 输出路径时候可以点击查看是否存在，比较方便
    logging.info(f'Task: {args.task_name}')
    logging.info(f'Experiment name: {args.exp_name}')
    logging.info(f'Prompt template: {args.prompt_template}')

    # 输出更详细信息，json.dump(data,f) 输出到文件并且无返回值，json.dumps(data) 返回字符串
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
    # TODO: 多次尝试 inference 是否有区别？对于前后顺序影响的考虑
    # NOTE: 如果设置 temperature=0.0，输出就不会有随机性
    logging.info(f'Start inference on {args.dataset_name} with {args.model_name}')
    if not args.single_inference:
        raise NotImplementedError

    for idx, item in enumerate(tqdm(dataset[resume_start:], desc="Processing items"), start=resume_start + 1):
        audio_path = item['audio_path']
        prompt = generate_prompt(item, ref_num=args.ref_num)
        output = inference_single(
            audio_path=audio_path,
            prompt=prompt,
            model=model,
        )
        item['prompt'] = prompt
        item['output'] = output
        result.append(item)

        # save_interval 次数保存一次中间结果
        save_interval = 100
        if idx % save_interval == 0:
            partial_result_path = os.path.join(partial_result_base_dir, f'result_partial_{idx}.json')
            with open(partial_result_path, 'w') as f:
                json.dump(result, f, indent=4)
            logging.info(f'Intermediate result saved to {partial_result_path}')

    result_path = os.path.join(args.log_dir, f'{args.task_name}.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=4)
    logging.info(f'Final result saved to {result_path}')

    # 保存最终结果后删除中间结果
    shutil.rmtree(partial_result_base_dir)
    logging.info(f"Partial results directory {partial_result_base_dir} has been deleted.")

    end_time = time.time()
    logging.info(f'Time cost: {end_time - start_time:.2f}s')


if __name__ == '__main__':
    test_prompt()
    # main()