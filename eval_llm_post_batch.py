import os
import json
import logging
import argparse

from datetime import datetime
from vllm import LLM, SamplingParams
from .prompt import prompt_template_dict

model_base_dir = '/mnt/public/data/lh/models'

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--exp_name', type=str, help='Experiment name for tracking')
    parser.add_argument('--log_base_dir', type=str, default='logs', help='Root directory for experiment logs')
    parser.add_argument('--pre_log_dir', type=str, default='logs', help='Root directory for experiment logs')
    parser.add_argument('--model_name', type=str, default='Qwen2.5-14B-Instruct')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--calc_metrics', action='store_true', help='Calculate metrics after processing')
    parser.add_argument('--prompt_template_type', type=str, default='naive', help='Prompt template to use')
    args = parser.parse_args()
    return args


def setup_experiment(args):
    global prompt_template
    prompt_template = prompt_template_dict[args.prompt_template_type]
    args.prompt_template = prompt_template.format(prediction='prediction')

    date_str = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    args.exp_name = date_str

    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    os.makedirs(args.log_dir, exist_ok=True)

    config_path = os.path.join(args.log_dir, 'config.json')
    with open(config_path, 'w') as config_file:
        json.dump(vars(args), config_file, indent=4)

    origin_dir = os.path.join(args.log_dir, 'origin')
    os.makedirs(origin_dir, exist_ok=True)
    args.origin_dir = origin_dir

    for file_name in os.listdir(args.pre_log_dir):
        src_file = os.path.join(args.pre_log_dir, file_name)
        dest_file = os.path.join(origin_dir, file_name)
        if os.path.isfile(src_file):
            with open(src_file, 'rb') as src, open(dest_file, 'wb') as dest:
                dest.write(src.read())


def init_logging(args):
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.log_dir, 'log.txt')),
            logging.StreamHandler()
        ]
    )


def calc_metrics(result):
    error_output = 0
    predictions, answers = [], []
    for item in result:
        prediction = item['prediction'].strip()

        if prediction is None:
            error_output += 1
            # print(f'Error output: {item["output"]}')
        else:
            predictions.append(prediction)
            answers.append(item['answer'])

        if prediction == 1:
            print(item['output'])

    predictions = np.array(predictions)
    answers = np.array(answers)
    accuracy = np.mean(predictions == answers)
    f1 = f1_score(answers, predictions, average='weighted')

    print(f'Error output number: {error_output}')
    print(f'Accuracy: {accuracy}')
    print(f'F1 score: {f1}')


def main():
    args = parse_args()
    setup_experiment(args)
    init_logging(args)

    logging.info(f'Task name: {args.task_name}')
    logging.info(f'Experiment name: {args.exp_name}')
    logging.info(f'Model name: {args.model_name}')

    with open(args.meta_path, 'r') as f:
        result = json.load(f)

    prompts = []
    for item in result:
        prompt = prompt_template.format(item['output'])
        prompts.append(prompt)

    model_path = os.path.join(model_base_dir, args.model_name)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,      # 根据GPU数量调整 （单卡模型并非越多越好）
        gpu_memory_utilization=0.9,  # 设置GPU利用率
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

    # 使用 task_name 进行存储，方便后续直接移动
    processed_result_path = os.path.join(args.log_dir, f'{args.task_name}_processed.json')
    logging.info(f'Result saved to {processed_result_path}')
    with open(processed_result_path, 'w') as f:
        json.dump(result, f, indent=4)

    if args.calc_metrics:
        calc_metrics(result)


def test_prompt():
    print(prompt_template.format('caption_0', 'caption_1', 'prediction'))


if __name__ == '__main__':
    main()