import os
import json
import logging
import argparse
from datetime import datetime
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, help='Experiment name for tracking')
    parser.add_argument('--target', type=str, required=True, help='Target for evaluation')
    parser.add_argument('--log_base_dir', type=str, default='logs', help='Root directory for experiment logs')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    return args


def setup_experiment(args):
    # Set up logging
    if args.exp_name is None:
        date_str = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
        args.exp_name = f'calc_{date_str}'

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

    # Set up the target
    if os.path.isfile(args.target):
        args.meta_paths = [args.target]
    elif os.path.isdir(args.target):
        args.meta_paths = [os.path.join(args.target, f) for f in os.listdir(args.target) if f.endswith('.json')]
    else:
        logging.error(f"Target {args.target} is neither a file nor a directory")
        raise ValueError(f"Target {args.target} is neither a file nor a directory")

    for meta_path in args.meta_paths:
        logging.info(f'Target file: {meta_path}')


def calc_metrics(result):
    TP, FP, FN, TN = 0, 0, 0, 0
    legal_values = {'0', '1'}
    
    for item in result:
        prediction = str(item['prediction'])
        answer = str(item['answer'])

        if prediction in legal_values:
            TP += int(prediction == "1" and answer == "1") # True Positive
            FP += int(prediction == "1" and answer == "0") # False Positive
            TN += int(prediction == "0" and answer == "0") # True Negative
            FN += int(prediction == "0" and answer == "1") # False Negative
        else:
            FN += int(answer == "1")
            FP += int(answer == "0")

    total = TP + TN + FP + FN
    accuracy = 1.0 * (TP + TN) / total if total != 0 else 0
    f1 = 2.0 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0

    return {
        'acc': accuracy * 100, 
        'f1': f1 * 100,
    }


def check(result):
    valid_values = {'0', '1', 'tie', 'unknown'}
    for item in result:
        prediction = item['prediction'].strip()
        if prediction not in valid_values:
            logging.warning(f'Caption_0: {item["caption_0"]}')
            logging.warning(f'Caption_1: {item["caption_1"]}')
            logging.warning(f'Output: {item["output"]}')
            logging.warning(f'Invalid prediction: {prediction}')


def main():
    args = parse_args()
    setup_experiment(args)

    Main_result, Hallu_result = [], []
    for meta_path in args.meta_paths:
        logging.info(f'Processing file: {meta_path}')
        with open(meta_path, 'r') as f:
            result = json.load(f)

        task_name = os.path.basename(meta_path).split('.')[0]
        task_attr = task_name.split('_')
        task_attr = task_attr[:-1] if len(task_attr) > 4 else task_attr

        task_attr_name = ['Dataset', 'Type', 'Version', 'Model']
        assert len(task_attr) == len(task_attr_name), f"Task name {task_name} does not match expected format"

        import copy
        origin_result = copy.deepcopy(result)

        ans = dict(zip(task_attr_name, task_attr))
        total_count = len(result)
        zero_count = sum(1 for item in result if item['prediction'] == '0')
        one_count = sum(1 for item in result if item['prediction'] == '1')
        tie_count = sum(1 for item in result if item['prediction'] == 'tie')
        unknown_count = total_count - zero_count - one_count - tie_count
        
        ans.update({
            'total': total_count,
            'zero': 100.0 * zero_count / total_count,
            'one': 100.0 * one_count / total_count,
            'tie': 100.0 * tie_count / total_count,
            'unknown': 100.0 * unknown_count / total_count,
        })

        if 'Main' in task_name:
            origin_num = len(result)
            result = [item for item in result if item['score'] <= -2 or item['score'] >= 2]
            logging.info(f'Filtered result: {len(result)} out of {origin_num} items')

            pair_type_dict = {
                'HH': ['Human-Human'],
                'HM': ['Human-Machine_1', 'Human-Machine_2'],
                'MM': ['Machine-Machine_1', 'Machine-Machine_2', 'Machine-Machine_3'],
                'All': ['Human-Human', 'Human-Machine_1', 'Human-Machine_2', 'Machine-Machine_1', 'Machine-Machine_2', 'Machine-Machine_3'],
                'HM_1': ['Human-Machine_1'],
                'HM_2': ['Human-Machine_2'],
                'MM_1': ['Machine-Machine_1'],
                'MM_2': ['Machine-Machine_2'],
                'MM_3': ['Machine-Machine_3'],
            }
            pair_type_dict['All'] = pair_type_dict['HH'] + pair_type_dict['HM'] + pair_type_dict['MM']

            for pair_type, name_list in pair_type_dict.items():
                pair_result = [item for item in result if item['pair_type'] in name_list]
                pair_metric = calc_metrics(pair_result)

                ans[pair_type + '_acc'] = pair_metric['acc']
                ans[pair_type + '_f1'] = pair_metric['f1']
            Main_result.append(ans)

        elif 'Hallu' in task_name:
            metric = calc_metrics(result)
            ans.update(metric)
            Hallu_result.append(ans)

    Main_df = pd.DataFrame(Main_result)
    Main_df.to_excel(os.path.join(args.log_dir, 'Main_result.xlsx'), index=False)
    Main_df.to_csv(os.path.join(args.log_dir, 'Main_result.csv'), index=False)

    Hallu_df = pd.DataFrame(Hallu_result)
    Hallu_df.to_excel(os.path.join(args.log_dir, 'Hallu_result.xlsx'), index=False)
    Hallu_df.to_csv(os.path.join(args.log_dir, 'Hallu_result.csv'), index=False)


if __name__ == '__main__':
    main()