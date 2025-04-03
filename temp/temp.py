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

import json
import numpy as np
from sklearn.metrics import f1_score

def output_to_prediction(output):
    digits = set(c for c in output if c.isdigit())
    if digits == {'0'}:
        return 0
    elif digits == {'1'}:
        return 1

    if 'first' in output and 'second' not in output:
        return 0
    elif 'second' in output and 'first' not in output:
        return 1
    
    return None

def calc_metrics(result_path):
    with open(result_path, 'r') as f:
        result = json.load(f)

    error_output = 0
    predictions, answers = [], []
    for item in result:
        prediction = output_to_prediction(item['output'])
        item['prediction'] = prediction
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

if __name__ == '__main__':
    result_path = '/mnt/public/data/lh/chy/evaluation/res/GAMA_AudioCaps_Hallu_v1.json'
    calc_metrics(result_path)


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