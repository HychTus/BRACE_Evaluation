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