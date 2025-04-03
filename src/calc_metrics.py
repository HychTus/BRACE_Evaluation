import os
import json
import pandas as pd

# TODO 还有不同的类型分类的问题
# 在外部进行 tie 和 unknown 的数量统计
# calc_metrics 是已经分好类了
def calc_metrics(result):
    TP, FP, FN, TN = 0, 0, 0, 0
    legal_values = {'0', '1'}
    
    for item in result:
        # 统一转换为 string 讨论
        prediction = str(item['prediction'])
        answer = str(item['answer'])

        if prediction in legal_values:
            TP += int(prediction == "1" and answer == "1") # True Positive
            FP += int(prediction == "1" and answer == "0") # False Positive
            TN += int(prediction == "0" and answer == "0") # True Negative
            FN += int(prediction == "0" and answer == "1") # False Negative
        else:
            # 这里可以选择忽视（如果还没有处理好 unknown 的话）
            FN += int(answer == "1") # ilegal 视为预测了 0
            FP += int(answer == "0") # ilegal 视为预测了 1

    # accuracy (TP + TN) / (TP + TN + FP + FN)
    total = TP + TN + FP + FN
    accuracy = 1.0 * (TP + TN) / total if total != 0 else 0
    # F1-Score F1 = 2*TP / (2*TP + FP + FN)
    f1 = 2.0 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0

    return {
        'accuracy': accuracy * 100, 
        'f1': f1 * 100,
    }


def check(result):
    # TODO: 先判定是否有非法的，对于 unknown 再手动进行判断
    # valid_values 可以进行调整

    valid_values = {'0', '1', 'tie', 'unknown'}
    for item in result:
        prediction = item['prediction'].strip()
        if prediction not in valid_values:
            print(f'Caption_0: {item["caption_0"]}')
            print(f'Caption_1: {item["caption_1"]}')
            print(f'Invalid prediction: {item["prediction"]}, Output: {item["output"]}')


if __name__ == '__main__':
    # NOTE: 由于已经预先计算了结果，所以可以直接计算结果，然后统计出来表格！
    # 每组实验都放到一起？
    # 这个程序不会多次进行运行，所以直接内嵌参数即可
    # 还有对于 Main 如何进行计算的考虑
    # 对于一组 log 同时运行
    # NOTE: 对于 0, 1, unknown, tie 有不同的处理方式，分别进行计算
    # 还需要对于数据量进行统计，比如 tie 和 unknown 数量，总数量
    
    log_base_dir = '/mnt/public/data/lh/chy/evaluation/logs'
    Main_result, Hallu_result = [], []

    for item in os.listdir(log_base_dir):
        item_path = os.path.join(log_base_dir, item)

        if os.path.isdir(item_path) == False:
            continue

        task_name = item.split('-')[0]
        print(f'Task name: {task_name}')
        result_path = os.path.join(item_path, f'processed_{task_name}.json')

        if not os.path.exists(result_path):
            print(f'File not exists: {result_path}')
            continue

        with open(result_path, 'r', ) as f:
            result = json.load(f)

        total_count = len(result)
        legal_count = sum(1 for item in result if item['prediction'] in {'0', '1'})
        tie_count = sum(1 for item in result if item['prediction'] == 'tie')
        unknown_count = total_count - legal_count - tie_count
        bias_count = sum(1 for item in result if item['prediction'] == '0')
        
        # FIXME: ans 和 result 命名混淆
        ans = {
            'task_name': task_name,
            'total_count': total_count,
            'legal_percent': legal_count / total_count * 100,
            'tie_percent': tie_count / total_count * 100,
            'unknown_percent': unknown_count / total_count * 100,
            'bias_percent': bias_count / total_count * 100,
        }

        if 'Main' in task_name:
            # filter
            result = [item for item in result if item['score'] <= -2 or item['score'] >= 2]
            
            # 还是所有类别分开吧，方便进行讨论
            # 对于平均数相关的操作，交给 excel 来处理
            pair_type = set([item['pair_type'] for item in result])
            for type_name in pair_type:
                type_result = [item for item in result if item['pair_type'] == type_name]
                type_metric = calc_metrics(type_result)
                ans[type_name + '_acc'] = type_metric['accuracy']
                ans[type_name + '_f1'] = type_metric['f1']
            Main_result.append(ans)

        elif 'Hallu' in task_name:
            metric = calc_metrics(result)
            ans.update(metric)
            Hallu_result.append(ans)

    Main_df = pd.DataFrame(Main_result)
    Main_df.to_csv('Main_result.csv', index=False)

    Hallu_df = pd.DataFrame(Hallu_result)
    Hallu_df.to_csv('Hallu_result.csv', index=False)

# 对于每个 result 进行统计，按照不同的方法统计分数，一种是将 unknown 和 tie 直接是为错误，整体为二分类；另一种是将 unknown 和 tie 视为其他的分类，整体为四分类；
# 然后计算正确率和 F1 分数，并且对于 unknown 和 tie 的数量进行统计
# 最后将所有的统计结果输出，并且汇总到 csv 表格中保存
# 还有一种计算分数的方式是 unknown 的情况不纳入统计，按照 0，1，tie 三分类计算

# 另外还有过滤的问题，如果分数无法达到阈值