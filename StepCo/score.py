import warnings
warnings.filterwarnings('ignore')
import argparse
import os
import math

from config import Config
from utils import load_txt_data, post_process_value


config = Config()
FINAL = True


## loading result
parser = argparse.ArgumentParser(description="index of math word problem datasets")
parser.add_argument('--dataset_idx', type=int, required=True, metavar='', default=0, help=f"{config.dataset_name_list}")
args = parser.parse_args()
dataset_index = args.dataset_idx

dataset_name = config.dataset_name_list[dataset_index]
answer_save_path = os.path.join(config.result_save_root_path, f'{dataset_name}-{config.prompt_strategy}-{config.backend_LLM}.txt')
data = load_txt_data(answer_save_path)


if type(data[0].get('gold_answer'))==str:
    # gold = [eval(sub_data.get('gold_answer').replace(',', '')) for sub_data in data]
    gold = [post_process_value(sub_data.get('gold_answer')) for sub_data in data]
else:
    gold = [sub_data.get('gold_answer') for sub_data in data]
pred = []
for sub_data in data:
    count = -1
    for key, value in sub_data.items():
        if 'iter' in key:
            count += 1
    if FINAL:
        pred.append(post_process_value(sub_data[f'iter-{count}']['answer']))
    else:
        pred.append(post_process_value(sub_data[f'iter-0']['answer']))


count = 0
label = []
wrong_index = []
for i in range(len(data)):
    print(i, gold[i], pred[i])
    if math.isclose(gold[i], pred[i], rel_tol=1e-5, abs_tol=1e-5):
        count += 1
        label.append('right')
        data[i]['judge'] = 'right'
    elif math.isclose(gold[i] / 100, pred[i], rel_tol=1e-5, abs_tol=1e-5):
        count += 1
        label.append('right')
        data[i]['judge'] = 'right'
    elif math.isclose(gold[i], pred[i] / 100, rel_tol=1e-5, abs_tol=1e-5):
        count += 1
        label.append('right')
        data[i]['judge'] = 'right'
    else:
        label.append('wrong')
        data[i]['judge'] = 'wrong'
        wrong_index.append(i)
print(f'Accuracy of {dataset_name}-{config.prompt_strategy}-{config.backend_LLM}: {count / len(gold)*100:.2f}%')
# utils.save_json_file(data, save_path)
print(f'{len(gold)} problems have been solved. Index of the wrong question: {wrong_index}')
