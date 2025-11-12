import warnings
warnings.filterwarnings('ignore')
import os
import argparse
import json

from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch

from config import Config
from data_loader import DataLoader
from utils import num_data_in_txt_file
from solving_pipeline import pipeline
from verification import step_verify_score


config = Config()


## loading dataset
parser = argparse.ArgumentParser(description="index of math word problem datasets")
parser.add_argument('--dataset_idx', type=int, required=True, metavar='', default=0, help=f"{config.dataset_name_list}")
args = parser.parse_args()
dataset_index = args.dataset_idx

dataset_name = config.dataset_name_list[dataset_index]
data_loader = DataLoader(dataset_name, config.dataset_root_path)
problems = data_loader.get_problems()
gold_answers = data_loader.get_gold_answers()
gold_reasoning_paths = data_loader.get_gold_reasoning_paths()
if dataset_name=='MATH':
    subject, level = data_loader.get_other_information()
data_loader.print_avg_tokens(problems)
answer_save_path = os.path.join(config.result_save_root_path, f'{dataset_name}-{config.prompt_strategy}-{config.backend_LLM}.txt')
print(f'[INFO] Answer save path: {answer_save_path}')


## solving math word problems
if not os.path.exists(answer_save_path):
    num_problem_solved = 0
else:
    num_problem_solved = num_data_in_txt_file(answer_save_path)
for question_idx in tqdm(range(len(problems)), desc=f'{dataset_name}-{config.prompt_strategy}-{config.backend_LLM}'):
    question_idx += num_problem_solved
    solve_process_record = {}
    problem = problems[question_idx]
    solve_process_record['problem'] = problem
    solve_process_record['gold_answer'] = gold_answers[question_idx]

    final_answer, solve_process_record = pipeline(
        solve_process_record,
        step_verify_score
    )

    solve_process_record['final_answer'] = final_answer
    with open(answer_save_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(solve_process_record, ensure_ascii=False) + '\n')

