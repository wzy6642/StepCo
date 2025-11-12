import warnings
warnings.filterwarnings('ignore')
import os
import json

import tiktoken


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding_name = "cl100k_base"
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


class DataLoader:
    def __init__(self, dataset_name, root_path):
        self.dataset_name = dataset_name
        self.root_path = root_path
        self.dataset_path = os.path.join(self.root_path, f'{dataset_name}.json')
        self.read_dataset()
        self.print_dataset_info()

    def read_dataset(self):
        with open(self.dataset_path, 'r') as f:
            dataset = json.load(f)
            self.dataset = dataset

    def get_problems(self):
        if self.dataset_name not in ['MATH', 'AQuA', 'ASDiv']:
            problems = [data.get('problem') for data in self.dataset]
        elif self.dataset_name == 'ASDiv':
            conditions = [data.get('Body') for data in self.dataset]
            query = [data.get('Question') for data in self.dataset]
            problems = [conditions[idx] + ' ' + query[idx] for idx in range(len(conditions))]
        else:
            problems = [data.get('original_question') for data in self.dataset]
        if self.dataset_name == 'AQuA':
            options = self.get_options()
            problems = [f'{problems[idx]} \nAnswer Choices: {options[idx]}' for idx in range(len(problems))]
        self.problems = problems
        return problems

    def get_gold_answers(self):
        gold_answers = [data.get('gold_answer') if self.dataset_name != 'ASDiv' else data.get('Answer') for data in self.dataset]
        self.gold_answers = gold_answers
        return gold_answers

    def get_gold_reasoning_paths(self):
        gold_reasoning_paths = [data.get('equation') if self.dataset_name != 'ASDiv' else data.get('Formula') for data in self.dataset]
        return gold_reasoning_paths

    def get_options(self):
        options = [data.get('options') for data in self.dataset]
        options = ['(' + '('.join(i) for i in options]
        options = [i.replace('(', ' (').replace(')', ') ') for i in options]
        return options

    def get_other_information(self):
        subject = [data.get('subject') for data in self.dataset]
        level = [data.get('level') for data in self.dataset]
        return subject, level

    def print_dataset_info(self):
        print(f'[INFO] {self.dataset_name} - Path of dataset: {self.dataset_path}')
        print(f'[INFO] {self.dataset_name} - Number of math word problems: {len(self.dataset)}')

    def print_avg_tokens(self, problems):
        print(f'[INFO] {self.dataset_name} - Average tokens in problems: {sum([num_tokens_from_string(problem) for problem in problems]) / (len(problems)):.2f}')