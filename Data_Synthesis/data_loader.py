import warnings
warnings.filterwarnings('ignore')

import json
import re
import os

import tiktoken


class LoadData:
    def __init__(self, data_path) -> None:
        self.data_path = data_path

    def load_data_jsonl(self):
        """
            Load data from a *.jsonl file at the specified path.
        """
        with open(self.data_path, 'r') as f:
            decoder = json.JSONDecoder()
            data = f.readlines()
            data = [decoder.raw_decode(i)[0] for i in data]
        return data

    def load_data_folder(self):
        """
            Load data from *.json files in the specified folder and its subdirectories.
        """
        file_list, data = [], []
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.json'):
                    file_list.append(os.path.join(root, file))
        for file_path in file_list:
            with open(file_path, 'r', encoding='utf-8') as file:
                data.append(json.load(file))
        return data

    def data_info(self):
        """
            Print basic information about the dataset
        """
        encoding_name = "cl100k_base"
        print(f'[info] number of math word problems: {len(self.data)}')
        print(f'[info] average number of tokens per question in the training set: {sum([self.num_tokens_from_string(question, encoding_name) for question in self.questions]) / len(self.data)}')

    def data_parse_math(self, data):
        self.data = data
        questions = [sub_data.get('problem') for sub_data in self.data]
        reasoning_paths = [sub_data.get('solution') for sub_data in self.data]
        levels = [sub_data.get('level') for sub_data in self.data]
        types = [sub_data.get('type') for sub_data in self.data]
        answers = [self.get_answer_math(reasoning_path) for reasoning_path in reasoning_paths]
        self.questions = questions
        self.reasoning_paths = reasoning_paths
        self.answers = answers
        self.levels = levels
        self.types = types
        return questions, reasoning_paths, answers, levels, types

    def get_answer_math(self, reasoning_path):
        """
            Extract the answer from the reasoning path in the MATH dataset.
        """
        ANS_RE = re.compile(r"\\boxed{(.*?)}\$")
        match = ANS_RE.search(reasoning_path)
        if match:
            match_str = match.group(1).strip()
            return match_str
        else:
            ANS_RE = re.compile(r"\\boxed{(.*?)}\.")
            match = ANS_RE.search(reasoning_path)
            if match:
                match_str = match.group(1).strip()
                return match_str
            else:
                return "[invalid]"

    def data_parse_gsm8k(self, data):
        self.data = data
        questions = [sub_data.get('question') for sub_data in self.data]
        reasoning_paths = [sub_data.get('answer') for sub_data in self.data]
        numerical_answers = [self.get_answer_gsm8k(reasoning_path) for reasoning_path in reasoning_paths]
        self.questions = questions
        self.reasoning_paths = reasoning_paths
        self.numerical_answers = numerical_answers
        return questions, reasoning_paths, numerical_answers

    def get_answer_gsm8k(self, reasoning_path):
        """
            Extract the numerical answer from the reasoning path in the GSM8K dataset.
        """
        ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
        match = ANS_RE.search(reasoning_path)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return eval(match_str)
        else:
            return "[invalid]"

    def num_tokens_from_string(self, string: str, encoding_name: str) -> int:
        """
            Return the number of tokens in a given string based on the specified encoding.
        """
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
