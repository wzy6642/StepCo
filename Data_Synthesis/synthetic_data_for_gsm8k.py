import warnings
warnings.filterwarnings('ignore')
import os
import re
import json

from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from pprint import pprint
from modelscope import snapshot_download
import torch
import transformers
import Levenshtein

from config import Config
from data_loader import LoadData
from utils import load_txt_data, answered_by_llama3_8b
from prompt_template import gsm8k_correct_reasoning_path_prompt_template, gsm8k_incorrect_reasoning_path_prompt_template, auto_cot_gsm8k_prompt_template, direct_gsm8k_prompt_template


## load GSM8K training dataset
config = Config()
dataset_name = 'GSM8K'
dataset_path = os.path.join(config.dataset_root_path, f'{dataset_name}_train.jsonl')
data_loder = LoadData(dataset_path)
# data = data_loder.load_data_jsonl()[config.begin_idx:config.end_idx]
data = data_loder.load_data_jsonl()
questions, reasoning_paths, answers = data_loder.data_parse_gsm8k(data)
data_loder.data_info()
save_path = os.path.join(config.data_save_root_path, f'{dataset_name}_train.txt')     
print(f'[info] synthetic data saved to {save_path}')


## loading large language model
huggingface_name = config.LLM_config.get(config.backend_LLM)
cache_dir = config.LLM_cache_dir
model_id = snapshot_download(huggingface_name, cache_dir=cache_dir)
transformers_pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)
terminators = [
    transformers_pipeline.tokenizer.eos_token_id,
    transformers_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]


correct_prompt_template = [gsm8k_correct_reasoning_path_prompt_template, auto_cot_gsm8k_prompt_template]
correct_incorrect_prompt_template = [gsm8k_correct_reasoning_path_prompt_template, gsm8k_incorrect_reasoning_path_prompt_template]

## generate synthetic dataset
if not os.path.exists(save_path):
    add_idx = 0
else:
    solved_questions = load_txt_data(save_path)[-1]
    add_idx = solved_questions.get('index')+1
for index in tqdm(range(len(data)), desc=f'{dataset_name} {config.backend_LLM} data synthetic'):
    index += add_idx
    if answers[index] != "[invalid]":
        record = {}
        step_idx = 0
        question = questions[index]
        answer = answers[index]
        record['gold_answer'] = answer
        record['Q'] = {
            'tag': 'Q',
            'identifier': 'Q',
            'parent': 'None',
            'data': question,
            'type': 'root_node'
        }
        for depth in range(config.max_depth):
            if step_idx == 0:
                step_record = {}
                for i in range(config.repetition_number):
                    instruction = """"""
                    input_to_LLM = correct_prompt_template[i].format(**{'question': question, 'instruction': instruction})
                    initial_reasoning_path = answered_by_llama3_8b(
                        transformers_pipeline,
                        terminators,
                        input_to_LLM,
                        config.max_new_tokens,
                        config.top_p,
                        config.temperature
                    )
                    print(f'[info] Initial reasoning path: {initial_reasoning_path}')
                    try:
                        # <Step 1> ... </Step 1>
                        # </Step 1> ... </Step 1>
                        pattern = f'</?Step {step_idx + 1}>(.*?)</Step {step_idx + 1}>'
                        init_step = re.findall(pattern, initial_reasoning_path, re.DOTALL)[0].strip()
                    except Exception as e:
                        # <Step 1> ... <Step 2>
                        if f'<Step {step_idx + 1}>' in initial_reasoning_path:
                            pattern = f'<Step {step_idx + 1}>(.*?)<Step {step_idx + 2}>'
                            init_step = re.findall(pattern, initial_reasoning_path, re.DOTALL)[0].strip()
                        # Step 1: ... Step 2:
                        elif f'Step {step_idx + 1}:' in initial_reasoning_path:
                            pattern = f'Step {step_idx + 1}:(.*?)Step {step_idx + 2}:'
                            init_step = re.findall(pattern, initial_reasoning_path, re.DOTALL)[0].strip()
                    step_name = f'S{step_idx + 1}{i + 1}'
                    print(f'[INFO] Step {step_name}: {init_step}')
                    record[step_name] = {
                        'tag': step_name,
                        'identifier': step_name,
                        'parent': 'Q',
                        'data': init_step,
                        'type': 'child_node'
                    }
                    step_record[step_name] = init_step
                step_info = list(step_record.values())
                if len(step_info)==2:
                    if Levenshtein.distance(step_info[0],step_info[1])<=config.max_levenshtein_dis:
                        record.pop(list(step_record.keys())[1])
                step_idx += 1
            else:
                for parent_node_name in list([key for key in list(record.keys()) if key.count('S') == step_idx]):
                    pre_reasoning_steps_id = ['S' + step_name for step_name in parent_node_name.split('S') if step_name != '']      # S11, S21, S31
                    pre_reasoning_steps_id = [''.join(pre_reasoning_steps_id[:i + 1]) for i in range(len(pre_reasoning_steps_id))]  # S11, S11S21, S11S21S31
                    pre_reasoning_steps_info = [record.get(step_id).get('data') for step_id in pre_reasoning_steps_id]
                    pre_reasoning_steps_type = [record.get(step_id).get('type') for step_id in pre_reasoning_steps_id]
                    if 'S11' in parent_node_name:
                        print('[info] using correct reasoning paths as demonstrations')
                        prompt_template = correct_prompt_template
                    else:
                        print('[info] using incorrect reasoning paths as demonstrations')
                        prompt_template = correct_incorrect_prompt_template
                    if 'leaf_node' not in pre_reasoning_steps_type:
                        step_record = {}
                        for i in range(config.repetition_number):
                            if i == 0:
                                instruction = """\nPlease generate the next reasoning step based on the math word problem below and the reasoning steps already generated.\n"""
                            else:
                                instruction = """\nPlease generate the next incorrect reasoning step based on the math word problem below and the reasoning steps already generated.\n"""
                            input_to_LLM = prompt_template[i].format(**{'question': question, 'instruction': instruction})
                            existing_solved_steps = ''
                            for pre_step_idx in range(len(pre_reasoning_steps_info)):
                                existing_solved_steps += f'<Step {pre_step_idx + 1}> \n{pre_reasoning_steps_info[pre_step_idx]} \n</Step {pre_step_idx + 1}>\n\n'
                            input_to_LLM += existing_solved_steps
                            subsequent_reasoning_step = answered_by_llama3_8b(
                                transformers_pipeline,
                                terminators,
                                input_to_LLM,
                                config.max_new_tokens,
                                config.top_p,
                                config.temperature
                            )
                            if '<ans>' in existing_solved_steps:
                                pattern = r'<ans>(.*?)</ans>'
                                follow_step = re.findall(pattern, existing_solved_steps, re.DOTALL)[0].strip()
                                node_type = 'leaf_node'
                                step_name = f'{parent_node_name}S{step_idx + 1}{i + 1}'
                                print(f'[INFO] Step {step_name}: {follow_step}')
                                record[step_name] = {
                                    'tag': step_name,
                                    'identifier': step_name,
                                    'parent': parent_node_name,
                                    'data': follow_step,
                                    'type': node_type
                                }
                                break
                            else:
                                try:
                                    # <Step 2> ... </Step 2>
                                    # </Step 2> ... </Step 2>
                                    pattern = f'</?Step {step_idx + 1}>(.*?)</Step {step_idx + 1}>'
                                    follow_step = re.findall(pattern, subsequent_reasoning_step, re.DOTALL)[0].strip()
                                    node_type = 'child_node'
                                except Exception as e:
                                    # <Step 2> ... <Step 3>
                                    if f'<Step {step_idx + 1}>' in subsequent_reasoning_step and f'<Step {step_idx + 2}>' in subsequent_reasoning_step:
                                        pattern = f'<Step {step_idx + 1}>(.*?)<Step {step_idx + 2}>'
                                        follow_step = re.findall(pattern, subsequent_reasoning_step, re.DOTALL)[0].strip()
                                        node_type = 'child_node'
                                    # Step 2: ... Step 3:
                                    elif f'Step {step_idx + 1}:' in subsequent_reasoning_step and f'Step {step_idx + 2}:' in subsequent_reasoning_step:
                                        pattern = f'Step {step_idx + 1}:(.*?)Step {step_idx + 2}:'
                                        follow_step = re.findall(pattern, subsequent_reasoning_step, re.DOTALL)[0].strip()
                                        node_type = 'child_node'
                                    # <Step 2> ... <ans>(.*?)</ans>
                                    elif f'<Step {step_idx + 1}>' in subsequent_reasoning_step and f'<Step {step_idx + 2}>' not in subsequent_reasoning_step and '</ans>' in subsequent_reasoning_step:
                                        pattern = f'<Step {step_idx + 1}>(.*?)</ans>'
                                        follow_step = re.findall(pattern, subsequent_reasoning_step, re.DOTALL)[0].strip()
                                        follow_step = follow_step + '</ans>'
                                        node_type = 'child_node'
                                    else:
                                        if '<ans>' in subsequent_reasoning_step:
                                            pattern = r'<ans>(.*?)</ans>'
                                            follow_step = re.findall(pattern, subsequent_reasoning_step, re.DOTALL)[0].strip()
                                            node_type = 'leaf_node'
                                            step_name = f'{parent_node_name}S{step_idx + 1}{i + 1}'
                                            print(f'[INFO] Step {step_name}: {follow_step}')
                                            record[step_name] = {
                                                'tag': step_name,
                                                'identifier': step_name,
                                                'parent': parent_node_name,
                                                'data': follow_step,
                                                'type': node_type
                                            }
                                            break
                                        elif 'The answer is' in subsequent_reasoning_step:
                                            pattern = r'The answer is(.*?)\$\.'
                                            follow_step = re.findall(pattern, subsequent_reasoning_step, re.DOTALL | re.IGNORECASE)[0].strip()
                                            node_type = 'leaf_node'
                                            step_name = f'{parent_node_name}S{step_idx + 1}{i + 1}'
                                            print(f'[INFO] Step {step_name}: {follow_step}')
                                            record[step_name] = {
                                                'tag': step_name,
                                                'identifier': step_name,
                                                'parent': parent_node_name,
                                                'data': follow_step,
                                                'type': node_type
                                            }
                                            break
                                step_name = f'{parent_node_name}S{step_idx + 1}{i + 1}'
                                print(f'[INFO] Step {step_name}: {follow_step}')
                                record[step_name] = {
                                    'tag': step_name,
                                    'identifier': step_name,
                                    'parent': parent_node_name,
                                    'data': follow_step,
                                    'type': node_type
                                }
                                step_record[step_name] = follow_step
                            step_info = list(step_record.values())
                            # print(step_info)
                            if len(step_info) == 2:
                                if Levenshtein.distance(step_info[0], step_info[1]) <= config.max_levenshtein_dis:
                                    record.pop(list(step_record.keys())[1])
                step_idx += 1
        record['index'] = index
        with open(save_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


