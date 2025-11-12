import warnings
warnings.filterwarnings('ignore')
import os
import re


def load_txt_data(path):
    with open(path, 'r', encoding='gb18030', errors='ignore') as f:
        data = f.readlines()
    data = [eval(sub_data) for sub_data in data]
    return data


def num_data_in_txt_file(txt_file: str) -> int:
    """Returns the number of data points in a text file."""
    with open(txt_file, 'r', encoding='gb18030', errors='ignore') as f:
        data = f.readlines()
    return len(data)


def get_reasoning_steps(solution: str) -> list:
    steps = re.split(r'<Step(?: \d+)?>', solution)
    steps = [step.strip() for step in steps]
    pattern = r'<Step(?: \d+)?>'
    steps = [re.sub(pattern, '', step) for step in steps]
    pattern = r'</Step(?: \d+)?>'
    steps = [re.sub(pattern, '', step) for step in steps]
    for idx in range(len(steps)):
        if len(steps[0]) < 5:
            dec = 1
        else:
            dec = 0
        if idx != len(steps)-1 and len(steps[idx]) >=5:
            steps[idx] = f'<Step {idx+1-dec}> ' + steps[idx] + f' </Step {idx+1-dec}>'
    return steps


def find_first_smaller_index(arr, threshold):
    for i in range(len(arr)):
        if arr[i] < threshold:
            return i+1
    return 0


def post_process_value(generate_answer, location=-1):
    if type(generate_answer) == str:
        pattern = r'\\boxed{(.*?)}'
        answer = re.findall(pattern, generate_answer)
        if len(answer) != 0:
            generate_answer = answer[0]
            generate_answer = generate_answer.replace(',', '').replace('<', '').replace('>', '').replace('[', '').replace(']', '').replace('\\', '').replace('{', '').replace('}', '')  # 10,329 -> 10329
            generate_answer = generate_answer.replace('$', '')  # $10329 -> 10329
            generate_answer = generate_answer.replace('*', '')  # **8** -> 8

            # 去除答案中的英文字母
            generate_answer = ''.join(char for char in generate_answer if not char.isalpha())  # 2520 students -> 2520
            generate_answer = ''.join(char for char in generate_answer if char not in ['(', ')'])  # (12) -> 12
            if type(generate_answer) == str and ':' in generate_answer:                             # 3:00 -> 3
                generate_answer = generate_answer.replace(':', '.')
            if type(generate_answer) == str:
                generate_answer = generate_answer.strip()  # '2520 ' -> '2520'
            if len(generate_answer) > 1 and generate_answer[-1] == '.':  # 27. -> 27
                generate_answer = generate_answer[:-1]
            generate_answer = generate_answer.strip()
            if type(generate_answer) == str and '%' in generate_answer:                             # '20%' -> 0.2
                generate_answer = str(float(generate_answer.rstrip('%')) / 100)
            if ' ' in generate_answer:                                                              # 22 42 -> 42
                generate_answer = generate_answer.split(' ')[0]
            generate_answer = eval(generate_answer)

        else:
            # 特殊符号处理
            generate_answer = generate_answer.replace(',', '').replace('<', '').replace('>', '').replace('[', '').replace(']', '').replace('\\', '').replace('{', '').replace('}', '')  # 10,329 -> 10329
            generate_answer = generate_answer.replace('$', '')  # $10329 -> 10329
            generate_answer = generate_answer.replace('*', '')  # **8** -> 8
            generate_answer = ''.join(char for char in generate_answer if not char.isalpha())  # 2520 students -> 2520
            generate_answer = ''.join(char for char in generate_answer if char not in ['(', ')'])  # (12) -> 12
            generate_answer = generate_answer.strip()  # '2520 ' -> '2520'
            if len(generate_answer) > 1 and generate_answer[-1] == '.':  # 27. -> 27
                generate_answer = generate_answer[:-1]
            generate_answer = generate_answer.strip()

            if ' ' in generate_answer:                                                              # 22 42 -> 42
                generate_answer = generate_answer.split(' ')[location]
            if generate_answer in ['-', '=', '+', '\\', '.']:                                                       # - -> 0
                generate_answer = 0
            if type(generate_answer) == str and len(generate_answer) >= 1:                          # '' -> 0
                pass
            else:
                generate_answer = 0
            if type(generate_answer) == str and '%' in generate_answer:                             # '20%' -> 0.2
                generate_answer = float(generate_answer.rstrip('%')) / 100
            if type(generate_answer) == str and ':' in generate_answer:                             # 3:00 -> 3
                generate_answer = generate_answer.replace(':', '.')
            if type(generate_answer) == str and len(generate_answer) > 1 and generate_answer[-1] in ['.', '/']: # 27. -> 27
                generate_answer = generate_answer[:-1]
            if type(generate_answer) == str and len(generate_answer) > 1 and generate_answer[0] in ['.', '/']: # 27. -> 27
                generate_answer = generate_answer[1:]
                generate_answer = generate_answer.strip()
        if type(generate_answer) == str:
            generate_answer = eval(generate_answer)
    return generate_answer
