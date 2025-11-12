import warnings
warnings.filterwarnings('ignore')
import math
import json

import pandas as pd


def load_txt_data(path):
    with open(path, 'r', encoding='gb18030', errors='ignore') as f:
        data = f.readlines()
    all_data = list()
    for idx in range(len(data)):
        try:
            sub_data = eval(data[idx])
            all_data.append(sub_data)
        except Exception as e:
            print(f'[info] format error: {idx}')
    # data = [eval(sub_data) for sub_data in data]
    return all_data


def write_list_to_json(data, save_path):
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)


def read_json_file(save_path):
    data = pd.read_json(save_path)
    return data


def post_process_value(generate_answer, location=-1):
    try:
        generate_answer = generate_answer.replace(',', '')  # 10,329 -> 10329
        generate_answer = ''.join(char for char in generate_answer if not char.isalpha())  # 2520 students -> 2520
        generate_answer = ''.join(char for char in generate_answer if char not in ['(', ')'])  # (12) -> 12
        generate_answer = generate_answer.strip()  # '2520 ' -> '2520'
        if type(generate_answer) == str and len(generate_answer) >= 1 and generate_answer[-1] == '.':  # 27. -> 27
            generate_answer = generate_answer[:-1]
        generate_answer = generate_answer.strip()
        if ' ' in generate_answer:  # 22 42 -> 42
            generate_answer = generate_answer.split(' ')[location]
        if type(generate_answer) == str and len(generate_answer) >= 1:  # '' -> 0
            pass
        else:
            generate_answer = '0'
        if generate_answer in ['-', '=', '+', '.']:  # - -> 0
            generate_answer = '0'
        # generate_answer = generate_answer.replace('</>', '')
        # generate_answer = generate_answer.replace('$', '')
        # generate_answer = generate_answer.replace('<>', '').replace('=', '')
        if type(generate_answer) == str and '%' in generate_answer:  # '20%' -> 0.2
            generate_answer = float(generate_answer.rstrip('%')) / 100
        if type(generate_answer) == str and ':' in generate_answer:  # 3:00 -> 3
            generate_answer = generate_answer.replace(':', '.')
        if type(generate_answer) == str and len(generate_answer) >= 1 and generate_answer[-1] in ['.', '/']:  # 27. -> 27
            generate_answer = generate_answer[:-1]
        if type(generate_answer) == str:
            generate_answer = generate_answer.replace('</>', '').replace('\\', '').replace('£', '').replace(':', '')
            generate_answer = generate_answer.replace('$', '')
            generate_answer = generate_answer.replace('<>', '').replace('=', '').replace('?', '').replace('\'', '')
            if len(generate_answer) == 0 or generate_answer == '.':
                generate_answer = '0'
            if generate_answer[-1] == '.':
                generate_answer = generate_answer[:-1]
            if len(generate_answer) >= 2 and generate_answer[0] == '0':
                generate_answer = generate_answer[1:]
            generate_answer = eval(generate_answer)
    except Exception as e:
        generate_answer = 0
    return generate_answer


def check_answer_correct_gsm8k(gold, pred):
    if type(gold) == str:
        gold = gold.replace(',', '')
        gold = eval(gold)
    if type(pred) == str:
        pred = pred.replace(',', '')
        try:
            pred = eval(pred)
        except Exception as e:
            pred = 0
    if math.isclose(gold, pred, rel_tol=1e-5, abs_tol=1e-5):
        return True
    elif math.isclose(gold / 100, pred, rel_tol=1e-5, abs_tol=1e-5):
        return True
    elif math.isclose(gold, pred / 100, rel_tol=1e-5, abs_tol=1e-5):
        return True
    else:
        return False


def answered_by_llama3_8b(pipeline, terminators, input, max_new_tokens, temperature, top_p):
    messages = [
        {"role": "user", "content": input}
    ]
    prompt = pipeline.tokenizer.apply_chat_template(
    		messages,
    		tokenize=False,
    		add_generation_prompt=True
    )
    outputs = pipeline(
        prompt,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )
    response = outputs[0]["generated_text"][len(prompt):]
    return response


