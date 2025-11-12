import warnings
warnings.filterwarnings('ignore')

from openai_response import answered_by_openai
# from verification import step_verify_score
from prompt_template import zero_shot_cot_prompt_template, get_numerical_answer_prompt_template, stepwise_rectify_prompt_template_v2
from utils import get_reasoning_steps, find_first_smaller_index, post_process_value
from config import Config
step_tag = 'ки'
config = Config()


def initialization(problem, record, verifier):
    input_str = zero_shot_cot_prompt_template.format(**{'instruction': "\n", 'question': problem})
    initial_reasoning_path = answered_by_openai(input_str)
    record["iter-0"] = {}
    record["iter-0"]["reasoning_path"] = initial_reasoning_path
    output_supervised_verifier = verifier(f"Q: {problem} \n A: {initial_reasoning_path} {step_tag}")
    record["iter-0"]["OSV"] = output_supervised_verifier

    input_str = get_numerical_answer_prompt_template.format(**{' ': '', 'question': problem, 'reasoning_path': initial_reasoning_path})
    initial_answer = answered_by_openai(input_str)
    record["iter-0"]["answer"] = initial_answer
    return initial_answer


def rectification(problem, record, verifier, num_iter):
    reasoning_steps = get_reasoning_steps(record.get(f'iter-{num_iter-1}').get('reasoning_path'))
    reasoning_steps_with_tag = '\n'.join([step + f' {step_tag}' for step in reasoning_steps if len(step) >= 5])
    process_supervised_verifier = verifier(f"Q: {record['problem']} \n A: {reasoning_steps_with_tag}")
    first_incorrect_step_idx = find_first_smaller_index(process_supervised_verifier, config.threshold)
    # record[f'iter-{num_iter - 1}']['reasoning_path'] = '\n'.join([step for step in reasoning_steps if len(step) >= 5])

    if first_incorrect_step_idx == 0:
        return record[f'iter-{num_iter-1}']["answer"]
    else:
        record[f'iter-{num_iter}'] = {}
        record[f'iter-{num_iter-1}']["PSV"] = process_supervised_verifier
        input_str = stepwise_rectify_prompt_template_v2.format(
            **{'question': record['problem'], 'reasoning_path': record.get(f'iter-{num_iter-1}').get('reasoning_path'),
               'step_index': first_incorrect_step_idx, 'probability': format(process_supervised_verifier[first_incorrect_step_idx-1]*100, '.2f')})
        rectified_reasoning_path = answered_by_openai(input_str)
        record[f'iter-{num_iter}']["reasoning_path"] = rectified_reasoning_path
        output_supervised_verifier = verifier(f"Q: {problem} \n A: {rectified_reasoning_path} {step_tag}")
        record[f'iter-{num_iter}']["OSV"] = output_supervised_verifier

        input_str = get_numerical_answer_prompt_template.format(**{' ': '', 'question': problem, 'reasoning_path': rectified_reasoning_path})
        rectified_answer = answered_by_openai(input_str)
        record[f'iter-{num_iter}']["answer"] = rectified_answer
        return rectified_answer


def pipeline(record, verifier):
    answer_record = []
    answer_record_str = []
    problem = record['problem']
    answer = initialization(problem, record, verifier)
    answer_record.append(post_process_value(answer))
    answer_record_str.append(answer)
    if record.get("iter-0").get("OSV")[0] >= config.threshold:
        return answer, record
    else:
        print("[INFO] Initial answer can be incorrect")
        for num_iter in range(config.max_iterations):
            answer = rectification(problem, record, verifier, num_iter+1)
            answer_record.append(post_process_value(answer))
            answer_record_str.append(answer)
            if record.get(f"iter-{num_iter+1}") != None and  record.get(f"iter-{num_iter+1}").get("OSV")[0] >= config.threshold:
                return answer, record
            if len(answer_record)>=2:
                if answer_record[-1]==answer_record[-2]:
                    return answer_record_str[-1], record
        return answer_record_str[-1], record
