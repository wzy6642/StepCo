import warnings
warnings.filterwarnings('ignore')
import os
import glob
import re
from pprint import pprint
from collections import Counter

from treelib import Node, Tree

from config import Config
from utils import load_txt_data, check_answer_correct_gsm8k, post_process_value, write_list_to_json, read_json_file


## load synthetic dataset
config = Config()
dataset_name = 'GSM8K'
reasoning_path_accuracy = []
accuracy_record = []
files_path = [file_path for file_path in glob.glob(config.data_save_root_path + '/*') if dataset_name+'_train' in file_path and 'txt' in file_path]
synthetic_data = [load_txt_data(file_path) for file_path in files_path]
synthetic_data = sum(synthetic_data, [])
synthetic_data = [{key:val for key, val in data.items() if key != 'index'} for data in synthetic_data if type(data)==dict]
save_path = os.path.join(config.data_save_root_path, dataset_name+'_StepVerify_train.json')


class AttributeNode(object):
    def __init__(self, detail, number_list):
        self.detail = detail
        self.number_list = number_list

def get_number_list(string):
    pattern = r'-?\d+(?:\.\d+)?'
    string = string.replace(',', '')
    number_list = re.findall(pattern, string)
    # if len(number_list)>=1:
    #     number_list = [eval(number) for number in number_list]
    return number_list


## construct the tree
for reasoning_path in synthetic_data:
    tree = Tree()
    for reasoning_step_name in reasoning_path:
        if type(reasoning_path.get(reasoning_step_name)) == dict:
            # 在树中逐个添加节点
            tree.create_node(
                tag = reasoning_path.get(reasoning_step_name).get('tag'), # type
                identifier = reasoning_path.get(reasoning_step_name).get('identifier'),
                parent = None if reasoning_path.get(reasoning_step_name).get('parent')=='None' else reasoning_path.get(reasoning_step_name).get('parent'),
                data = AttributeNode(
                    detail = reasoning_path.get(reasoning_step_name).get('data'),
                    number_list = get_number_list(reasoning_path.get(reasoning_step_name).get('data'))
                )
            )


    for key in list(reasoning_path.keys()):
        node_details = reasoning_path.get(key)
        if type(node_details)==dict:
            if node_details.get('type')!='root_node':
                step_name = key
                answer_list = [post_process_value(node.data.detail) for node in tree.leaves(step_name)]
                step_accuracy = len([answer for answer in answer_list if check_answer_correct_gsm8k(reasoning_path.get('gold_answer'), answer)]) / len(answer_list)
                step_info = ['S'+sub_step_name for sub_step_name in step_name.split('S') if sub_step_name!='']
                step_info = [''.join(step_info[:i+1]) for i in range(len(step_info))]
                # print(step_info)
                step_info = [tree.get_node(node_id).data.detail for node_id in step_info]
                for step_idx in range(len(step_info)):
                    step_info[step_idx] = f'<Step {step_idx+1}>\n{step_info[step_idx]}\n</Step {step_idx+1}>'
                step_info = f"Q: {tree.get_node('Q').data.detail}\nA: \n" + '\n\n'.join(step_info)
                # print(f'Accuracy of step {step_name}: {step_accuracy}')
                # print(step_info)
                reasoning_path_accuracy.append(
                    {
                        'reasoning_path': step_info,
                        'accuracy': step_accuracy
                    }
                )
                accuracy_record.append(step_accuracy)
print(len(reasoning_path_accuracy))
print(Counter(accuracy_record))


write_list_to_json(reasoning_path_accuracy, save_path)

