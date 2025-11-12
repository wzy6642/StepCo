import warnings
warnings.filterwarnings('ignore')
import json


def read_json_file(data_path):
    with open(data_path) as jsonfile:
        data = json.load(jsonfile)
    return data


def write_list_to_json(data, save_path):
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)