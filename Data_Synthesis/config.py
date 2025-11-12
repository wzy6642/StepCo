import warnings
warnings.filterwarnings('ignore')

class Config:
    def __init__(self):
        self.dataset_name_list = ['MATH', 'GSM8K']
        # Root directory for the datasets
        self.dataset_root_path = "xxx"
        # Name of the backend large model
        self.backend_LLM = "Llama-3-8B-Instruct"

        self.max_new_tokens = 512
        self.temperature = 0.7
        self.top_p = 0.98
        self.top_k = 0
        # Maximum Levenshtein distance between two strings (used for measuring string similarity)
        self.max_levenshtein_dis = 5
        # Maximum number of children per node in the tree (binary tree, so each node can have at most 2 children)
        self.repetition_number = 2
        # Maximum depth of the reasoning tree (i.e., the maximum number of reasoning steps a question can involve)
        self.max_depth = 10
        # Path where large models are stored
        self.LLM_cache_dir = 'xxx'
        # Mapping of large model names to their corresponding Huggingface model names
        self.LLM_config = {
            "Llama-3-8B-Instruct": "LLM-Research/Meta-Llama-3-8B-Instruct"
        }
        # Root directory for saving synthetic data
        self.data_save_root_path = 'xxx'
