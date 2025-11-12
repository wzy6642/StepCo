import warnings
warnings.filterwarnings('ignore')


class Config:
    def __init__(self):
        self.dataset_name_list = ['SVAMP', 'SingleOp', 'SingleEq', 'MultiArith', 'MATH', 'GSM-ICM', 'GSM-IC2', 'GSM8K', 'AQuA', 'AddSub', 'ASDiv']
        # The root directory where datasets are stored
        self.dataset_root_path = "xxx"
        # The prompt strategy used for the model
        self.prompt_strategy = "Stepwise-Correction"

        # Backend model name (large language model) to be used
        self.backend_LLM = "gpt-4o"
        self.openai_LLM_base_url = "xxx"
        self.openai_LLM_api_key = "xxx"
        self.max_new_tokens = 2048
        self.temperature = 0.7
        self.top_p = 0.95
        self.top_k = 0

        # Directory to cache the verification models
        self.verification_model_cache_dir = 'xxx'
        # Name of the verification model to be used
        self.verification_model = 'peiyi9979/math-shepherd-mistral-7b-prm'
        # Root directory where results are saved
        self.result_save_root_path = 'xxx'

        self.threshold = 0.5
        self.max_iterations = 5

