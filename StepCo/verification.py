import warnings
warnings.filterwarnings('ignore')

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch

from config import Config


## load verification model
config = Config()
good_token = '+'
bad_token = '-'
step_tag = 'ки'
device = torch.device("cuda")
tokenizer = AutoTokenizer.from_pretrained(config.verification_model, cache_dir=config.verification_model_cache_dir)
candidate_tokens = tokenizer.encode(f"{good_token} {bad_token}")[1:]  # [648, 387]
step_tag_id = tokenizer.encode(f"{step_tag}")[-1]  # 12902
model = AutoModelForCausalLM.from_pretrained(config.verification_model, cache_dir=config.verification_model_cache_dir, device_map="auto").eval()


def step_verify_score(input_seq):
    input_id = torch.tensor([tokenizer.encode(input_seq)]).to(device)
    with torch.no_grad():
        logits = model(input_id).logits[:, :, candidate_tokens]
        scores = logits.softmax(dim=-1)[:, :, 0]
        step_scores = scores[input_id == step_tag_id]

    return step_scores.cpu().detach().numpy().tolist()
