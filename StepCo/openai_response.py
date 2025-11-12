import warnings
warnings.filterwarnings('ignore')

from openai import OpenAI
from config import Config


config = Config()
client = OpenAI(
    api_key=config.openai_LLM_api_key,
    base_url=config.openai_LLM_base_url
)


def check_string(s):
    if s == "" or s == "IP访问频率过高,请稍后再试":
        raise ValueError("Empty string encountered.")


def answered_by_openai(user_input):
    prompt = [
        {"role": "system", "content": user_input}
    ]
    
    response = client.chat.completions.create(
            model=config.backend_LLM,
            messages=prompt,
            stream=False,
            temperature=config.temperature,
            max_tokens=config.max_new_tokens,
            top_p=config.top_p,
            frequency_penalty=0,
            presence_penalty=0,
        )
    try:
        check_string(response)
    except Exception as e:
        print(e)
        sys.exit(1)
    response = dict(response)
    answer = response.get('choices')[0].message.content
    # completion_tokens = response.get('usage').completion_tokens
    # prompt_tokens = response.get('usage').prompt_tokens
    # total_tokens=response.get('usage').total_tokens
    return answer # , completion_tokens, prompt_tokens, total_tokens
