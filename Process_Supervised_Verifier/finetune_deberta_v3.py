import warnings
warnings.filterwarnings('ignore')
import os
import random

from transformers import HfArgumentParser, TrainingArguments, set_seed, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, EvalPrediction, default_data_collator, Trainer
from transformers.trainer_utils import get_last_checkpoint
from datasets import Dataset
import pandas as pd
import evaluate

from config import Config, DataTrainingArguments, ModelArguments, logging_config
from utils import read_json_file


## model config
my_config = Config()
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
logger = logging_config(training_args)


## Detecting last checkpoint.
last_checkpoint = None
if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )
    elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(
            f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
            "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
        )
# Set seed before initializing model.
set_seed(training_args.seed)


## load model
num_labels = 1
# Load pretrained model and tokenizer
# In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
# download model & vocab.
config = AutoConfig.from_pretrained(
    model_args.model_name_or_path,
    num_labels=num_labels,
    finetuning_task="text-classification",
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    token=model_args.token,
    trust_remote_code=model_args.trust_remote_code,
)
config.problem_type = "regression"
logger.info("setting problem type to regression")
tokenizer = AutoTokenizer.from_pretrained(
    model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    use_fast=model_args.use_fast_tokenizer,
    revision=model_args.model_revision,
    token=model_args.token,
    trust_remote_code=model_args.trust_remote_code,
)
model = AutoModelForSequenceClassification.from_pretrained(
    model_args.model_name_or_path,
    from_tf=bool(".ckpt" in model_args.model_name_or_path),
    config=config,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    token=model_args.token,
    trust_remote_code=model_args.trust_remote_code,
    ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
)
padding = "max_length"


## load data
gsm8k_dataset_path = os.path.join(my_config.dataset_root_path, f'GSM8K_StepVerify_train.json')
gsm8k_train_data = read_json_file(gsm8k_dataset_path)
math_dataset_path = os.path.join(my_config.dataset_root_path, f'MATH_StepVerify_train.json')
math_train_data = read_json_file(math_dataset_path)
train_data = gsm8k_train_data + math_train_data

gsm8k_dataset_path = os.path.join(my_config.dataset_root_path, f'GSM8K_StepVerify_test.json')
gsm8k_test_data = read_json_file(gsm8k_dataset_path)
math_dataset_path = os.path.join(my_config.dataset_root_path, f'MATH_StepVerify_test.json')
math_test_data = read_json_file(math_dataset_path)
test_data = gsm8k_test_data + math_test_data
train_data = train_data + test_data

reasoning_steps = [str(sub_data.get('reasoning_path')) for sub_data in train_data]
labels = [float(sub_data.get('accuracy')) for sub_data in train_data]
train_datasets = Dataset.from_pandas(pd.DataFrame({'reasoning_step': reasoning_steps, 'label': labels}))
max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

def preprocess_function(examples):
    # Tokenize the texts
    result = tokenizer(examples["reasoning_step"], padding=padding, max_length=max_seq_length, truncation=True)
    return result

# Running the preprocessing pipeline on all the datasets
with training_args.main_process_first(desc="dataset map pre-processing"):
    train_dataset = train_datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )
# Log a few random samples from the training set:
if training_args.do_train:
    if data_args.shuffle_train_dataset:
        logger.info("Shuffling the training dataset")
        train_dataset = train_dataset.shuffle(seed=data_args.shuffle_seed)
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")


metric = evaluate.load("mse", cache_dir=model_args.cache_dir)
logger.info("Using mean squared error (mse) as regression score, you can use --metric_name to overwrite.")
def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds)
    result = metric.compute(predictions=preds, references=p.label_ids)
    if len(result) > 1:
        result["combined_score"] = np.mean(list(result.values())).item()
    return result


data_collator = default_data_collator
# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=None,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Training
if training_args.do_train:
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    max_train_samples = len(train_dataset)
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))
    trainer.save_model()  # Saves the tokenizer too for easy upload
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
if training_args.push_to_hub:
    trainer.push_to_hub(**kwargs)
else:
    trainer.create_model_card(**kwargs)
