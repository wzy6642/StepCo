# StepCo

We propose StepCo, a novel framework that uses an iterative verify-then-revise process to progressively identify and revise incorrect steps in LLM-generated reasoning paths.

## Steps to Run the Code
### 1. Constructing Process Supervision Dataset

First, you need to switch to the Data_Synthesis directory. To generate the problem-solving steps, run the following command, which will use GPU 0 by default:
```Python
CUDA_VISIBLE_DEVICES=0 python run.py
```
After generating the problem-solving steps, run the following command to compute the probability of getting the correct answer for each step:
```Python
python tree_construction.py
```
This will output the probability associated with each reasoning step.
> Note: You will need to specify the paths for reading and saving datasets, as well as the model name and configuration for inference steps.

### 2. Training Process Supervised Verifier

Then, you need to switch to the Process_Supervised_Verifier directory. Run the fine-tuning script using the following command:
```Python
CUDA_VISIBLE_DEVICES=0,1,2 python finetune_deberta_v3.py \
--model_name_or_path microsoft/deberta-v3-large \
--cache_dir xxx \
--output_dir xxx \
--shuffle_train_dataset \
--do_train \
--max_seq_length 256 \
--per_device_train_batch_size 16 \
--learning_rate 2e-6 \
--num_train_epochs 100 \
--pad_to_max_length \
--save_steps 500 \
--gradient_accumulation_steps 10
```
> Note: Before running the code, specify the paths for saving the model and loading the dataset.

### 3. Solving Math Word Problems

Finally, you need to switch to the StepCo directory. Run the following command to automatically solve the math word problem:
```Python
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_idx 0
```
> Note: You need to specify the root directory where your test dataset is stored. For interacting with OpenAI's API, you must configure your API keys and any additional required settings.
