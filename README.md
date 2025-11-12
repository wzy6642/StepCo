# StepCo

## Steps to Run the Code
### 1. Construct Process Supervision Dataset

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
