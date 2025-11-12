zero_shot_cot_prompt_template = """
You are an excellent math teacher and are able to solve a given math word problem step-by-step to get the correct answer.
Request: 
1. Mark the each reasoning step with <Step> </Step> tags.
For example: <Step 3> Calculate the amount of money Olivia has left.
Money left = Initial money - Total cost of bagels
Money left = $23 - $15 = $8 </Step 3>
2. Mark the final answer with <ans> </ans> tags.
For example: The answer is <ans>8</ans>.
{instruction}
Q: {question}
A: Let's think step by step.
"""


get_numerical_answer_prompt_template = """
Please generate the final answer for the given question based on the provided reasoning path. Make sure to enclose the final answer in '\\boxed{ }'.
Q: {question}
A: {reasoning_path}
Therefore, the answer (expressed in Arabic numerals and without units) is:
"""


stepwise_rectify_prompt_template = """
You are an intelligent student who is able to correct the reasoning path starting from a specified step pointed out by the teacher to ensure the result is accurate.

Q: {question}

A: {reasoning_path}

The step {step_index} might contain an error; therefore, rectify the reasoning steps starting from step {step_index} (including step {step_index}) onward to obtain the correct solution. The reasoning steps before step {step_index} should remain unchanged.

The rectified reasoning path is:
<Step 1>
"""


stepwise_rectify_prompt_template_v2 = """
You are an intelligent student who is able to correct the reasoning path starting from a specified step pointed out by the teacher to ensure the result is accurate.

Q: {question}

A: {reasoning_path}

The probability of obtaining the correct answer for step {step_index} is {probability}\% and may contain errors. You need to revise the reasoning steps starting from step {step_index} (including step {step_index}) onward to increase the probability of getting the correct answer. The reasoning steps before step {step_index} should remain unchanged.

The rectified reasoning path is:
<Step 1>
"""