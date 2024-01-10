#
# Example credit: https://huggingface.co/docs/transformers/v4.35.2/en/main_classes/pipelines#transformers.pipeline
# Text classification test using default HF Distilbert model: https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english
# Batch size doubles from 1 to 128, over a synthetic dataset of 5000 simulated prompts.
#

import textwrap
import pandas as pd
import numpy as np
from transformers import pipeline
from torch.utils.data import Dataset
from datasets import load_dataset
from tqdm.auto import tqdm

from transformers import AutoTokenizer
from transformers import  LlamaForCausalLM
import torch
import time

num_questions=100
model_name="meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name) # use_auth_token=True)

dataset = load_dataset('gsm8k', 'main') #See: https://huggingface.co/datasets/gsm8k for details on gsm8k dataset
questions = dataset['test']['question'][:num_questions]

# Function to format prompt according to Llama2 expected chat instruction format
def format_prompt(prompt: str) -> str:
    llama_template = textwrap.dedent(f"""\
    <s>[INST] <<SYS>>You provide just the answer you are asked for with no preamble. Do not repeat the question. Be succinct.<</SYS>>
 
    {prompt} [/INST]
    """)
 
    return llama_template
 
# Add the Llama2 instruction format to each prompt
formatted_prompts = [format_prompt(q) for q in questions]
  
# Convert the instructions to a DataFrame format
instructions = pd.DataFrame(data={'text': formatted_prompts})

# Print a random sample question and formatted instruction
print("-" * 30)
random_idx = instructions.sample(n=1).index[0]
print(f"Random question:\n```{questions[random_idx]}```\n\n")
print(f"Instruction:\n```{instructions.loc[random_idx]['text']}```")
print("-" * 30)

device = "cuda"

if device == "cuda":
   print(f"going down GPU pipe..")
   pipeline = pipeline(
       "text-generation",
       model=model_name,
       torch_dtype=torch.float16,
       device_map="auto") # if you have GPU
   pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id
   pipeline.tokenizer.padding_side='left'
else:
   torch.set_num_threads(16)
   print(f"going down CPU pipe..")
   pipeline = pipeline(
       "text-generation",
       model=model,
       torch_dtype=torch.float32) #if you have CPU

repetitions=5
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
timings=np.zeros((repetitions,1))
for batch_size in [1, 5, 10, 20]:
    print("-" * 30)
    print(f"Streaming batch_size={batch_size}")
#    for out in tqdm(pipeline(formatted_prompts, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, batch_size=batch_size, max_new_tokens=50), total=20):
    for rep in range(repetitions):
        starter.record()
        sequences = pipeline(formatted_prompts, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, batch_size=batch_size, min_new_tokens=20, max_new_tokens=25)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time
        print(f"rep={rep} timing = {timings[rep]}")
        if rep == 0:
           for i, seq in enumerate(sequences):
                print(f"Q: {questions[i]}")
                print(f"A: {seq}")
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
#for rep in range(repetitions):
#    print(f"timings[{rep}] = {timings[rep]}")
    print(f"mean time = {mean_syn} +/- {std_syn} milliseconds")
