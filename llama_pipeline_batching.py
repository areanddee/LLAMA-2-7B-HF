#
# Example credit: https://huggingface.co/docs/transformers/v4.35.2/en/main_classes/pipelines#transformers.pipeline
# Text classification test using default HF Distilbert model: https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english
# Batch size doubles from 1 to 128, over a synthetic dataset of 5000 simulated prompts.
#

from transformers import pipeline
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from transformers import AutoTokenizer
from transformers import  LlamaForCausalLM
import torch
import time

model_name="meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

class MyDataset(Dataset):
    def __len__(self):
        return 559

    def __getitem__(self, i):
        return "This is a test"

device = "cuda"

if device == "cuda":
   print(f"going down GPU pipe..")
   pipeline = pipeline(
       "text-generation",
       model=model_name,
       torch_dtype=torch.float16,
       device_map="auto") # if you have GPU
   pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id
else:
   torch.set_num_threads(16)
   print(f"going down CPU pipe..")
   pipeline = pipeline(
       "text-generation",
       model=model,
       torch_dtype=torch.float32) #if you have CPU

dataset = MyDataset()

for batch_size in [1, 5, 10, 20]:
    print("-" * 30)
    print(f"Streaming batch_size={batch_size}")
    for out in tqdm(pipeline(dataset, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, batch_size=batch_size, max_new_tokens=50), total=20):
        pass
