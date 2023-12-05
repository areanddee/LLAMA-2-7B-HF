#
# Example credit: https://huggingface.co/docs/transformers/v4.35.2/en/main_classes/pipelines#transformers.pipeline
# Text classification test using default HF Distilbert model: https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english
# Batch size doubles from 1 to 128, over a synthetic dataset of 5000 simulated prompts.
#

from transformers import pipeline
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from transformers import AutoTokenizer
import torch
import time

model = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model)

class MyDataset(Dataset):
    def __len__(self):
        return 5000

    def __getitem__(self, i):
        return "This is a test"

device = "cuda"

if device == "cuda":
   print(f"going down GPU pipe..")
   pipeline = pipeline(
       "text-generation",
       model=model,
       torch_dtype=torch.float16,
       device_map="auto") # if you have GPU
else:
   torch.set_num_threads(16)
   print(f"going down CPU pipe..")
   pipeline = pipeline(
       "text-generation",
       model=model,
       torch_dtype=torch.float32) #if you have CPU

dataset = MyDataset()

for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    print("-" * 30)
    print(f"Streaming batch_size={batch_size}")
    for out in tqdm(pipeline(dataset, batch_size=batch_size), total=len(dataset)):
        pass
