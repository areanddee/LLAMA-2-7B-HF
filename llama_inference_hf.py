#
# Text generation pipelines...
#
# https://huggingface.co/docs/transformers/v4.35.2/en/main_classes/pipelines#transformers.pipeline
#

from transformers import AutoTokenizer
import transformers
import torch
import time

model_name = "meta-llama/Llama-2-7b-chat-hf"
model = .from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = "cuda"

if device == "cuda":
   print(f"going down GPU pipe..")
   pipeline = transformers.pipeline(
       "text-generation",
       model=model_name,
       tokenizer=tokenizer,
       torch_dtype=torch.float16,
       device_map="auto") # if you have GPU
else:
   torch.set_num_threads(16)
   print(f"going down CPU pipe..")
   pipeline = transformers.pipeline(
       "text-generation",
       model=model,
       torch_dtype=torch.float32) #if you have CPU

tic = time.perf_counter()
sequences = pipeline(
    ['I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?', 'Who was Albert Einstein?'],
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    pad_token_id = model.config.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_length=128,
    batch_size=2
)
toc = time.perf_counter()
print(f"Ran LLAMA pipeline on {device} in {toc - tic:0.4f} seconds")

for seq in sequences:
    print(f"Result: {seq['generated_text']}")
