# LLAMA-2-7B-HF

Instructions for downloading, and running the pretrained HF llama2-chat model on a system with PyTorch and transformers....

For running llama2-chat inference, must have installed accelerate package to use GPUs:

>pip install accelerate

For pipeline batching example

Example credit: https://huggingface.co/docs/transformers/v4.35.2/en/main_classes/pipelines#transformers.pipeline
Text classification test using default HF Distilbert model: https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english
Batch size doubles from 1 to 128, over a synthetic dataset of 5000 simulated prompts.
Preconfigured to run on 1 GPU (device=0)#


