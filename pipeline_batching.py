#
# Example credit: https://huggingface.co/docs/transformers/v4.35.2/en/main_classes/pipelines#transformers.pipeline
# Text classification test using default HF Distilbert model: https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english
# Batch size doubles from 1 to 128, over a synthetic dataset of 5000 simulated prompts.
#

from transformers import pipeline
from torch.utils.data import Dataset
from tqdm.auto import tqdm

pipe = pipeline("text-classification", device=0)


class MyDataset(Dataset):
    def __len__(self):
        return 5000

    def __getitem__(self, i):
        return "This is a test"


dataset = MyDataset()

for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    print("-" * 30)
    print(f"Streaming batch_size={batch_size}")
    for out in tqdm(pipe(dataset, batch_size=batch_size), total=len(dataset)):
        pass
