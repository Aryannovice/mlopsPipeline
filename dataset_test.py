from datasets import load_dataset

dataset = load_dataset(
    "osunlp/Multimodal-Mind2Web",
    split="train[:10]"
)

print(dataset[0])
