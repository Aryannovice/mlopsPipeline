from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "google/gemma-3-4b-it"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("MODEL LOADED SUCCESSFULLY")
