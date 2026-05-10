from unsloth import FastLanguageModel
from peft import PeftModel
from transformers import AutoTokenizer

base_model_name = "google/gemma-3-4b-it"
adapter_path = "gemma_desktop_adapter"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_name,
    max_seq_length=2048,
    load_in_4bit=False,
)

model = PeftModel.from_pretrained(
    model,
    adapter_path
)

merged_model = model.merge_and_unload()

merged_model.save_pretrained("merged_gemma_model")
tokenizer.save_pretrained("merged_gemma_model")
