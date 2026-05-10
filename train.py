from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "google/gemma-3-4b-it",
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
)

dataset = load_dataset(
    "osunlp/Multimodal-Mind2Web",
    split = "train[:100]"
)
def formatting_func(example):
    texts = [
        f"Task: {task}\nAction: {action}"
        for task, action in zip(
            example["confirmed_task"],
            example["action_reprs"]
        )
    ]
    return texts
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
formatting_func = formatting_func,
    max_seq_length = max_seq_length,
    args = TrainingArguments(        
per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 10,
        learning_rate = 2e-4,
        fp16 = False,
bf16  = True,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        output_dir = "outputs",
    ),
)

trainer.train()

model.save_pretrained("gemma_desktop_adapter")
tokenizer.save_pretrained("gemma_desktop_adapter")
