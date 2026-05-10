import mlflow
import mlflow.transformers

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch
mlflow.set_tracking_uri("databricks")
model_path = "./merged_gemma_model"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print("Starting MLflow run...")

mlflow.set_experiment("/Shared/gemma_desktop_agent")

with mlflow.start_run():

    mlflow.log_param("base_model", "gemma-3-4b-it")
    mlflow.log_param("fine_tuning", "QLoRA")
    mlflow.log_param("dataset", "Multimodal-Mind2Web")

    print("Logging model to MLflow...")

    mlflow.transformers.log_model(
        transformers_model={
            "model": model,
            "tokenizer": tokenizer,
        },
        artifact_path="gemma_model",
task = "text-generation"
    )

    print("DONE LOGGING!")
