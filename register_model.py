import mlflow

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

run_id = "68d261db057b4cb7b10bd3bf5909705c"

model_uri = f"runs:/{run_id}/gemma_model"

result = mlflow.register_model(
    model_uri=model_uri,
name="workspace.default.gemma_desktop_agent"
)

print("MODEL REGISTERED!")
print(result)
