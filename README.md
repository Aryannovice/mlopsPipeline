# Gemma 3 LLMOps Pipeline with MLflow and Databricks

## Overview

This project demonstrates an end-to-end LLMOps workflow using Google Gemma 3, QLoRA fine-tuning, MLflow experiment tracking, and Databricks Unity Catalog model registration.

The workflow includes:

- Cloud-based fine-tuning on AWS EC2 GPU
- QLoRA optimization using Unsloth
- LoRA adapter merging
- Inference validation
- MLflow experiment tracking
- Remote Databricks logging
- Unity Catalog model registration

---

# Architecture

```text
Local Laptop
    ↓ SSH
AWS EC2 GPU
    ↓
Gemma 3 QLoRA Fine-Tuning
    ↓
Merged Deployable Model
    ↓
MLflow Tracking
    ↓
Databricks Remote Experiment Logging
    ↓
Unity Catalog Model Registry
```

---

# Tech Stack

- Python
- Unsloth
- Hugging Face Transformers
- PEFT / QLoRA
- TRL
- MLflow
- Databricks
- AWS EC2
- PyTorch

---

# Model Information

| Component | Value |
|---|---|
| Base Model | google/gemma-3-4b-it |
| Fine-Tuning Method | QLoRA |
| Quantization | 4-bit |
| Dataset | osunlp/Multimodal-Mind2Web |
| Framework | Unsloth + TRL |
| Tracking | MLflow |
| Registry | Databricks Unity Catalog |

---

# Fine-Tuning Workflow

The model was fine-tuned using:

- LoRA adapters
- 4-bit quantization
- Gradient checkpointing
- PEFT optimization

Training was performed on an AWS EC2 GPU instance.

---

# MLflow Integration

MLflow was used for:

- Experiment tracking
- Parameter logging
- Model artifact logging
- Run management
- Databricks integration

---

# Databricks Integration

The trained model was:

- Logged remotely to Databricks MLflow
- Registered into Unity Catalog
- Versioned using Model Registry

Registered Model:

```text
workspace.default.gemma_desktop_agent
```

---

# Project Structure

```text
desktop_agent/
│
├── train.py
├── infer.py
├── merge_model.py
├── mlflow_log_model.py
├── register_model.py
├── requirements.txt
├── README.md
└── .gitignore
└── outputs/
```

---

# Setup

## Clone Repository

```bash
git clone <repo_url>
cd desktop_agent
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Training

```bash
python train.py
```

---

# Merge LoRA Adapters

```bash
python merge_model.py
```

---

# Run Inference

```bash
python infer.py
```

---

# MLflow Logging

```bash
python mlflow_log_model.py
```

---

# Model Registration

```bash
python register_model.py
```

---



---

# Screenshots

I have added an outputs folder and saved all screenshots there .

---

# Author

Ayush Pandey
