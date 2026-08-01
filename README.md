# Foundation LLM — 50M Parameter GPT Model
By Ramprasadh

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ramprasadh/foundation-llm/blob/main/Foundation_LLM_50M.ipynb)

A complete, lightweight, end-to-end framework to build, pretrain, evaluate, and generate text with a **50M-parameter decoder-only GPT model** from scratch using PyTorch. Optimized for Google Colab free T4 GPU runtimes and local environments.

---

## Architecture Overview

| Parameter | Value |
|---|---|
| **Parameters** | ~50 Million |
| **Layers (`n_layer`)** | 8 |
| **Attention Heads (`n_head`)** | 8 |
| **Embedding Dimension (`n_embd`)** | 512 |
| **Context Length (`block_size`)** | 1024 tokens |
| **Vocabulary Size (`vocab_size`)** | 50,304 (tiktoken `gpt2`) |
| **Precision** | Mixed Precision (`fp16` / `autocast`) |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.95$, weight decay $0.1$) |
| **Learning Rate Schedule** | Cosine decay with warmup |

---

## Directory Structure

```text
foundation-llm/
├── config/
│   └── model_50m_colab.yaml       # Hyperparameters & path configurations
├── data/
│   ├── prepare.py                 # Tokenization script (sample & OpenWebText)
│   └── sample_corpus.txt          # Default local text dataset
├── src/
│   ├── __init__.py
│   ├── dataset.py                 # Memory-mapped binary token dataset loader
│   ├── model.py                   # GPT Transformer architecture (~50M params)
│   ├── train.py                   # Main training loop with checkpointing
│   ├── evaluate.py                # Validation perplexity evaluator
│   └── generate.py                # Autoregressive text sampling script
├── test.py                        # OpenAI / API connection test script
├── build_notebook.py              # Script to build the Colab self-bootstrapping notebook
├── Foundation_LLM_50M.ipynb       # Generated self-contained Colab notebook
└── requirements.txt               # Dependencies list
```

---

## Step-by-Step Guide: Running on Google Colab (Free T4 GPU)

### Step 1: Open Google Colab
Go to **[Google Colab](https://colab.research.google.com/)**.

### Step 2: Upload the Notebook
1. Click **Upload** in the Colab start menu.
2. Select `Foundation_LLM_50M.ipynb` from this project folder.

### Step 3: Enable GPU Acceleration
1. Navigate to **Runtime → Change runtime type**.
2. Set **Hardware accelerator** to **GPU** (T4).
3. Click **Save**.

### Step 4: Execute the Notebook
1. Select **Runtime → Run all**.
2. The notebook will automatically:
   - Check GPU status (`nvidia-smi`).
   - Mount Google Drive to `/content/drive/MyDrive/foundation-llm/`.
   - Scaffold all source files and directories.
   - Install dependencies (`requirements.txt`).
   - Run a 10-step training smoke test.
   - Prepare dataset (sample or OpenWebText subset).
   - Start full pretraining with automatic checkpointing to Drive.
   - Run interactive text generation.

---

## Step-by-Step Guide: Local Setup & Running

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Prepare Training Data

#### Option A: Use Sample Corpus
```bash
python data/prepare.py --sample
```

#### Option B: Stream OpenWebText Subset
```bash
python data/prepare.py --subset --max_docs 200000
```

### Step 3: Train the Model

#### Run Full Pretraining:
```bash
python src/train.py --config config/model_50m_colab.yaml
```

#### Run Quick Smoke Test (e.g., 50 steps):
```bash
python src/train.py --config config/model_50m_colab.yaml --max_steps 50
```

#### Auto-resume Training from Latest Checkpoint:
```bash
python src/train.py --config config/model_50m_colab.yaml --resume auto
```

### Step 4: Generate Text
```bash
python src/generate.py --checkpoint checkpoints/latest.pt --prompt "The history of AI" --max_tokens 150 --temperature 0.8
```

### Step 5: Evaluate Validation Perplexity
```bash
python src/evaluate.py --config config/model_50m_colab.yaml --checkpoint checkpoints/latest.pt
```

---

## Rebuilding the Google Colab Notebook

If you modify any source code in `src/`, `config/`, or `data/`, regenerate `Foundation_LLM_50M.ipynb`:

```bash
python build_notebook.py
```
