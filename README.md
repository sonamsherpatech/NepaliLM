# NepaliLM

NepaliLM is a project designed to clean raw Nepali text data (from sources like Wikipedia, news sites, and CC-100), structure it into instruction-response pairs, and fine-tune a Large Language Model (e.g., Qwen 2.5) using QLoRA for Nepali language comprehension and generation.

---

## ⚠️ Project Status: Incomplete (Pending GPU Resources)

> [!IMPORTANT]  
> **This project is currently NOT complete.**  
> The final training and testing phase to build the fine-tuned model has been postponed due to a lack of sufficient local GPU infrastructure (e.g., the local 6GB VRAM GPU is insufficient to run a full 3B or 7B parameter model training run on the full dataset). 
> 
> **Future Action Plan**: The training pipeline is fully set up and ready to run. The final model will be trained and evaluated as soon as higher-end GPU resources (such as an A100 via RunPod or other cloud infrastructure) become available.

---

## 📂 Project Structure

- **`data/`** *(Git ignored)*: Contains raw datasets, cleaned datasets, and final generated instruction/validation splits (`train.jsonl`, `val.jsonl`).
- **`scripts/`**: The dataset preparation pipeline:
  - [`collect_data.py`](file:///F:/NepaliLM/scripts/collect_data.py): Script to aggregate Nepali text data from different sources.
  - [`scrape_news.py`](file:///F:/NepaliLM/scripts/scrape_news.py): Utilities for scraping Nepali news articles.
  - [`clean_data.py`](file:///F:/NepaliLM/scripts/clean_data.py): Text normalization and cleaning functions.
  - [`build_instructions.py`](file:///F:/NepaliLM/scripts/build_instructions.py): Converts processed texts into natural Q&A instruction pairs using custom formatting templates.
  - [`verify_dataset.py`](file:///F:/NepaliLM/scripts/verify_dataset.py): Checks formatting, structure, and sizes of the training/validation splits.
- **`training/`**: Model fine-tuning logic:
  - [`train.py`](file:///F:/NepaliLM/training/train.py): Python script using Hugging Face `transformers`, `peft` (LoRA), `bitsandbytes` (4-bit quantization), and `trl` (SFTTrainer) for fine-tuning. Includes presets for:
    - `test` (RTX 3050 pipeline check)
    - `local` (RTX 3050 full run on 3B model)
    - `runpod` (A100 full run on 7B model)
- **`requirements.txt`**: Complete list of Python libraries and dependencies required.
- **`.gitignore`**: Excludes huge data logs, model weights, and python cache/virtual environments from repository history.

---

## 🚀 Setup & Usage (Ready-to-Run)

### 1. Install Dependencies
Ensure you have a virtual environment set up and run:
```bash
pip install -r requirements.txt
```

### 2. Prepare the Datasets
Generate the training instruction sets:
```bash
python scripts/build_instructions.py
python scripts/verify_dataset.py
```

### 3. Model Fine-Tuning (Requires GPU)
Run a quick training validation test:
```bash
python training/train.py --mode test
```

Once suitable GPU infrastructure (e.g., A100 GPU) is acquired:
```bash
python training/train.py --mode runpod
```
