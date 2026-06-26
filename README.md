# DeepSeek RAG Demo

This demo runs a fine-tuned DeepSeek model with RAG context from local files, JSON datasets, Markdown/text files, HTML files, and web pages.

## Folder Structure

```text
demo/
├── README.md
├── requirements.txt
├── run_rag.sh
├── config/
│   └── rag_config.json
├── input/
│   ├── code.txt
│   └── rag_sources.txt
├── data/
│   └── Adapted_dataset_Deepseek_coder_clean.json
├── model/
│   └── deepseek_finetuned/
├── outputs/
└── scripts/
    ├── finetune_deepseek2.py
    └── rag_inference_config.py
```

## Setup

From inside this `demo/` folder:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If a virtual environment is already active, you can skip creating a new one and only run:

```bash
pip install -r requirements.txt
```

## What To Edit For RAG Inference

Only edit these two files for normal RAG inference:

```text
input/code.txt
input/rag_sources.txt
```

Put the Verilog/SystemVerilog code to analyze in:

```text
input/code.txt
```

Put RAG sources in:

```text
input/rag_sources.txt
```

Each RAG source should be one file path or URL per line.

Example:

```text
data/Adapted_dataset_Deepseek_coder_clean.json
https://opentitan.org/book/index.html
input/my_notes.txt
input/assertion_rules.md
input/local_doc.html
```

Lines starting with `#` are ignored.

## Run RAG Inference

From inside this `demo/` folder:

```bash
bash run_rag.sh
```

The generated answer is printed in the terminal and saved to:

```text
outputs/rag_output.txt
```

## Config

The main config file is:

```text
config/rag_config.json
```

Default config:

```json
{
  "model_dir": "model/deepseek_finetuned",
  "input_code_file": "input/code.txt",
  "output_file": "outputs/rag_output.txt",
  "rag_sources_file": "input/rag_sources.txt",
  "top_k": 5,
  "chunk_chars": 1800,
  "chunk_overlap": 250,
  "max_context_chars": 7000,
  "max_html_pages_per_url": 20,
  "max_new_tokens": 2048,
  "temperature": 0.7,
  "top_p": 0.9
}
```

Users usually only need to edit `input/code.txt` and `input/rag_sources.txt`. Edit `config/rag_config.json` only if you want to change model path, generation settings, retrieval settings, or output file path.

## Scripts

The main script used for RAG inference is:

```text
scripts/rag_inference_config.py
```

The fine-tuning script is also included:

```text
scripts/finetune_deepseek2.py
```

You do not need to run the fine-tuning script for normal demo usage. It is included only if you want to retrain or create a new fine-tuned model later.

## Fine-Tuning

The VERT training dataset is included here:

```text
data/Adapted_dataset_Deepseek_coder_clean.json
```

Fine-tuning is only needed if you want to recreate or update the model.

Before running fine-tuning, check these settings inside:

```text
scripts/finetune_deepseek2.py
```

They should match the demo folder layout:

```python
DATASET_FILE = "data/Adapted_dataset_Deepseek_coder_clean.json"
OUTPUT_DIR   = "model/deepseek_finetuned"
```

If you want to avoid overwriting the existing model, use a new output directory:

```python
OUTPUT_DIR = "model/deepseek_finetuned_v2"
```

Then run from inside the `demo/` folder:

```bash
python scripts/finetune_deepseek2.py
```

After fine-tuning finishes, point `config/rag_config.json` to the model folder:

```json
"model_dir": "model/deepseek_finetuned"
```

or, if you used a new folder:

```json
"model_dir": "model/deepseek_finetuned_v2"
```


## Notes

If the server has no internet, remove web URLs from `input/rag_sources.txt` and use local files instead.

If `bash run_rag.sh` fails because dependencies are missing, activate the venv and run:

```bash
pip install -r requirements.txt
```

If the model folder is not included in the zip, place the fine-tuned model here before running:

```text
model/deepseek_finetuned/
```
