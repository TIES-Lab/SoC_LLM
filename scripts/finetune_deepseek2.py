#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune DeepSeek-Coder-6.7B-Instruct on a dataset of single-field "text" entries.
Each "text" already includes ### Instruction and ### Response sections.
"""

import os, json, re, torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments,
    BitsAndBytesConfig, DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# -------- CONFIG --------
DATASET_FILE = "Adapted_dataset_Deepseek_coder_clean.json"           # <-- your file name
MODEL_NAME   = "deepseek-ai/deepseek-coder-6.7b-instruct"
OUTPUT_DIR   = "./deepseek_finetuned"
USE_4BIT     = True                     # QLoRA
EPOCHS       = 2
BATCH        = 2
ACC_STEPS    = 8
LR           = 2e-4
MAX_LEN      = 2048

# -------- HELPERS --------
RESPONSE_RE = re.compile(r"###\s*Response\s*:", re.IGNORECASE)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), "JSON must be a list of objects."
    return data

def split_text(entry):
    """Return dict with prompt_text / response_text split at '### Response:'."""
    txt = entry["text"].replace("\r", "")
    txt = re.sub(r"\[EOT\]\s*$", "", txt)
    m = RESPONSE_RE.search(txt)
    if not m:
        return {"prompt": txt, "response": ""}
    return {"prompt": txt[:m.start()].strip() + "\n### Response:\n",
            "response": txt[m.end():].strip()}

'''
def tokenize(batch, tok):
    ids, labs, masks = [], [], []
    for t in batch["text"]:
        parts = split_text({"text": t})
        full = parts["prompt"] + parts["response"] + tok.eos_token
        tok_full = tok(full, truncation=True, max_length=MAX_LEN)
        tok_prompt = tok(parts["prompt"], truncation=True, max_length=MAX_LEN)
        input_ids = tok_full["input_ids"]
        labels = input_ids.copy()
        plen = len(tok_prompt["input_ids"])
        for i in range(min(plen, len(labels))):
            labels[i] = -100
        ids.append(input_ids); labs.append(labels); masks.append(tok_full["attention_mask"])
    return {"input_ids": ids, "labels": labs, "attention_mask": masks}
'''


def tokenize(batch, tok):
    """
    Tokenize with explicit padding and truncation so all examples
    in a batch have the same length.
    """
    input_ids_list, labels_list, attn_masks_list = [], [], []

    for text in batch["text"]:
        parts = split_text({"text": text})

        prompt = parts["prompt"]
        response = parts["response"]

        # combine into one full example
        full_text = prompt + response + tok.eos_token

        # tokenize both with consistent padding & truncation
        tok_full = tok(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
        )
        tok_prompt = tok(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
        )

        input_ids = tok_full["input_ids"]
        attn_mask = tok_full["attention_mask"]
        labels = input_ids.copy()

        prompt_len = len(tok_prompt["input_ids"])
        # mask prompt tokens (no loss on them)
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attn_masks_list.append(attn_mask)

    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attn_masks_list,
    }


# -------- MAIN --------
def main():
    print("Fine-tuning DeepSeek-Coder on dataset.json")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=USE_4BIT,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_cfg)

    ds = Dataset.from_list(load_json(DATASET_FILE))
    #ds = ds.map(lambda b: tokenize(b, tok), batched=True, remove_columns=ds.column_names)
    ds = ds.map(
        lambda b: tokenize(b, tok),
        batched=True,
        remove_columns=ds.column_names,
        )
    ds = ds.map(
         lambda x: tok.pad(x, padding=True, return_tensors=None),
        batched=True,
        )


    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=ACC_STEPS,
        learning_rate=LR,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        lr_scheduler_type="cosine",
        optim="paged_adamw_32bit",
        logging_steps=20,
        save_steps=200,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      tokenizer=tok, data_collator=collator)
    trainer.train()
    trainer.save_model(OUTPUT_DIR); tok.save_pretrained(OUTPUT_DIR)
    print("Model saved at", OUTPUT_DIR)

if __name__ == "__main__":
    main()

