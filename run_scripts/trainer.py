import json
from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

from peft import LoraConfig, get_peft_model

# Pick the SAME base family you plan to use in Ollama.
# If you want "llama3.1:8b" in Ollama, use the matching Llama 3.1 8B Instruct weights here.
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

TRAIN_JSONL = "data/chakma_train.jsonl"
VAL_JSONL   = "data/chakma_val.jsonl"
OUT_DIR     = "outputs/chakma_lora"

MAX_LEN = 768  # keep similar to your inference num_ctx
LR = 2e-4

def build_prompt(system: str, src: str) -> str:
    # A stable, minimal prompt that matches your inference behavior
    return f"{system}\n\nText:\n{src}\n\nTranslation:\n"

@dataclass
class PadCollator:
    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # features already have input_ids, attention_mask, labels
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id

        def pad(seq, pad_value):
            return seq + [pad_value] * (max_len - len(seq))

        input_ids = torch.tensor([pad(f["input_ids"], pad_id) for f in features], dtype=torch.long)
        attention = torch.tensor([pad(f["attention_mask"], 0) for f in features], dtype=torch.long)
        labels = torch.tensor([pad(f["labels"], -100) for f in features], dtype=torch.long)

        return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}

def main():
    ds = load_dataset("json", data_files={"train": TRAIN_JSONL, "validation": VAL_JSONL})

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # QLoRA-style load (4-bit) – best chance to fit on 12GB
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(model, lora)

    def tokenize(ex):
        system = ex["system"]
        src = ex["input"]
        tgt = ex["output"]

        prompt = build_prompt(system, src)
        full = prompt + tgt

        prompt_ids = tok(prompt, add_special_tokens=True, truncation=True, max_length=MAX_LEN)["input_ids"]
        full_enc = tok(full, add_special_tokens=True, truncation=True, max_length=MAX_LEN)

        input_ids = full_enc["input_ids"]
        attention = full_enc["attention_mask"]

        # Mask the prompt part so we only train on the target translation tokens
        labels = input_ids.copy()
        cut = min(len(prompt_ids), len(labels))
        for i in range(cut):
            labels[i] = -100

        return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}

    ds_tok = ds.map(tokenize, remove_columns=ds["train"].column_names)

    args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=LR,
        num_train_epochs=2,
        logging_steps=25,
        save_steps=250,
        eval_steps=250,
        evaluation_strategy="steps",
        save_total_limit=2,
        fp16=True,
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        data_collator=PadCollator(tok),
    )

    trainer.train()

    # Saves PEFT adapter as safetensors + config (what Ollama can import as an adapter directory)
    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)

    print(f"Saved adapter to: {OUT_DIR}")

if __name__ == "__main__":
    main()
