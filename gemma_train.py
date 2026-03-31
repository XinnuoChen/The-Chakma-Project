import os
import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# =========================================================
# CONFIG
# =========================================================
CSV_PATH = "Final_Chakma.csv"
BASE_MODEL = "google/gemma-3-4b-it"
OLLAMA_BASE = "gemma3:4b"
OUTPUT_DIR = "./chemini1_0"
OLLAMA_MODEL_NAME = "chemini1_0"

SYSTEM_PROMPT = (
    "You are a concise bilingual dictionary assistant. "
    "Given a headword and optional part of speech, return only the English gloss. "
    "Preserve semicolon-separated meanings exactly if present. "
    "Do not add commentary, examples, or extra wording."
)

SEED = 42
TEST_SPLIT = 0.02
MAX_LENGTH = 256

NUM_EPOCHS = 1
LEARNING_RATE = 1e-4
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

PROMPT_VARIANTS_PER_ROW = 1

# =========================================================
# HELPERS
# =========================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_dictionary_rows(path: str):
    """
    Reads rows like:
    100,Kushāsan,n,mis-administration; misrule; mis-government.

    Important:
    - first column is page number, not a unique id
    - gloss may contain commas, so we split only on the first 3 commas
    """
    rows = []
    bad_lines = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue

            parts = line.split(",", 3)
            if len(parts) < 4:
                bad_lines += 1
                continue

            page, headword, pos, gloss = [p.strip() for p in parts]

            if not headword or not gloss:
                continue

            rows.append(
                {
                    "page": page,
                    "headword": headword,
                    "pos": pos,
                    "gloss": gloss,
                }
            )

    if not rows:
        raise ValueError("No usable rows found in the CSV/text file.")

    if bad_lines:
        print(f"Skipped {bad_lines} malformed line(s).")

    return rows

def build_user_prompt(headword: str, pos: str, variant: int) -> str:
    pos_line = f"\nPart of speech: {pos}" if pos else ""

    templates = [
        f"What does '{headword}' mean in English?{pos_line}",
        f"Give the English gloss for this dictionary entry.\nHeadword: {headword}{pos_line}",
        f"Translate this headword into concise English.\nWord: {headword}{pos_line}",
        f"Dictionary entry lookup:\nHeadword: {headword}{pos_line}\nReturn only the gloss.",
        f"English meaning for: {headword}{pos_line}",
    ]
    return templates[variant % len(templates)]

def build_training_examples(rows):
    """
    Build message-format examples. We will tokenize these manually so that
    we can also provide token_type_ids required by Gemma 3 during training.
    """
    examples = []

    for row in rows:
        for variant in range(PROMPT_VARIANTS_PER_ROW):
            user_prompt = build_user_prompt(row["headword"], row["pos"], variant)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": row["gloss"]},
            ]

            examples.append(
                {
                    "messages": messages,
                    "meta_page": row["page"],
                    "meta_headword": row["headword"],
                    "meta_pos": row["pos"],
                    "meta_gloss": row["gloss"],
                }
            )

    if not examples:
        raise ValueError("No training examples were created.")

    return examples

def tokenize_example(example, tokenizer, max_length):
    """
    Convert chat messages to a single text string using the tokenizer's
    chat template, then tokenize manually.
    """
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    # Full-sequence LM loss for now.
    labels = input_ids.copy()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": [0] * len(input_ids),
        "labels": labels,
    }

@dataclass
class Gemma3TextCollator:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids = []
        attention_mask = []
        token_type_ids = []
        labels = []

        for f in features:
            seq_len = len(f["input_ids"])
            pad_len = max_len - seq_len

            input_ids.append(f["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            token_type_ids.append(f["token_type_ids"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def write_modelfile(output_dir: str, ollama_base: str) -> str:
    modelfile_path = os.path.join(output_dir, "Modelfile")
    modelfile_text = f"""FROM {ollama_base}
ADAPTER ./adapter

SYSTEM \"\"\"{SYSTEM_PROMPT}\"\"\"

PARAMETER temperature 0.2
PARAMETER num_ctx 4096
"""
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile_text)

    return modelfile_path

# =========================================================
# MAIN
# =========================================================
def main():
    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not detected. Training Gemma 3 4B LoRA without a GPU is usually impractical."
        )

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading rows...")
    rows = load_dictionary_rows(CSV_PATH)
    print(f"Usable dictionary rows: {len(rows)}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Building dataset...")
    examples = build_training_examples(rows)
    dataset = Dataset.from_list(examples)

    if len(dataset) < 50:
        test_size = max(1, int(round(len(dataset) * 0.1)))
        split = dataset.train_test_split(test_size=test_size, seed=SEED)
    else:
        split = dataset.train_test_split(test_size=TEST_SPLIT, seed=SEED)

    train_dataset = split["train"]
    eval_dataset = split["test"]

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples:  {len(eval_dataset)}")

    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_example(x, tokenizer, MAX_LENGTH),
        remove_columns=train_dataset.column_names,
    )
    eval_dataset = eval_dataset.map(
        lambda x: tokenize_example(x, tokenizer, MAX_LENGTH),
        remove_columns=eval_dataset.column_names,
    )

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print("Loading base model in 4-bit QLoRA mode...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        report_to="none",
        gradient_checkpointing=True,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim="paged_adamw_32bit",
        remove_unused_columns=False,
        seed=SEED,
        dataset_text_field="input_ids",
    )

    data_collator = Gemma3TextCollator(tokenizer)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("Starting training...")
    trainer.train()

    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    print("Saving adapter...")
    trainer.model.save_pretrained(str(adapter_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(adapter_dir))

    samples_path = output_dir / "sample_rows.jsonl"
    with open(samples_path, "w", encoding="utf-8") as f:
        for row in rows[:20]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    modelfile_path = write_modelfile(str(output_dir), OLLAMA_BASE)

    print("\nDone.")
    print(f"Adapter saved to:   {adapter_dir}")
    print(f"Modelfile saved to: {modelfile_path}")

    print("\nNext steps:")
    print(f"1) cd {output_dir}")
    print(f"2) ollama create {OLLAMA_MODEL_NAME} -f Modelfile")
    print(f"3) ollama run {OLLAMA_MODEL_NAME}")

if __name__ == "__main__":
    main()