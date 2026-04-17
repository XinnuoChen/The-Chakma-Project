from __future__ import annotations

"""
eval_dictionary.py — Dictionary-lookup evaluation pipeline.

Compares trained LoRA adapters against their base models on a held-out 2 %
slice of Final_Chakma.csv (Chakma headword → English gloss).

Models evaluated:
  1. google/gemma-3-4b-it              (base)
  2. google/gemma-3-4b-it + chemini1_0 (adapter)
  3. meta-llama/Llama-3.1-8B-Instruct              (base)
  4. meta-llama/Llama-3.1-8B-Instruct + llama31_chakgpt1_0 (adapter)

Usage:
    python eval_dictionary.py                 # run all 4 models
    python eval_dictionary.py --models gemma  # only Gemma pair
    python eval_dictionary.py --models llama  # only Llama pair
"""

import argparse
import csv
import gc
import json
import re
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import sacrebleu
from rouge_score import rouge_scorer

# ============================================================
# Config — mirrors gemma_train.py exactly
# ============================================================
CSV_PATH = Path(__file__).resolve().parent / "Final_Chakma.csv"
SEED = 42
TEST_SPLIT = 0.02

SYSTEM_PROMPT = (
    "You are a concise bilingual dictionary assistant. "
    "Given a headword and optional part of speech, return only the English gloss. "
    "Preserve semicolon-separated meanings exactly if present. "
    "Do not add commentary, examples, or extra wording."
)

MODEL_CONFIGS = [
    {
        "label": "gemma-base",
        "base_model": "google/gemma-3-4b-it",
        "adapter_path": None,
        "family": "gemma",
    },
    {
        "label": "gemma-adapter",
        "base_model": "google/gemma-3-4b-it",
        "adapter_path": str(Path(__file__).resolve().parent / "chemini1_0" / "adapter"),
        "family": "gemma",
    },
    {
        "label": "llama-base",
        "base_model": "meta-llama/Llama-3.1-8B-Instruct",
        "adapter_path": None,
        "family": "llama",
    },
    {
        "label": "llama-adapter",
        "base_model": "meta-llama/Llama-3.1-8B-Instruct",
        "adapter_path": str(
            Path(__file__).resolve().parent / "llama31_chakgpt1_0" / "adapter"
        ),
        "family": "llama",
    },
]

GENERATION_KWARGS = dict(
    max_new_tokens=128,
    do_sample=False,
    pad_token_id=None,  # filled per-tokenizer at runtime
)

# ============================================================
# Data loading — identical to gemma_train.py:59-102
# ============================================================
def load_dictionary_rows(path: str):
    rows = []
    bad_lines = 0

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
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
                {"page": page, "headword": headword, "pos": pos, "gloss": gloss}
            )

    if not rows:
        raise ValueError("No usable rows found in the CSV/text file.")

    if bad_lines:
        print(f"  Skipped {bad_lines} malformed line(s).")

    return rows


# ============================================================
# Build prompt messages (Template 0 only, no assistant turn)
# ============================================================
def build_prompt_messages(headword: str, pos: str):
    """Return the chat-message list used at inference time (no assistant turn)."""
    pos_line = f"\nPart of speech: {pos}" if pos else ""
    user_content = f"What does '{headword}' mean in English?{pos_line}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ============================================================
# Build training-style examples (for split reproducibility)
# ============================================================
def _build_training_examples(rows):
    """Mirrors gemma_train.py build_training_examples with PROMPT_VARIANTS_PER_ROW=1."""
    examples = []
    for row in rows:
        pos_line = f"\nPart of speech: {row['pos']}" if row["pos"] else ""
        user_prompt = f"What does '{row['headword']}' mean in English?{pos_line}"
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
    return examples


def build_test_set(csv_path: str):
    """Load CSV → build examples → split 98/2 seed 42 → return test partition."""
    rows = load_dictionary_rows(csv_path)
    print(f"  Total dictionary rows: {len(rows)}")

    examples = _build_training_examples(rows)
    dataset = Dataset.from_list(examples)

    if len(dataset) < 50:
        test_size = max(1, int(round(len(dataset) * 0.1)))
    else:
        test_size = TEST_SPLIT

    split = dataset.train_test_split(test_size=test_size, seed=SEED)
    test_ds = split["test"]
    print(f"  Test set size: {len(test_ds)}")
    return test_ds


# ============================================================
# Model loading
# ============================================================
def load_model_and_tokenizer(base_model_id: str, adapter_path: str | None):
    compute_dtype = (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.eval()
    return model, tokenizer


def unload_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Inference
# ============================================================
def run_inference(model, tokenizer, test_examples):
    predictions = []
    total = len(test_examples)

    for i, ex in enumerate(test_examples):
        headword = ex["meta_headword"]
        pos = ex["meta_pos"]
        messages = build_prompt_messages(headword, pos)

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        gen_kwargs = {**GENERATION_KWARGS, "pad_token_id": tokenizer.eos_token_id}

        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)

        # Decode only the newly generated tokens
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, prompt_len:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        predictions.append(text)

        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"    [{i + 1}/{total}]")

    return predictions


# ============================================================
# Text normalization & metrics
# ============================================================
def normalize(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def levenshtein_distance(s: str, t: str) -> int:
    if len(s) < len(t):
        return levenshtein_distance(t, s)
    if len(t) == 0:
        return len(s)
    prev = list(range(len(t) + 1))
    for i, cs in enumerate(s):
        curr = [i + 1]
        for j, ct in enumerate(t):
            cost = 0 if cs == ct else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def levenshtein_similarity(pred: str, ref: str) -> float:
    if not pred and not ref:
        return 1.0
    dist = levenshtein_distance(pred, ref)
    max_len = max(len(pred), len(ref))
    return 1.0 - dist / max_len


def compute_metrics(predictions: list[str], references: list[str]) -> dict:
    norm_preds = [normalize(p) for p in predictions]
    norm_refs = [normalize(r) for r in references]

    # Exact match (case-insensitive)
    em_scores = [
        1.0 if p.lower() == r.lower() else 0.0
        for p, r in zip(norm_preds, norm_refs)
    ]
    em = sum(em_scores) / len(em_scores) * 100

    # Corpus BLEU
    bleu = sacrebleu.corpus_bleu(norm_preds, [norm_refs])

    # Corpus chrF
    chrf = sacrebleu.corpus_chrf(norm_preds, [norm_refs])

    # ROUGE-L (average F1)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_scores = [scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(norm_preds, norm_refs)]
    rouge_l = sum(rouge_scores) / len(rouge_scores) * 100

    # Levenshtein similarity
    lev_scores = [levenshtein_similarity(p, r) for p, r in zip(norm_preds, norm_refs)]
    lev_sim = sum(lev_scores) / len(lev_scores) * 100

    return {
        "exact_match": round(em, 2),
        "bleu": round(bleu.score, 2),
        "chrf": round(chrf.score, 2),
        "rouge_l": round(rouge_l, 2),
        "levenshtein_sim": round(lev_sim, 2),
    }


# ============================================================
# Output helpers
# ============================================================
def print_results_table(all_metrics: dict):
    header = f"{'Model':<20} {'EM%':>8} {'BLEU':>8} {'chrF':>8} {'ROUGE-L':>8} {'LevSim%':>8}"
    sep = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for label, m in all_metrics.items():
        print(
            f"{label:<20} {m['exact_match']:>8.2f} {m['bleu']:>8.2f} "
            f"{m['chrf']:>8.2f} {m['rouge_l']:>8.2f} {m['levenshtein_sim']:>8.2f}"
        )
    print(sep)


def write_per_example_csv(
    test_examples, all_predictions: dict, out_path: str
):
    fieldnames = ["headword", "pos", "reference"]
    for label in all_predictions:
        fieldnames.append(label)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, ex in enumerate(test_examples):
            row = {
                "headword": ex["meta_headword"],
                "pos": ex["meta_pos"],
                "reference": ex["meta_gloss"],
            }
            for label, preds in all_predictions.items():
                row[label] = normalize(preds[i])
            writer.writerow(row)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Dictionary-lookup evaluation")
    parser.add_argument(
        "--models",
        choices=["all", "gemma", "llama"],
        default="all",
        help="Which model family to evaluate (default: all)",
    )
    parser.add_argument(
        "--csv",
        default=str(CSV_PATH),
        help="Path to Final_Chakma.csv",
    )
    args = parser.parse_args()

    # Filter model configs
    if args.models == "all":
        configs = MODEL_CONFIGS
    else:
        configs = [c for c in MODEL_CONFIGS if c["family"] == args.models]

    print("=" * 60)
    print("Dictionary Lookup Evaluation Pipeline")
    print("=" * 60)

    # Build test set
    print("\n[1/3] Building test set ...")
    test_ds = build_test_set(args.csv)
    test_examples = list(test_ds)
    references = [ex["meta_gloss"] for ex in test_examples]

    # Run inference for each model
    all_predictions = {}
    all_metrics = {}

    print(f"\n[2/3] Running inference ({len(configs)} model(s)) ...")
    for cfg in configs:
        label = cfg["label"]
        print(f"\n  --- {label} ---")
        adapter_info = cfg["adapter_path"] or "none"
        print(f"  Base:    {cfg['base_model']}")
        print(f"  Adapter: {adapter_info}")

        model, tokenizer = load_model_and_tokenizer(
            cfg["base_model"], cfg["adapter_path"]
        )
        preds = run_inference(model, tokenizer, test_examples)
        all_predictions[label] = preds

        metrics = compute_metrics(preds, references)
        all_metrics[label] = metrics
        print(f"  Metrics: {metrics}")

        unload_model(model)
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Output
    print(f"\n[3/3] Writing results ...")

    out_dir = Path(__file__).resolve().parent
    csv_out = out_dir / "eval_results.csv"
    json_out = out_dir / "eval_summary.json"

    write_per_example_csv(test_examples, all_predictions, str(csv_out))
    print(f"  Per-example CSV: {csv_out}")

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"  Summary JSON:    {json_out}")

    print_results_table(all_metrics)
    print("\nDone.")


if __name__ == "__main__":
    main()
