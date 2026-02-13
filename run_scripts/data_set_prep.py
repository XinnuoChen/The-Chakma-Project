import csv
import json
import random
from pathlib import Path

SYSTEM = """/no_think
Translate English -> Romanized Chakma.
Output ONLY the translation(s). Preserve numbers/punctuation/line breaks.
Keep URLs/emails/product codes unchanged. Don't translate text inside backticks.
If unsure, copy tokens rather than invent.
""".strip()

def prepare_jsonl(
    in_csv: str,
    out_train: str,
    out_val: str,
    src_col: str = "en",
    tgt_col: str = "tgt",
    val_frac: float = 0.02,
    seed: int = 42,
    limit: int | None = None,
):
    random.seed(seed)

    rows = []
    with open(in_csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise ValueError("CSV has no header.")
        for i, row in enumerate(r, start=1):
            if limit is not None and i > limit:
                break
            src = (row.get(src_col) or "").strip()
            tgt = (row.get(tgt_col) or "").strip()
            if not src or not tgt:
                continue
            rows.append({"src": src, "tgt": tgt})

    random.shuffle(rows)
    n_val = max(1, int(len(rows) * val_frac)) if rows else 0
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    Path(out_train).parent.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path, items):
        with open(path, "w", encoding="utf-8") as f:
            for ex in items:
                # Keep it template-free + easy to mask inputs during training
                f.write(json.dumps({
                    "system": SYSTEM,
                    "input": ex["src"],
                    "output": ex["tgt"],
                }, ensure_ascii=False) + "\n")

    write_jsonl(out_train, train_rows)
    write_jsonl(out_val, val_rows)

    print(f"Prepared: {len(train_rows)} train, {len(val_rows)} val")
    print(f"Train: {out_train}")
    print(f"Val:   {out_val}")

if __name__ == "__main__":
    prepare_jsonl(
        in_csv="parallel_seed.csv",
        out_train="data/chakma_train.jsonl",
        out_val="data/chakma_val.jsonl",
        src_col="en",
        tgt_col="tgt",
        val_frac=0.02,
    )
