import csv
import json
import requests
import time
import os

OLLAMA_URL = "http://localhost:11434/api/chat"
#MODEL = "gemma3:4b"
MODEL = "llama3.1:8b"
TARGET_LANG = "Romanized Chakma"

SYSTEM = f"""/no_think
You are a translation engine.

Rules:
- Translate from English to {TARGET_LANG}.
- Output ONLY the translation. No quotes. No explanations.
- Preserve all numbers, dates, currency symbols, punctuation, and line breaks.
- Keep names, product codes, URLs, emails unchanged.
- Do not translate text inside backticks `like this`.
- If unsure, copy the source token rather than inventing.
"""

def ollama_translate(session: requests.Session, text: str) -> str:
    # Ask for strict JSON output so we don't get "thinking" only.
    user_prompt = (
        f"/no_think\n"
        f"Return ONLY valid JSON in this exact shape:\n"
        f'{{"translation": "<{TARGET_LANG} translation here>"}} \n'
        f"No other keys. No extra text.\n\n"
        f"Text:\n{text}"
    )

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "keep_alive": "10m",
        "format": "json",  # <<< IMPORTANT
        "options": {
            "temperature": 0,
            "num_predict": 200,
            "num_ctx": 1024,
        },
    }

    r = session.post(OLLAMA_URL, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()

    content = (data.get("message") or {}).get("content", "")
    content = content.strip()

    # Parse JSON content
    try:
        obj = json.loads(content)
        out = (obj.get("translation") or "").strip()
    except Exception:
        out = ""

# 🔧 FIX: if JSON parsing didn't give us anything,
# just return the raw content
    if not out and content.strip():
        return content.strip()

# Only error if it's truly empty
    if not out:
        thinking_preview = (data.get("message") or {}).get("thinking", "")[:160]
        raise ValueError(
            "No translation produced. "
            f"content_preview={content[:160]!r} thinking_preview={thinking_preview!r}"
        )

    return out



def draft_translations(
    in_csv="parallel_seed.csv",
    out_csv=r"C:\Users\Ed\Documents\testbed_llama.csv",
    limit=50,
    progress_every=5,
    retry=1
):
    rows = []
    t0 = time.time()

    with requests.Session() as session:
        with open(in_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for i, row in enumerate(reader, start=1):
                if i > limit:
                    break

                en = row.get("en", "")
                if not en:
                    row["tgt"] = ""
                    row["source"] = ""
                    row["error"] = "missing_en"
                    rows.append(row)
                    continue

                last_err = ""
                for attempt in range(retry + 1):
                    try:
                        tgt = ollama_translate(session, en)
                        row["tgt"] = tgt
                        row["source"] = "ollama_draft"
                        row["error"] = ""
                        break
                    except Exception as e:
                        last_err = f"{type(e).__name__}: {e}"
                        if attempt < retry:
                            time.sleep(0.5)
                        else:
                            row["tgt"] = ""
                            row["source"] = ""
                            row["error"] = last_err

                rows.append(row)

                if progress_every and (i % progress_every == 0 or i == limit):
                    elapsed = time.time() - t0
                    rate = i / elapsed if elapsed > 0 else 0
                    remaining = (limit - i) / rate if rate > 0 else float("inf")
                    print(f"[{i}/{limit}] elapsed={elapsed:.1f}s "
                          f"rate={rate:.2f} rows/s ETA={remaining:.1f}s")

    # Ensure consistent columns
    fieldnames = list(rows[0].keys())
    for col in ("tgt", "source", "error"):
        if col not in fieldnames:
            fieldnames.append(col)

    # Robust write: write tmp then replace
    tmp_csv = out_csv + ".tmp"
    with open(tmp_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # If Excel has the file open, this replace can still fail.
    os.replace(tmp_csv, out_csv)

    print(f"Wrote {len(rows)} drafted rows to {out_csv}")

if __name__ == "__main__":
    draft_translations(limit=100, progress_every=5, retry=1)
