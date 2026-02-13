import csv
import json
import requests
import time
import os
from tqdm import tqdm  # <- add this

import csv
import json
import requests
import time
import os
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.1:8b"
TARGET_LANG = "Romanized Chakma"

# 🔧 Keep system prompt SHORT to reduce tokens + overhead.
SYSTEM = f"""/no_think
Translate English -> {TARGET_LANG}.
Output ONLY the translation(s). Preserve numbers/punctuation/line breaks.
Keep URLs/emails/product codes unchanged. Don't translate text inside backticks.
If unsure, copy tokens rather than invent.
"""

def _post_ollama(session: requests.Session, payload: Dict[str, Any]) -> Dict[str, Any]:
    r = session.post(OLLAMA_URL, json=payload, timeout=1800)
    r.raise_for_status()
    return r.json()

def _extract_translation_from_response(data: Dict[str, Any]) -> str:
    """
    Ollama returns JSON. With format="json", message.content may itself be JSON-as-text.
    We'll accept:
      - {"translation": "..."}
      - {"translations": ["...", ...]}
    but this helper just returns raw content string for caller to parse.
    """
    content = ((data.get("message") or {}).get("content", "") or "").strip()
    return content

def ollama_translate_batch(
    session: requests.Session,
    items: List[str],
    num_predict: int = 60,
    num_ctx: int = 768,
    keep_alive: str = "60m",
) -> List[str]:
    """
    Translate a batch of English strings in one request.
    Returns list of translations in same order.
    """
    # Number + delimiter the items so the model can keep alignment.
    # Ask for JSON array only (no long schema explanation).
    joined = "\n".join([f"{i+1}. {t}" for i, t in enumerate(items)])

    user_prompt = (
        "/no_think\n"
        "Return ONLY valid JSON in this exact shape:\n"
        '{"translations":["...","..."]}\n'
        f"Translate each numbered line to {TARGET_LANG} in order.\n\n"
        f"{joined}"
    )

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "keep_alive": keep_alive,
        "format": "json",
        "options": {
            "temperature": 0,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
        },
    }

    data = _post_ollama(session, payload)
    content = _extract_translation_from_response(data)

    # Parse JSON. If it fails, fall back to raw content (but then we can't split safely).
    try:
        obj = json.loads(content)
    except Exception:
        raise ValueError(f"Batch JSON parse failed. content_preview={content[:200]!r}")

    out = obj.get("translations", None)
    if not isinstance(out, list) or not all(isinstance(x, str) for x in out):
        raise ValueError(f"Bad batch shape. keys={list(obj.keys())} content_preview={content[:200]!r}")

    if len(out) != len(items):
        raise ValueError(f"Batch size mismatch: got {len(out)} expected {len(items)}")

    return [s.strip() for s in out]

def warmup(session: requests.Session) -> None:
    """
    One tiny request to load the model before the main loop.
    """
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "/no_think\nReturn ONLY valid JSON:\n{\"translation\":\"ok\"}\nText:\nHi"},
        ],
        "stream": False,
        "keep_alive": "60m",
        "format": "json",
        "options": {"temperature": 0, "num_predict": 8, "num_ctx": 256},
    }
    _post_ollama(session, payload)

def draft_translations(
    in_csv: str = "parallel_seed.csv",
    out_csv: str = r"C:\Users\Ed\Documents\testbed_llama.csv",
    limit: int = 20,
    batch_size: int = 10,          # 🔥 BIG SPEED WIN: 5–20 is usually best
    retry: int = 1,
    num_predict: int = 60,        # 🔧 reduce if rows are short (e.g. 60–100)
    num_ctx: int = 768,            # 🔧 reduce if rows are short (e.g. 512–768)
):
    t0 = time.time()

    # Write as we go (so you ALWAYS see progress + it doesn't feel "stuck")
    tmp_csv = out_csv + ".tmp"

    with requests.Session() as session:
        print("Warming model...", flush=True)
        warmup(session)
        print("Warm. Starting draft...", flush=True)

        with open(in_csv, newline="", encoding="utf-8") as fin, open(tmp_csv, "w", newline="", encoding="utf-8") as fout:
            reader = csv.DictReader(fin)
            if not reader.fieldnames:
                raise ValueError("Input CSV has no header/fieldnames.")

            fieldnames = list(reader.fieldnames)
            for col in ("tgt", "source", "error"):
                if col not in fieldnames:
                    fieldnames.append(col)
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()

            batch_rows: List[Dict[str, str]] = []
            batch_texts: List[str] = []
            written = 0

            pbar = tqdm(total=limit, desc="Drafting", unit="row", dynamic_ncols=True)

            def flush_batch():
                nonlocal batch_rows, batch_texts, written
                if not batch_rows:
                    return

                # Handle rows with missing "en" without sending to model
                idx_map: List[int] = []  # positions in batch_rows that have text
                to_translate: List[str] = []

                for j, r in enumerate(batch_rows):
                    en = (r.get("en") or "").strip()
                    if not en:
                        r["tgt"] = ""
                        r["source"] = ""
                        r["error"] = "missing_en"
                    else:
                        idx_map.append(j)
                        to_translate.append(en)

                if to_translate:
                    last_err = ""
                    for attempt in range(retry + 1):
                        try:
                            outs = ollama_translate_batch(
                                session,
                                to_translate,
                                num_predict=num_predict,
                                num_ctx=num_ctx,
                                keep_alive="60m",
                            )
                            # Put results back into correct rows
                            for k, out in enumerate(outs):
                                rpos = idx_map[k]
                                batch_rows[rpos]["tgt"] = out
                                batch_rows[rpos]["source"] = "ollama_draft"
                                batch_rows[rpos]["error"] = ""
                            break
                        except Exception as e:
                            last_err = f"{type(e).__name__}: {e}"
                            if attempt < retry:
                                time.sleep(0.5)
                            else:
                                # Mark only the rows that were supposed to be translated as failed
                                for rpos in idx_map:
                                    batch_rows[rpos]["tgt"] = ""
                                    batch_rows[rpos]["source"] = ""
                                    batch_rows[rpos]["error"] = last_err

                for r in batch_rows:
                    writer.writerow(r)
                    written += 1
                    pbar.update(1)

                batch_rows = []
                batch_texts = []

            for row_i, row in enumerate(reader, start=1):
                if row_i > limit:
                    break

                # Collect rows for batching (we'll also handle missing_en in flush_batch)
                batch_rows.append(row)
                
               
                pbar.set_postfix_str("queued")

                if len(batch_rows) >= batch_size:
                    flush_batch()



            # Flush tail
            flush_batch()
            pbar.close()

    os.replace(tmp_csv, out_csv)

    elapsed = time.time() - t0
    rate = (min(limit, written) / elapsed) if elapsed > 0 else 0
    print(f"Wrote {written} rows to {out_csv} in {elapsed:.1f}s ({rate:.2f} rows/s)", flush=True)

if __name__ == "__main__":
    draft_translations(
        limit=20,
        batch_size=10,
        retry=1,
        num_predict=60,
        num_ctx=768,
    )















r'''
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.1:70b"
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
        "format": "json",
        "options": {
            "temperature": 0,
            "num_predict": 200,
            "num_ctx": 1024,
        },
    }

    r = session.post(OLLAMA_URL, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()

    content = ((data.get("message") or {}).get("content", "") or "").strip()

    try:
        obj = json.loads(content)
        out = (obj.get("translation") or "").strip()
    except Exception:
        out = ""

    if not out and content:
        return content

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
    retry=1
):
    rows = []
    t0 = time.time()

    with requests.Session() as session:
        with open(in_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # tqdm bar: total=limit gives a real progress bar even though we break early
            for i, row in enumerate(tqdm(reader, total=limit, desc="Drafting", unit="row"), start=1):
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

    # Ensure consistent columns
    fieldnames = list(rows[0].keys()) if rows else ["en", "tgt", "source", "error"]
    for col in ("tgt", "source", "error"):
        if col not in fieldnames:
            fieldnames.append(col)

    tmp_csv = out_csv + ".tmp"
    with open(tmp_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    os.replace(tmp_csv, out_csv)

    elapsed = time.time() - t0
    print(f"Wrote {len(rows)} drafted rows to {out_csv} in {elapsed:.1f}s")


if __name__ == "__main__":
    draft_translations(limit=100, retry=1)
'''
