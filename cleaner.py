import csv
import json
import re
from typing import Dict, Iterable, List, Optional, Tuple

BOM = "\ufeff"

def _norm_header(h: str) -> str:
    if h is None:
        return ""
    h = h.replace(BOM, "")
    h = h.strip()
    # keep original names mostly, but normalize common issues:
    return h

def _norm_text(s: str, keep_internal_spaces: bool = True) -> str:
    """Normalize line endings, strip nulls, trim."""
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\x00", "")               # strip null bytes
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.strip()
    if not keep_internal_spaces:
        s = re.sub(r"[ \t]+", " ", s)
    return s

_JSON_TRANSLATION_RE = re.compile(
    r'^\s*\{\s*"translation"\s*:\s*(".*"|[^}]+)\s*\}\s*$',
    re.DOTALL
)

def _extract_translation_if_wrapped(value: str) -> str:
    """
    If a cell contains JSON like {"translation":"..."} or similar,
    extract the translation. Otherwise return original.
    """
    v = _norm_text(value)
    if not v:
        return v

    # 1) Try JSON parse directly
    if v.startswith("{") and v.endswith("}"):
        try:
            obj = json.loads(v)
            if isinstance(obj, dict) and "translation" in obj:
                return _norm_text(obj.get("translation", ""))
        except Exception:
            pass

    # 2) Try a regex extraction for slightly malformed JSON-ish strings
    m = _JSON_TRANSLATION_RE.match(v)
    if m:
        raw = m.group(1).strip()
        # If it looks like a JSON string, unquote it safely
        if raw.startswith('"') and raw.endswith('"'):
            try:
                return _norm_text(json.loads(raw))
            except Exception:
                return _norm_text(raw.strip('"'))
        return _norm_text(raw)

    # 3) Handle "translation: ..." patterns
    low = v.lower()
    if low.startswith("translation:"):
        return _norm_text(v.split(":", 1)[1])

    return v

def clean_parallel_csv(
    in_csv: str,
    out_csv: str,
    *,
    required_cols: Tuple[str, ...] = ("en",),
    ensure_cols: Tuple[str, ...] = ("tgt", "source", "error"),
    drop_empty_en: bool = True,
    dedupe_by_en: bool = True,
    keep_internal_spaces: bool = True,
    sanitize_tgt_json: bool = True,
    encoding_in: str = "utf-8-sig",
    encoding_out: str = "utf-8"
) -> Dict[str, int]:
    """
    Cleans a parallel CSV used by your translation loop.
    Returns stats dict (rows_in, rows_out, dropped_empty_en, deduped, etc.)
    """
    stats = {
        "rows_in": 0,
        "rows_out": 0,
        "dropped_empty_en": 0,
        "deduped": 0,
        "missing_required_cols": 0,
    }

    with open(in_csv, newline="", encoding=encoding_in) as f:
        reader = csv.DictReader(f)

        # Normalize headers
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header row.")
        orig_fields = list(reader.fieldnames)
        norm_fields = [_norm_header(h) for h in orig_fields]

        # Map original->normalized so we can read rows properly
        # (DictReader uses original keys; we’ll re-key rows)
        field_map = dict(zip(orig_fields, norm_fields))

        # Required columns check
        lower_fields = {h.lower(): h for h in norm_fields}
        for col in required_cols:
            if col.lower() not in lower_fields:
                stats["missing_required_cols"] += 1

        if stats["missing_required_cols"] > 0:
            raise ValueError(
                f"Missing required columns: {required_cols}. "
                f"Found: {norm_fields}"
            )

        # Output fields: keep all original normalized fields + ensure_cols
        out_fields: List[str] = []
        seen = set()
        for h in norm_fields:
            key = h
            if key not in seen and key != "":
                out_fields.append(key)
                seen.add(key)

        for c in ensure_cols:
            if c not in seen:
                out_fields.append(c)
                seen.add(c)

        # Clean rows
        out_rows: List[Dict[str, str]] = []
        seen_en = set()

        for row in reader:
            stats["rows_in"] += 1

            # re-key with normalized headers
            r: Dict[str, str] = {}
            for k, v in row.items():
                nk = field_map.get(k, k)
                r[nk] = _norm_text(v, keep_internal_spaces=keep_internal_spaces)

            # Ensure required/extra columns exist
            for c in ensure_cols:
                r.setdefault(c, "")

            # Normalize key column: 'en'
            # (handle if header was 'EN' etc.)
            en_key = lower_fields["en"]
            en = _norm_text(r.get(en_key, ""), keep_internal_spaces=keep_internal_spaces)

            if drop_empty_en and not en:
                stats["dropped_empty_en"] += 1
                continue

            if dedupe_by_en:
                en_sig = en
                if en_sig in seen_en:
                    stats["deduped"] += 1
                    continue
                seen_en.add(en_sig)

            # Optional: sanitize tgt if it contains JSON wrapper
            if sanitize_tgt_json:
                if "tgt" in r:
                    r["tgt"] = _extract_translation_if_wrapped(r["tgt"])

            # Build row in output field order
            out_row = {fn: r.get(fn, "") for fn in out_fields}
            out_rows.append(out_row)

    # Write cleaned CSV
    with open(out_csv, "w", newline="", encoding=encoding_out) as f:
        w = csv.DictWriter(
            f,
            fieldnames=out_fields,
            quoting=csv.QUOTE_MINIMAL,   # safe for Excel
            lineterminator="\n"
        )
        w.writeheader()
        w.writerows(out_rows)

    stats["rows_out"] = len(out_rows)
    return stats


if __name__ == "__main__":
    # Example usage: clean input before drafting
    stats = clean_parallel_csv(
        in_csv=r"C:\Users\Ed\Documents\testbed_llama.csv",
        out_csv=r"C:\Users\Ed\Documents\parallel_seed.cleaned.csv",
        drop_empty_en=True,
        dedupe_by_en=True,
        sanitize_tgt_json=True,
    )
    print("Clean complete:", stats)