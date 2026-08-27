"""
NepaliLM — Task 1b: Data Cleaning Pipeline
Cleans raw Nepali text: Unicode normalize, filter Hindi/Roman, dedupe, quality filter.
"""

import re
import json
import unicodedata
import hashlib
from pathlib import Path
from typing import Optional

RAW_DIR     = Path("data/raw")
CLEANED_DIR = Path("data/cleaned")
CLEANED_DIR.mkdir(parents=True, exist_ok=True)

# ── Devanagari Unicode range ──────────────────────────────────────────────────
DEVANAGARI_RANGE = (0x0900, 0x097F)     # core Devanagari block
DEVANAGARI_EXT   = (0xA8E0, 0xA8FF)     # extended Devanagari

# Characters common in Hindi but rare/wrong in Nepali
# ड़ (0x095C), ढ़ (0x095D), ऩ (0x0929 with nukta), ऱ (0x0931 with nukta)
HINDI_SPECIFIC_CHARS = set("\u095C\u095D\u0929\u0931\u0934")

# Minimum ratio of Devanagari characters in a sample
MIN_DEVANAGARI_RATIO = 0.5

# Length filters
MIN_CHARS = 50
MAX_CHARS = 4000


# ── Unicode helpers ───────────────────────────────────────────────────────────

def normalize_unicode(text: str) -> str:
    """NFC normalize — critical for consistent Devanagari encoding."""
    return unicodedata.normalize("NFC", text)


def is_devanagari_char(ch: str) -> bool:
    cp = ord(ch)
    return (DEVANAGARI_RANGE[0] <= cp <= DEVANAGARI_RANGE[1] or
            DEVANAGARI_EXT[0]   <= cp <= DEVANAGARI_EXT[1])


def devanagari_ratio(text: str) -> float:
    """Fraction of non-whitespace characters that are Devanagari."""
    non_ws = [c for c in text if not c.isspace()]
    if not non_ws:
        return 0.0
    dev = sum(1 for c in non_ws if is_devanagari_char(c))
    return dev / len(non_ws)


# ── Filters ───────────────────────────────────────────────────────────────────

def has_too_many_hindi_chars(text: str) -> bool:
    """Detect if text has significant Hindi-specific characters."""
    hindi_count = sum(1 for c in text if c in HINDI_SPECIFIC_CHARS)
    total_dev   = sum(1 for c in text if is_devanagari_char(c))
    if total_dev == 0:
        return False
    # Reject if >5% of Devanagari chars are Hindi-specific
    return (hindi_count / total_dev) > 0.05


def has_roman_nepali(text: str) -> bool:
    """Detect if text is predominantly Roman (transliterated Nepali)."""
    non_ws = [c for c in text if not c.isspace()]
    if not non_ws:
        return True
    latin = sum(1 for c in non_ws if c.isascii() and c.isalpha())
    return (latin / len(non_ws)) > 0.4


def has_spam_patterns(text: str) -> bool:
    """Detect low-quality/spam text."""
    spam_patterns = [
        r"http[s]?://",           # URLs (keep text clean)
        r"www\.",
        r"@\w+",                  # social handles
        r"[A-Z]{5,}",             # excessive caps (spam)
        r"(.)\1{6,}",             # repeated chars: aaaaaaaa
        r"(\S+\s)\1{3,}",         # repeated phrases
    ]
    for pattern in spam_patterns:
        if re.search(pattern, text):
            return True
    return False


def is_quality(text: str) -> bool:
    """Master quality gate — returns True if text passes all checks."""
    if len(text) < MIN_CHARS or len(text) > MAX_CHARS:
        return False
    if devanagari_ratio(text) < MIN_DEVANAGARI_RATIO:
        return False
    if has_roman_nepali(text):
        return False
    if has_too_many_hindi_chars(text):
        return False
    if has_spam_patterns(text):
        return False
    return True


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Clean and normalize a Nepali text sample."""
    # 1. NFC normalize
    text = normalize_unicode(text)

    # 2. Remove URLs
    text = re.sub(r"http[s]?://\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)

    # 3. Remove HTML entities and tags
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)

    # 4. Normalize whitespace - preserve paragraph breaks
    text = re.sub(r"[ \t\u00a0\u2000-\u200a]+", " ", text)  # collapse multiple horizontal spaces
    text = re.sub(r"\r\n|\r", "\n", text)                  # normalize line endings
    text = re.sub(r"\n{3,}", "\n\n", text)                 # limit consecutive newlines to max 2
    text = text.strip()

    # 5. Normalize Nepali punctuation
    text = text.replace("\u200b", "")   # zero-width space
    text = text.replace("\u200c", "")   # zero-width non-joiner (keep if intentional)
    text = text.replace("\ufeff", "")   # BOM

    return text


def fingerprint(text: str) -> str:
    """Hash for deduplication — first 200 chars after normalization."""
    sample = text[:200].strip()
    return hashlib.md5(sample.encode("utf-8")).hexdigest()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def clean_file(input_path: Path, output_path: Path) -> dict:
    """
    Clean a single JSONL file.
    Returns stats dict.
    """
    stats = {
        "total": 0,
        "passed": 0,
        "rejected_length": 0,
        "rejected_devanagari": 0,
        "rejected_hindi": 0,
        "rejected_roman": 0,
        "rejected_spam": 0,
        "rejected_duplicate": 0,
    }

    seen_hashes = set()

    with (open(input_path,  "r", encoding="utf-8") as fin,
          open(output_path, "w", encoding="utf-8") as fout):

        for line in fin:
            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue

            raw_text = sample.get("text", "")
            stats["total"] += 1

            # Clean first
            text = clean_text(raw_text)

            # Length check
            if len(text) < MIN_CHARS or len(text) > MAX_CHARS:
                stats["rejected_length"] += 1
                continue

            # Devanagari ratio
            if devanagari_ratio(text) < MIN_DEVANAGARI_RATIO:
                stats["rejected_devanagari"] += 1
                continue

            # Roman Nepali
            if has_roman_nepali(text):
                stats["rejected_roman"] += 1
                continue

            # Hindi contamination
            if has_too_many_hindi_chars(text):
                stats["rejected_hindi"] += 1
                continue

            # Spam patterns
            if has_spam_patterns(text):
                stats["rejected_spam"] += 1
                continue

            # Deduplication
            fp = fingerprint(text)
            if fp in seen_hashes:
                stats["rejected_duplicate"] += 1
                continue
            seen_hashes.add(fp)

            # Passed — write cleaned sample
            cleaned = {"text": text, "source": sample.get("source", "unknown")}
            if "title" in sample:
                cleaned["title"] = clean_text(sample["title"])
            fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            stats["passed"] += 1

            if stats["passed"] % 10_000 == 0:
                print(f"  Cleaned {stats['passed']:,} samples so far...")

    return stats


def print_stats(filename: str, stats: dict):
    total     = stats["total"]
    passed    = stats["passed"]
    rejected  = total - passed
    pass_rate = (passed / total * 100) if total else 0

    print(f"\n{'─'*50}")
    print(f"  File    : {filename}")
    print(f"  Total   : {total:>10,}")
    print(f"  Passed  : {passed:>10,}  ({pass_rate:.1f}%)")
    print(f"  Rejected: {rejected:>10,}")
    print(f"    length     : {stats['rejected_length']:>8,}")
    print(f"    devanagari : {stats['rejected_devanagari']:>8,}")
    print(f"    roman      : {stats['rejected_roman']:>8,}")
    print(f"    hindi      : {stats['rejected_hindi']:>8,}")
    print(f"    spam       : {stats['rejected_spam']:>8,}")
    print(f"    duplicate  : {stats['rejected_duplicate']:>8,}")


def run_cleaning_pipeline():
    print("NepaliLM — Data Cleaning Pipeline")
    print("="*50)

    raw_files = list(RAW_DIR.glob("*.jsonl"))
    if not raw_files:
        print("No raw files found in data/raw/")
        print("Run scripts/collect_data.py first.")
        return

    all_stats = {}
    for raw_path in sorted(raw_files):
        out_path = CLEANED_DIR / raw_path.name
        print(f"\nCleaning: {raw_path.name}")
        stats = clean_file(raw_path, out_path)
        print_stats(raw_path.name, stats)
        all_stats[raw_path.name] = stats

    # Combined summary
    total_passed = sum(s["passed"] for s in all_stats.values())
    total_all    = sum(s["total"]  for s in all_stats.values())
    print(f"\n{'='*50}")
    print(f"TOTAL CLEANED: {total_passed:,} / {total_all:,} samples")
    print(f"Output: {CLEANED_DIR}/")
    print(f"\nNext step: run scripts/build_instructions.py")


if __name__ == "__main__":
    run_cleaning_pipeline()