"""
NepaliLM — Task 1a: Data Collection
Downloads CC-100 Nepali and Wikipedia Nepali datasets from HuggingFace.
"""

import os
import json
import urllib.request
import lzma
from pathlib import Path
from datasets import load_dataset

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_cc100(max_samples: int = 500_000):
    """Download CC-100 Nepali subset from raw URL directly."""
    print("Downloading CC-100 Nepali...")
    url = "https://data.statmt.org/cc-100/ne.txt.xz"
    output_path = RAW_DIR / "cc100_nepali.jsonl"
    count = 0

    try:
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        with urllib.request.urlopen(req) as response:
            with lzma.open(response, mode="rt", encoding="utf-8") as f:
                with open(output_path, "w", encoding="utf-8") as out_f:
                    for line in f:
                        text = line.strip()
                        if not text:
                            continue
                        out_f.write(json.dumps({"text": text, "source": "cc100"}, ensure_ascii=False) + "\n")
                        count += 1
                        if count % 10_000 == 0:
                            print(f"  {count:,} samples saved...")
                        if count >= max_samples:
                            break
    except Exception as e:
        print(f"Error downloading or processing CC-100: {e}")
        raise e

    print(f"CC-100: saved {count:,} samples → {output_path}")
    return count


def download_wikipedia(max_samples: int = 100_000):
    """Download Nepali Wikipedia from HuggingFace."""
    print("\nDownloading Nepali Wikipedia...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.ne", split="train", streaming=True)

    output_path = RAW_DIR / "wikipedia_nepali.jsonl"
    count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in ds:
            # Save title and text separately
            f.write(json.dumps({"title": sample["title"], "text": sample["text"], "source": "wikipedia"}, ensure_ascii=False) + "\n")
            count += 1
            if count % 5_000 == 0:
                print(f"  {count:,} articles saved...")
            if count >= max_samples:
                break

    print(f"Wikipedia: saved {count:,} articles → {output_path}")
    return count


def download_nepali_news():
    """Download Nepali news dataset from HuggingFace."""
    print("\nDownloading Nepali news dataset...")
    try:
        ds = load_dataset("caspro/Nepali_News_Dataset", split="train")
        output_path = RAW_DIR / "nepali_news.jsonl"
        count = 0

        with open(output_path, "w", encoding="utf-8") as f:
            for sample in ds:
                text = sample.get("body") or sample.get("text") or sample.get("content") or ""
                if text.strip():
                    f.write(json.dumps({"text": text, "source": "nepali_news"}, ensure_ascii=False) + "\n")
                    count += 1

        print(f"News: saved {count:,} articles → {output_path}")
        return count
    except Exception as e:
        print(f"News dataset unavailable: {e}")
        return 0


def print_summary():
    print("\n" + "="*50)
    print("DATA COLLECTION SUMMARY")
    print("="*50)
    total = 0
    for f in RAW_DIR.glob("*.jsonl"):
        lines = sum(1 for _ in open(f, encoding="utf-8"))
        size_mb = f.stat().st_size / 1_000_000
        print(f"  {f.name:<30} {lines:>8,} samples   {size_mb:>6.1f} MB")
        total += lines
    print(f"  {'TOTAL':<30} {total:>8,} samples")
    print("="*50)


if __name__ == "__main__":
    print("NepaliLM — Data Collection Pipeline")
    print("="*50)

    download_cc100(max_samples=500_000)
    download_wikipedia(max_samples=100_000)
    download_nepali_news()
    print_summary()

    print("\nNext step: run scripts/clean_data.py")