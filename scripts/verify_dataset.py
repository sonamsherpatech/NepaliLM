"""
NepaliLM — Task 1d: Dataset Verification
Spot-checks your dataset before you spend money on training.
Prints samples, checks Devanagari purity, length distribution.
"""

import json
import random
import unicodedata
from pathlib import Path
from collections import Counter

INSTRUCTIONS_DIR = Path("data/instructions")
CLEANED_DIR      = Path("data/cleaned")


def devanagari_ratio(text: str) -> float:
    non_ws = [c for c in text if not c.isspace()]
    if not non_ws:
        return 0.0
    dev = sum(1 for c in non_ws if 0x0900 <= ord(c) <= 0x097F)
    return dev / len(non_ws)


def check_dataset(path: Path, n_samples: int = 5):
    print(f"\n{'='*60}")
    print(f"Checking: {path.name}")
    print(f"{'='*60}")

    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line.strip()))

    print(f"Total pairs: {len(pairs):,}")

    # Length stats
    prompt_lens   = [len(p["prompt"])   for p in pairs]
    response_lens = [len(p["response"]) for p in pairs]

    print(f"\nPrompt length   — min: {min(prompt_lens)}, avg: {sum(prompt_lens)//len(prompt_lens)}, max: {max(prompt_lens)}")
    print(f"Response length — min: {min(response_lens)}, avg: {sum(response_lens)//len(response_lens)}, max: {max(response_lens)}")

    # Devanagari purity
    dev_ratios = [devanagari_ratio(p["response"]) for p in pairs]
    avg_ratio  = sum(dev_ratios) / len(dev_ratios)
    low_dev    = sum(1 for r in dev_ratios if r < 0.5)
    print(f"\nDevanagari ratio — avg: {avg_ratio:.2%}, low-quality (<50%): {low_dev}")

    # Source distribution
    sources = Counter(p.get("source", "unknown") for p in pairs)
    if any(sources.values()):
        print(f"\nSource distribution:")
        for src, count in sources.most_common():
            print(f"  {src:<20} {count:>8,}")

    # Random samples
    print(f"\n{'─'*60}")
    print(f"Random samples ({n_samples}):")
    print(f"{'─'*60}")
    for i, pair in enumerate(random.sample(pairs, min(n_samples, len(pairs)))):
        print(f"\n[Sample {i+1}]")
        print(f"PROMPT:   {pair['prompt'][:120]}...")
        print(f"RESPONSE: {pair['response'][:200]}")
        ratio = devanagari_ratio(pair['response'])
        print(f"Dev ratio: {ratio:.2%}")


def run_verification():
    print("NepaliLM — Dataset Verification")

    train_path = INSTRUCTIONS_DIR / "train.jsonl"
    val_path   = INSTRUCTIONS_DIR / "val.jsonl"

    if not train_path.exists():
        print("No dataset found. Run scripts/build_instructions.py first.")
        return

    check_dataset(train_path, n_samples=5)
    check_dataset(val_path,   n_samples=2)

    print(f"\n{'='*60}")
    print("VERIFICATION COMPLETE")
    print("If samples look good → proceed to training/train.py")
    print("If samples have Hindi/Roman → re-run clean_data.py with stricter filters")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_verification()