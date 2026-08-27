"""
NepaliLM — Task 1c: Build Instruction Pairs
Converts cleaned Nepali text into instruction-response JSONL format for SFT.

Two strategies:
  1. Extract natural Q&A from text (news articles, Wikipedia)
  2. Generate synthetic pairs using an existing LLM (optional, needs API key)
"""

import re
import json
import random
from pathlib import Path
from typing import Optional

CLEANED_DIR     = Path("data/cleaned")
INSTRUCTIONS_DIR = Path("data/instructions")
INSTRUCTIONS_DIR.mkdir(parents=True, exist_ok=True)

# ── Prompt template ───────────────────────────────────────────────────────────
# This EXACT template must be used at inference time too.

PROMPT_TEMPLATE = """तलको प्रश्नको उत्तर स्पष्ट र सही नेपालीमा दिनुहोस्।

प्रश्न: {question}

उत्तर:"""


def format_pair(question: str, answer: str, source: str = "unknown") -> dict:
    """Format a Q&A pair into SFTTrainer-compatible format."""
    return {
        "prompt":   PROMPT_TEMPLATE.format(question=question.strip()),
        "response": answer.strip(),
        "text":     PROMPT_TEMPLATE.format(question=question.strip()) + " " + answer.strip(),
        "source":   source,
    }


# ── Strategy 1: Extract from Wikipedia ───────────────────────────────────────

def extract_first_paragraph(text: str) -> Optional[str]:
    """Extract a clean first paragraph from Wikipedia-style text."""
    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 80]
    if paragraphs:
        return paragraphs[0][:600]
    return None


def wiki_to_instruction_pairs(text: str, title: str = "", source: str = "wikipedia") -> list[dict]:
    """
    Convert a Wikipedia article into instruction pairs.
    Pattern: "X के हो?" (What is X?) → first paragraph answer
    """
    pairs = []
    para = extract_first_paragraph(text)
    if not para:
        return pairs

    if title:
        # "X के हो?" pattern
        pairs.append(format_pair(f"{title} के हो?", para, source=source))
        # "X बारे बताउनुस्" pattern
        pairs.append(format_pair(f"{title} बारे संक्षेपमा बताउनुस्।", para, source=source))

    return pairs


# ── Strategy 2: Extract from News articles ───────────────────────────────────

NEWS_QUESTION_TEMPLATES = [
    "यो समाचारको मुख्य बिन्दु के हो?",
    "यस लेखमा के भनिएको छ?",
    "यो घटनाको बारेमा के जानकारी छ?",
]

def news_to_instruction_pairs(text: str, source: str = "nepali_news") -> list[dict]:
    """Convert a news article into instruction pairs."""
    pairs = []
    # Take first 400 chars as summary
    summary = text[:400].strip()
    if len(summary) < 80:
        return pairs

    question = random.choice(NEWS_QUESTION_TEMPLATES)
    pairs.append(format_pair(question, summary, source=source))
    return pairs


# ── Strategy 3: Manual seed pairs ────────────────────────────────────────────
# High-quality hand-crafted pairs that anchor the model's behavior.
# Add more here as you build your dataset.

SEED_PAIRS = [
    {
        "question": "नेपालको राजधानी कहाँ छ?",
        "answer":   "नेपालको राजधानी काठमाडौं हो। यो नेपालको सबैभन्दा ठूलो शहर पनि हो र देशको राजनीतिक, आर्थिक तथा सांस्कृतिक केन्द्र हो।"
    },
    {
        "question": "नेपालको राष्ट्रिय भाषा कुन हो?",
        "answer":   "नेपालको राष्ट्रिय भाषा नेपाली हो। यो इन्डो-आर्यन भाषा परिवारमा पर्दछ र देवनागरी लिपिमा लेखिन्छ।"
    },
    {
        "question": "हिमालय पर्वत श्रृंखला कहाँ अवस्थित छ?",
        "answer":   "हिमालय पर्वत श्रृंखला दक्षिण एसियामा अवस्थित छ। यसले नेपाल, भारत, भुटान, चीन र पाकिस्तानसम्म फैलिएको छ। संसारको सबैभन्दा अग्लो शिखर सगरमाथा (८,८४८.८६ मिटर) पनि यही पर्वत श्रृंखलामा पर्छ।"
    },
    {
        "question": "नेपालमा कति वटा प्रदेश छन्?",
        "answer":   "नेपालमा ७ वटा प्रदेश छन्। ती हुन्: कोशी प्रदेश, मधेश प्रदेश, बागमती प्रदेश, गण्डकी प्रदेश, लुम्बिनी प्रदेश, कर्णाली प्रदेश र सुदूरपश्चिम प्रदेश।"
    },
    {
        "question": "सगरमाथाको उचाइ कति छ?",
        "answer":   "सगरमाथाको उचाइ ८,८४८.८६ मिटर (२९,०३१.७ फिट) छ। यो संसारको सबैभन्दा अग्लो पर्वत शिखर हो र नेपाल र चीनको सीमामा अवस्थित छ।"
    },
    {
        "question": "नेपालमा कुन कुन चाडपर्वहरू मनाइन्छन्?",
        "answer":   "नेपालमा दशैं, तिहार, छठ, होली, बुद्ध जयन्ती, इन्द्रजात्रा, गाईजात्रा लगायत अनेक चाडपर्वहरू मनाइन्छन्। दशैं र तिहार नेपालका सबैभन्दा ठूला चाडपर्वहरू हुन्।"
    },
    {
        "question": "नेपालको मुद्रा के हो?",
        "answer":   "नेपालको मुद्रा नेपाली रुपैयाँ हो। यसलाई संक्षेपमा NPR भनिन्छ। नेपाल राष्ट्र बैंक नेपालको केन्द्रीय बैंक हो जसले मुद्रा नीति सञ्चालन गर्छ।"
    },
    {
        "question": "कृषि भनेको के हो?",
        "answer":   "कृषि भनेको जमिनमा बाली लगाई खाद्यान्न, फलफूल, तरकारी तथा अन्य उत्पादन गर्ने काम हो। नेपालको अर्थतन्त्रमा कृषिको महत्त्वपूर्ण भूमिका छ र अधिकांश जनसंख्या कृषिमा निर्भर छ।"
    },
    {
        "question": "इन्टरनेट भनेको के हो र यसको उपयोग के छ?",
        "answer":   "इन्टरनेट भनेको विश्वभरका कम्प्युटर र उपकरणहरूलाई जोड्ने विशाल संजाल हो। यसको माध्यमबाट सूचना आदानप्रदान, सञ्चार, व्यापार, शिक्षा र मनोरञ्जन गर्न सकिन्छ।"
    },
    {
        "question": "जलवायु परिवर्तन भनेको के हो?",
        "answer":   "जलवायु परिवर्तन भनेको पृथ्वीको तापक्रम र मौसम ढाँचामा दीर्घकालीन परिवर्तन हो। मानवीय गतिविधिहरू जस्तै जीवाश्म इन्धन जलाउनु, वनफडानी आदिका कारण यो समस्या बढ्दो छ।"
    },
]


def build_seed_pairs() -> list[dict]:
    return [format_pair(p["question"], p["answer"], source="seed") for p in SEED_PAIRS]


# ── Main pipeline ─────────────────────────────────────────────────────────────

def build_instruction_dataset():
    print("NepaliLM — Building Instruction Pairs")
    print("="*50)

    all_pairs = []

    # 1. Seed pairs (always include — high quality anchor)
    seed = build_seed_pairs()
    all_pairs.extend(seed)
    print(f"Seed pairs:         {len(seed):>6,}")

    # 2. Extract from cleaned Wikipedia
    wiki_path = CLEANED_DIR / "wikipedia_nepali.jsonl"
    wiki_pairs = []
    if wiki_path.exists():
        with open(wiki_path, encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line.strip())
                title = sample.get("title", "").strip()
                body  = sample.get("text", "").strip()
                source = sample.get("source", "wikipedia")
                pairs = wiki_to_instruction_pairs(body, title, source=source)
                wiki_pairs.extend(pairs)
    all_pairs.extend(wiki_pairs)
    print(f"Wikipedia pairs:    {len(wiki_pairs):>6,}")

    # 3. Extract from cleaned news
    news_path = CLEANED_DIR / "nepali_news.jsonl"
    news_pairs = []
    if news_path.exists():
        with open(news_path, encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line.strip())
                source = sample.get("source", "nepali_news")
                pairs  = news_to_instruction_pairs(sample["text"], source=source)
                news_pairs.extend(pairs)
    all_pairs.extend(news_pairs)
    print(f"News pairs:         {len(news_pairs):>6,}")

    # 4. CC-100 is raw text — include as continued pre-training data separately
    cc100_path = CLEANED_DIR / "cc100_nepali.jsonl"
    cc100_pairs = []
    if cc100_path.exists():
        with open(cc100_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 50_000:   # cap for instruction set
                    break
                sample = json.loads(line.strip())
                # Wrap as simple completion instruction
                text = sample["text"][:300]
                source = sample.get("source", "cc100")
                cc100_pairs.append(format_pair(
                    "तलको अनुच्छेदलाई नेपालीमा व्याख्या गर्नुहोस्।\n\n" + text[:150],
                    text,
                    source=source
                ))
    all_pairs.extend(cc100_pairs)
    print(f"CC-100 pairs:       {len(cc100_pairs):>6,}")

    # Shuffle
    random.shuffle(all_pairs)

    # Train / validation split (95/5)
    split = int(len(all_pairs) * 0.95)
    train_pairs = all_pairs[:split]
    val_pairs   = all_pairs[split:]

    # Save
    train_path = INSTRUCTIONS_DIR / "train.jsonl"
    val_path   = INSTRUCTIONS_DIR / "val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for pair in train_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for pair in val_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\n{'='*50}")
    print(f"Train set: {len(train_pairs):,} pairs → {train_path}")
    print(f"Val set:   {len(val_pairs):,} pairs   → {val_path}")
    print(f"Total:     {len(all_pairs):,} pairs")
    print(f"\nNext step: run scripts/verify_dataset.py")
    print(f"Then:      training/train.py")


if __name__ == "__main__":
    build_instruction_dataset()