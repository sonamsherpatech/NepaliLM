"""
NepaliLM — News Scraper
Scrapes Nepali news articles from Kantipur and Nagarik.
Run this to supplement HuggingFace datasets with fresh data.

Usage:
    pip install requests beautifulsoup4
    python scripts/scrape_news.py
"""

import json
import time
import random
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from datetime import datetime

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; NepaliLM-DataBot/1.0; research use)"
}

DELAY_RANGE = (1.5, 3.0)   # polite crawl delay in seconds


# ── Kantipur ──────────────────────────────────────────────────────────────────

KANTIPUR_SECTIONS = [
    "https://ekantipur.com/news",
    "https://ekantipur.com/national",
    "https://ekantipur.com/economy",
    "https://ekantipur.com/technology",
]

def scrape_kantipur_article(url: str) -> dict | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        title_el = soup.find("h1")
        title    = title_el.get_text(strip=True) if title_el else ""

        # Article body — Kantipur uses .article-content or .detail-content
        body_el = (soup.find(class_="article-content") or
                   soup.find(class_="detail-content") or
                   soup.find("article"))
        if not body_el:
            return None

        paragraphs = [p.get_text(strip=True) for p in body_el.find_all("p")]
        text = "\n".join(p for p in paragraphs if len(p) > 20)

        if len(text) < 100:
            return None

        return {"title": title, "text": text, "url": url, "source": "kantipur"}

    except Exception as e:
        print(f"  Error scraping {url}: {e}")
        return None


def get_kantipur_article_links(section_url: str, max_links: int = 20) -> list[str]:
    try:
        r = requests.get(section_url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/news/" in href or "/national/" in href or "/economy/" in href:
                full = href if href.startswith("http") else "https://ekantipur.com" + href
                if full not in links:
                    links.append(full)
            if len(links) >= max_links:
                break
        return links
    except Exception as e:
        print(f"  Error fetching links from {section_url}: {e}")
        return []


# ── Nagarik ───────────────────────────────────────────────────────────────────

NAGARIK_SECTIONS = [
    "https://nagariknews.com/category/national",
    "https://nagariknews.com/category/economy",
    "https://nagariknews.com/category/society",
]

def scrape_nagarik_article(url: str) -> dict | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        title_el = soup.find("h1")
        title    = title_el.get_text(strip=True) if title_el else ""

        body_el = (soup.find(class_="article-body") or
                   soup.find(class_="entry-content") or
                   soup.find("article"))
        if not body_el:
            return None

        paragraphs = [p.get_text(strip=True) for p in body_el.find_all("p")]
        text = "\n".join(p for p in paragraphs if len(p) > 20)

        if len(text) < 100:
            return None

        return {"title": title, "text": text, "url": url, "source": "nagarik"}

    except Exception as e:
        print(f"  Error scraping {url}: {e}")
        return None


def get_nagarik_article_links(section_url: str, max_links: int = 20) -> list[str]:
    try:
        r = requests.get(section_url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "nagariknews.com" in href and "/20" in href:
                if href not in links:
                    links.append(href)
            if len(links) >= max_links:
                break
        return links
    except Exception as e:
        print(f"  Error fetching links: {e}")
        return []


# ── Main scraper ──────────────────────────────────────────────────────────────

def run_scraper(max_articles_per_source: int = 200):
    print("NepaliLM — News Scraper")
    print("="*50)
    print("Note: scraping ~200 articles per source with polite delays")

    output_path = RAW_DIR / "scraped_news.jsonl"
    total = 0

    with open(output_path, "w", encoding="utf-8") as fout:

        # Kantipur
        print("\nScraping Kantipur...")
        kantipur_count = 0
        for section in KANTIPUR_SECTIONS:
            if kantipur_count >= max_articles_per_source:
                break
            links = get_kantipur_article_links(section, max_links=50)
            for url in links:
                if kantipur_count >= max_articles_per_source:
                    break
                article = scrape_kantipur_article(url)
                if article:
                    fout.write(json.dumps(article, ensure_ascii=False) + "\n")
                    kantipur_count += 1
                    if kantipur_count % 20 == 0:
                        print(f"  Kantipur: {kantipur_count} articles")
                time.sleep(random.uniform(*DELAY_RANGE))
        print(f"  Kantipur total: {kantipur_count}")
        total += kantipur_count

        # Nagarik
        print("\nScraping Nagarik...")
        nagarik_count = 0
        for section in NAGARIK_SECTIONS:
            if nagarik_count >= max_articles_per_source:
                break
            links = get_nagarik_article_links(section, max_links=50)
            for url in links:
                if nagarik_count >= max_articles_per_source:
                    break
                article = scrape_nagarik_article(url)
                if article:
                    fout.write(json.dumps(article, ensure_ascii=False) + "\n")
                    nagarik_count += 1
                    if nagarik_count % 20 == 0:
                        print(f"  Nagarik: {nagarik_count} articles")
                time.sleep(random.uniform(*DELAY_RANGE))
        print(f"  Nagarik total: {nagarik_count}")
        total += nagarik_count

    print(f"\nTotal scraped: {total} articles → {output_path}")
    print("Next: run scripts/clean_data.py to include this in your clean dataset")


if __name__ == "__main__":
    run_scraper(max_articles_per_source=200)