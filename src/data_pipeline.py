"""
Data Cleaning Pipeline -- Week 3

Functions implementing a pretraining data cleaning pipeline:
deduplication, language filtering, PII removal, and quality filtering.

Functions:
  minhash_deduplication(texts)     - Remove duplicates via MinHash LSH
  filter_by_language(texts, lang)  - Keep only texts in target language
  strip_pii(text)                  - Remove emails, phone numbers, credit cards
  remove_html_noise(text)          - Strip HTML tags and boilerplate
  remove_repetitive_ngrams(text)   - Remove spam-like repetitive content
  run_cleaning_pipeline(texts)     - Run the full pipeline end-to-end
  show_pipeline_summary(raw, clean)- Display before/after statistics
"""

import os
import re
import json
from typing import Dict, List, Optional, Any
from collections import Counter


def minhash_deduplication(
    texts: List[str],
    threshold: float = 0.7,
    num_perm: int = 128,
) -> List[str]:
    """
    Remove near-duplicate texts using MinHash LSH.

    This is the same technique used by Meta (LLaMA), HuggingFace (FineWeb),
    and other major pretraining pipelines to deduplicate web crawl data.

    Args:
        texts: List of text documents
        threshold: Jaccard similarity threshold (0.7 = 70% similar)
        num_perm: Number of permutations for MinHash (more = accurate but slower)

    Returns:
        Deduplicated list of texts
    """
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        raise ImportError("Install: pip install datasketch")

    print(f"  [Dedup] Input: {len(texts)} documents (threshold={threshold})")

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    unique_texts = []
    duplicates_found = 0

    for i, doc in enumerate(texts):
        m = MinHash(num_perm=num_perm)
        for word in set(doc.split()):
            m.update(word.encode('utf8'))

        if not lsh.query(m):
            lsh.insert(f"doc{i}", m)
            unique_texts.append(doc)
        else:
            duplicates_found += 1

    print(f"  [Dedup] Output: {len(unique_texts)} unique ({duplicates_found} duplicates removed)")
    return unique_texts


def filter_by_language(
    texts: List[str],
    target_lang: str = "en",
) -> List[str]:
    """
    Filter texts to keep only those in the target language.

    Args:
        texts: List of text documents
        target_lang: ISO 639-1 language code (default: "en")

    Returns:
        Filtered list of texts in the target language
    """
    try:
        from langdetect import detect
    except ImportError:
        raise ImportError("Install: pip install langdetect")

    print(f"  [LangFilter] Input: {len(texts)} documents (target: {target_lang})")

    filtered = []
    rejected = 0
    for txt in texts:
        try:
            if detect(txt.strip()) == target_lang:
                filtered.append(txt)
            else:
                rejected += 1
        except Exception:
            rejected += 1

    print(f"  [LangFilter] Output: {len(filtered)} kept ({rejected} rejected)")
    return filtered


def strip_pii(text: str) -> str:
    """
    Remove PII (Personally Identifiable Information) using regex patterns.

    Handles: email addresses, credit card numbers, phone numbers, SSNs.

    Args:
        text: Input text

    Returns:
        Text with PII replaced by placeholder tokens
    """
    # Email addresses
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL]', text)
    # Credit card numbers (12-19 digits)
    text = re.sub(r'\b\d{12,19}\b', '[CREDIT_CARD]', text)
    # US phone numbers (various formats)
    text = re.sub(r'\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b', '[PHONE]', text)
    # SSN-like patterns
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
    # IP addresses
    text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP_ADDR]', text)
    return text


def remove_html_noise(text: str) -> str:
    """
    Remove HTML tags and boilerplate from text.

    Args:
        text: Input text (may contain HTML fragments)

    Returns:
        Clean text without HTML
    """
    try:
        from bs4 import BeautifulSoup
        text = BeautifulSoup(text, 'html.parser').get_text(separator=' ')
    except ImportError:
        # Fallback: regex-based HTML removal
        text = re.sub(r'<[^>]+>', ' ', text)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_repetitive_ngrams(
    text: str,
    n: int = 3,
    threshold: int = 3,
) -> str:
    """
    Remove spam-like repetitive n-gram patterns.

    Args:
        text: Input text
        n: N-gram size
        threshold: Minimum repetition count to trigger removal

    Returns:
        Text with repetitive n-grams collapsed
    """
    words = text.split()
    if len(words) < n:
        return text

    ngrams = [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
    counts = Counter(ngrams)
    repetitive = [ngram for ngram, count in counts.items() if count >= threshold]

    for phrase in repetitive:
        escaped = re.escape(phrase)
        text = re.sub(rf'(?:{escaped}\s*){{{threshold},}}', phrase + ' ', text)

    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text


def quality_filter(
    texts: List[str],
    min_chars: int = 50,
    max_chars: int = 100000,
    min_words: int = 10,
) -> List[str]:
    """
    Filter texts by basic quality heuristics.

    Args:
        texts: List of text documents
        min_chars: Minimum character count
        max_chars: Maximum character count
        min_words: Minimum word count

    Returns:
        Quality-filtered list of texts
    """
    print(f"  [Quality] Input: {len(texts)} documents")

    filtered = []
    for txt in texts:
        txt = txt.strip()
        word_count = len(txt.split())
        char_count = len(txt)

        if char_count < min_chars or char_count > max_chars:
            continue
        if word_count < min_words:
            continue

        filtered.append(txt)

    print(f"  [Quality] Output: {len(filtered)} kept ({len(texts) - len(filtered)} rejected)")
    return filtered


def run_cleaning_pipeline(
    texts: List[str],
    target_lang: str = "en",
    dedup_threshold: float = 0.7,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the full pretraining data cleaning pipeline.

    Pipeline stages:
    1. HTML removal
    2. Language filtering
    3. MinHash deduplication
    4. PII stripping
    5. Repetitive n-gram removal
    6. Quality filtering

    Args:
        texts: Raw text documents
        target_lang: Target language for filtering
        dedup_threshold: MinHash similarity threshold
        save_path: Path to save cleaned data as JSON (None to skip)

    Returns:
        Dict with 'cleaned_texts', 'stats', and per-stage counts
    """
    print("=" * 55)
    print("PRETRAINING DATA CLEANING PIPELINE")
    print("=" * 55)
    print(f"Input: {len(texts)} documents\n")

    stats = {"input_count": len(texts)}

    # Stage 1: HTML removal
    print("Stage 1: HTML noise removal")
    texts = [remove_html_noise(t) for t in texts]
    print(f"  Done ({len(texts)} documents)")

    # Stage 2: Language filtering
    print("\nStage 2: Language filtering")
    texts = filter_by_language(texts, target_lang)
    stats["after_lang_filter"] = len(texts)

    # Stage 3: Deduplication
    print("\nStage 3: MinHash deduplication")
    texts = minhash_deduplication(texts, threshold=dedup_threshold)
    stats["after_dedup"] = len(texts)

    # Stage 4: PII stripping
    print("\nStage 4: PII removal")
    texts = [strip_pii(t) for t in texts]
    print(f"  [PII] Processed {len(texts)} documents")

    # Stage 5: Repetitive n-gram removal
    print("\nStage 5: Repetitive n-gram removal")
    texts = [remove_repetitive_ngrams(t) for t in texts]
    print(f"  [N-gram] Processed {len(texts)} documents")

    # Stage 6: Quality filtering
    print("\nStage 6: Quality filtering")
    texts = quality_filter(texts)
    stats["after_quality"] = len(texts)

    stats["output_count"] = len(texts)
    stats["removal_rate"] = round(1 - len(texts) / stats["input_count"], 3) if stats["input_count"] > 0 else 0

    result = {
        "cleaned_texts": texts,
        "stats": stats,
    }

    print(f"\n--- Pipeline Summary ---")
    print(f"  Input:  {stats['input_count']} documents")
    print(f"  Output: {stats['output_count']} documents")
    print(f"  Removal rate: {stats['removal_rate']*100:.1f}%")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  Saved to: {save_path}")

    return result


def show_pipeline_summary(
    raw_texts: List[str],
    cleaned_texts: List[str],
) -> None:
    """
    Display before/after statistics for the cleaning pipeline.

    Args:
        raw_texts: Original texts
        cleaned_texts: Cleaned texts
    """
    raw_chars = sum(len(t) for t in raw_texts)
    clean_chars = sum(len(t) for t in cleaned_texts)
    raw_words = sum(len(t.split()) for t in raw_texts)
    clean_words = sum(len(t.split()) for t in cleaned_texts)

    print("=" * 55)
    print("PIPELINE BEFORE/AFTER")
    print("=" * 55)
    print(f"  {'Metric':<25} {'Before':>10} {'After':>10} {'Change':>10}")
    print(f"  {'-'*55}")
    print(f"  {'Documents':<25} {len(raw_texts):>10,} {len(cleaned_texts):>10,} {len(cleaned_texts)-len(raw_texts):>+10,}")
    print(f"  {'Total characters':<25} {raw_chars:>10,} {clean_chars:>10,} {clean_chars-raw_chars:>+10,}")
    print(f"  {'Total words':<25} {raw_words:>10,} {clean_words:>10,} {clean_words-raw_words:>+10,}")
    if raw_chars > 0:
        print(f"  {'Char reduction':<25} {'':>10} {'':>10} {(1-clean_chars/raw_chars)*100:>9.1f}%")
    print("=" * 55)
