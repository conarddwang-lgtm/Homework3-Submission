"""
Web Scraping Utilities -- Week 3

Functions for extracting clean text from web pages using
trafilatura (traditional) and Crawl4AI (modern LLM-ready).

Functions:
  extract_with_trafilatura(url)    - Extract text using trafilatura
  extract_with_crawl4ai(url)       - Extract LLM-ready markdown with Crawl4AI
  scrape_arxiv_abstracts(topic, n) - Scrape arXiv paper abstracts
  compare_extractors(url)          - Side-by-side comparison of both tools
"""

import os
import json
from typing import Dict, List, Optional, Any


def extract_with_trafilatura(url: str, include_tables: bool = False) -> Dict[str, Any]:
    """
    Extract clean text from a URL using trafilatura.

    Args:
        url: Web page URL
        include_tables: Whether to include table content

    Returns:
        Dict with 'text', 'url', 'method', 'char_count' keys
    """
    try:
        import trafilatura
    except ImportError:
        raise ImportError("Install: pip install trafilatura")

    import requests
    import time

    print(f"[trafilatura] Fetching: {url}")
    start = time.time()

    response = requests.get(url, timeout=15, headers={
        'User-Agent': 'Mozilla/5.0 (research-bot)'
    })
    html = response.text

    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=include_tables,
    )

    elapsed = time.time() - start

    result = {
        "text": text or "",
        "url": url,
        "method": "trafilatura",
        "char_count": len(text) if text else 0,
        "elapsed_seconds": round(elapsed, 2),
    }

    print(f"  Extracted {result['char_count']:,} chars in {elapsed:.1f}s")
    return result


def extract_with_crawl4ai(url: str) -> Dict[str, Any]:
    """
    Extract LLM-ready markdown from a URL using Crawl4AI.

    Crawl4AI produces clean markdown optimized for LLM consumption,
    automatically removing navigation, ads, and boilerplate.

    NOTE: Crawl4AI requires Python 3.10+. If unavailable, falls back
    to html2text for markdown conversion.

    Args:
        url: Web page URL

    Returns:
        Dict with 'text', 'url', 'method', 'char_count' keys
    """
    import sys
    import time

    # Try Crawl4AI first (requires Python 3.10+)
    if sys.version_info >= (3, 10):
        try:
            from crawl4ai import AsyncWebCrawler
            import asyncio

            print(f"[Crawl4AI] Fetching: {url}")
            start = time.time()

            async def _crawl():
                async with AsyncWebCrawler() as crawler:
                    result = await crawler.arun(url=url)
                    return result.markdown if result.success else ""

            try:
                loop = asyncio.get_running_loop()
                import nest_asyncio
                nest_asyncio.apply()
                text = loop.run_until_complete(_crawl())
            except RuntimeError:
                text = asyncio.run(_crawl())

            elapsed = time.time() - start
            result = {
                "text": text or "",
                "url": url,
                "method": "crawl4ai",
                "char_count": len(text) if text else 0,
                "elapsed_seconds": round(elapsed, 2),
            }
            print(f"  Extracted {result['char_count']:,} chars in {elapsed:.1f}s")
            return result

        except ImportError:
            print("  [Crawl4AI not installed -- falling back to html2text]")

    # Fallback: requests + html2text for markdown conversion
    return _extract_as_markdown(url)


def _extract_as_markdown(url: str) -> Dict[str, Any]:
    """
    Fallback markdown extractor using html2text (works on Python 3.9+).

    Converts HTML to clean Markdown, preserving links and structure --
    similar to what Crawl4AI does but without browser rendering.
    """
    import time

    try:
        import html2text
    except ImportError:
        raise ImportError("Install: pip install html2text")

    import requests as req

    print(f"[html2text] Fetching: {url}")
    start = time.time()

    resp = req.get(url, timeout=15, headers={
        'User-Agent': 'Mozilla/5.0 (research-bot)'
    })

    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = True
    h.ignore_emphasis = False
    h.body_width = 0  # Don't wrap lines
    h.skip_internal_links = True

    text = h.handle(resp.text).strip()

    elapsed = time.time() - start

    result = {
        "text": text,
        "url": url,
        "method": "html2text (markdown fallback)",
        "char_count": len(text),
        "elapsed_seconds": round(elapsed, 2),
    }

    print(f"  Extracted {result['char_count']:,} chars in {elapsed:.1f}s")
    return result


def scrape_arxiv_abstracts(
    topic: str = "large language models",
    max_results: int = 5,
    save_path: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Scrape arXiv paper abstracts for a given topic.

    Args:
        topic: Search query (e.g., "NLP", "AI safety", "robotics")
        max_results: Number of papers to fetch
        save_path: Path to save results as JSON (None to skip)

    Returns:
        List of dicts with 'title', 'abstract', 'url', 'authors' keys
    """
    import requests
    import xml.etree.ElementTree as ET
    import time

    print("=" * 55)
    print(f"Scraping arXiv: '{topic}' (max {max_results} papers)")
    print("=" * 55)

    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{topic}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    response = requests.get(base_url, params=params, timeout=30)
    root = ET.fromstring(response.text)

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    papers = []

    for entry in root.findall("atom:entry", ns):
        title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
        abstract = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
        url = entry.find("atom:id", ns).text.strip()
        authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]

        papers.append({
            "title": title,
            "abstract": abstract,
            "url": url,
            "authors": authors,
        })

        print(f"\n  [{len(papers)}] {title[:80]}...")
        print(f"      Authors: {', '.join(authors[:3])}{'...' if len(authors) > 3 else ''}")
        print(f"      Abstract: {abstract[:120]}...")
        time.sleep(0.5)  # Be polite to arXiv API

    print(f"\n  Total papers collected: {len(papers)}")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        print(f"  Saved to: {save_path}")

    return papers


def compare_extractors(url: str) -> Dict[str, Any]:
    """
    Side-by-side comparison of trafilatura vs Crawl4AI on the same URL.

    Args:
        url: Web page URL to test

    Returns:
        Dict with results from both extractors and comparison metrics
    """
    print("=" * 55)
    print("EXTRACTOR COMPARISON")
    print("=" * 55)
    print(f"URL: {url}\n")

    traf_result = extract_with_trafilatura(url)

    try:
        c4ai_result = extract_with_crawl4ai(url)
    except ImportError:
        print("  [Crawl4AI not installed -- skipping]")
        c4ai_result = {"text": "", "char_count": 0, "elapsed_seconds": 0, "method": "crawl4ai (not installed)"}

    c4ai_label = c4ai_result.get('method', 'crawl4ai')
    print(f"\n--- Comparison ---")
    print(f"  trafilatura:  {traf_result['char_count']:,} chars in {traf_result['elapsed_seconds']:.1f}s")
    print(f"  {c4ai_label}: {c4ai_result['char_count']:,} chars in {c4ai_result['elapsed_seconds']:.1f}s")

    return {
        "url": url,
        "trafilatura": traf_result,
        "crawl4ai": c4ai_result,
    }
