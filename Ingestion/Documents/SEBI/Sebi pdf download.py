"""
SEBI PDF Downloader  (v1)
Reads the JSON produced by sebi_legal_scraper.py,
opens each detail page, extracts the PDF URL from the iframe,
downloads the PDF — 7 workers running in parallel.

Usage:
    python sebi_pdf_downloader.py                        # picks latest JSON
    python sebi_pdf_downloader.py sebi_legal_links.json  # explicit file
"""

import json
import os
import re
import sys
import time
import pathlib
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup


# ─── Configuration ────────────────────────────────────────────────────────────

WORKERS         = 7            # parallel downloads
OUTPUT_DIR      = "sebi_pdfs"  # folder where PDFs are saved
REQUEST_TIMEOUT = 30           # seconds per HTTP request
RETRY_COUNT     = 3            # retries on failure
RETRY_DELAY     = 5            # seconds between retries

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Referer": "https://www.sebi.gov.in/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def slugify(text: str, max_len: int = 100) -> str:
    """Convert a title to a safe filename."""
    text = re.sub(r"[^\w\s\-]", "", text)
    text = re.sub(r"[\s]+", "_", text.strip())
    return text[:max_len]


def find_json(explicit: str | None = None) -> str:
    """Return path to the JSON file (explicit or latest in cwd)."""
    if explicit and os.path.isfile(explicit):
        return explicit
    # find the most recently modified sebi_legal_links_*.json in cwd
    candidates = sorted(
        pathlib.Path(".").glob("sebi_legal_links_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No sebi_legal_links_*.json found in current directory. "
            "Pass the filename as a command-line argument."
        )
    return str(candidates[0])


def extract_pdf_url(detail_url: str, session: requests.Session) -> str | None:
    """
    Fetch a SEBI detail page and extract the PDF URL from the iframe.
    iframe src looks like:
      ../../../web/?file=https://www.sebi.gov.in/sebi_data/attachdocs/...pdf
    or
      https://www.sebi.gov.in/web/?file=https://...pdf
    """
    resp = session.get(detail_url, timeout=REQUEST_TIMEOUT, headers=HEADERS)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # ── Strategy 1: iframe with ?file= param ─────────────────────────────────
    for iframe in soup.find_all("iframe"):
        src = iframe.get("src", "")
        if "?file=" in src or "&file=" in src:
            parsed = urllib.parse.urlparse(src)
            params = urllib.parse.parse_qs(parsed.query)
            file_param = params.get("file", [None])[0]
            if file_param and file_param.lower().endswith(".pdf"):
                return file_param

    # ── Strategy 2: direct <a> link to a PDF ─────────────────────────────────
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".pdf") and "sebi.gov.in" in href:
            return href

    # ── Strategy 3: any src/href that ends with .pdf ──────────────────────────
    for tag in soup.find_all(["iframe", "embed", "object", "a"]):
        for attr in ("src", "href", "data"):
            val = tag.get(attr, "")
            if val.lower().endswith(".pdf"):
                if val.startswith("http"):
                    return val
                # resolve relative URL
                return urllib.parse.urljoin(detail_url, val)

    return None


def download_pdf(
    pdf_url: str,
    dest_path: str,
    session: requests.Session,
) -> bool:
    """Download a PDF to dest_path. Returns True on success."""
    resp = session.get(pdf_url, timeout=REQUEST_TIMEOUT,
                       stream=True, headers=HEADERS)
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "")
    if "text/html" in content_type:
        # SEBI sometimes redirects PDF requests to HTML login pages
        return False

    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)

    size_kb = os.path.getsize(dest_path) / 1024
    if size_kb < 1:           # probably an error page
        os.remove(dest_path)
        return False
    return True


def process_record(record: dict, out_dir: str) -> dict:
    """
    Full pipeline for one record:
      1. Fetch detail page
      2. Extract PDF URL
      3. Download PDF
    Returns a status dict.
    """
    url    = record["url"]
    title  = record.get("title", "untitled")
    date   = record.get("date", "").replace(" ", "_").replace(",", "")
    section = record.get("section", "misc")

    # Build destination path
    section_dir = os.path.join(out_dir, slugify(section))
    filename    = f"{date}_{slugify(title)}.pdf"
    dest        = os.path.join(section_dir, filename)

    # Skip already downloaded
    if os.path.isfile(dest) and os.path.getsize(dest) > 1024:
        return {"status": "skipped", "title": title, "dest": dest}

    session = requests.Session()
    session.headers.update(HEADERS)

    for attempt in range(1, RETRY_COUNT + 1):
        try:
            pdf_url = extract_pdf_url(url, session)
            if not pdf_url:
                return {"status": "no_pdf", "title": title, "url": url}

            ok = download_pdf(pdf_url, dest, session)
            if ok:
                size_kb = os.path.getsize(dest) / 1024
                return {
                    "status": "ok",
                    "title": title,
                    "dest": dest,
                    "pdf_url": pdf_url,
                    "size_kb": round(size_kb, 1),
                }
            else:
                return {"status": "bad_response", "title": title, "pdf_url": pdf_url}

        except Exception as exc:
            if attempt < RETRY_COUNT:
                time.sleep(RETRY_DELAY)
            else:
                return {"status": "error", "title": title, "url": url, "error": str(exc)}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # ── Pick JSON file ────────────────────────────────────────────────────────
    json_path = sys.argv[1] if len(sys.argv) > 1 else None
    json_file = find_json(json_path)
    print(f"\nSEBI PDF Downloader")
    print(f"Input  : {json_file}")
    print(f"Output : {OUTPUT_DIR}/")
    print(f"Workers: {WORKERS}")

    with open(json_file, encoding="utf-8") as f:
        records = json.load(f)

    print(f"Records: {len(records)}\n")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Stats ─────────────────────────────────────────────────────────────────
    stats = {"ok": 0, "skipped": 0, "no_pdf": 0, "bad_response": 0, "error": 0}

    # ── Run with 7 workers ────────────────────────────────────────────────────
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            pool.submit(process_record, rec, OUTPUT_DIR): rec
            for rec in records
        }

        done = 0
        total = len(futures)

        for future in as_completed(futures):
            done += 1
            result = future.result()
            status = result.get("status", "error")
            stats[status] = stats.get(status, 0) + 1

            # Progress line
            title_short = result.get("title", "")[:60]
            if status == "ok":
                print(
                    f"  [{done:>4}/{total}] ✓ {title_short}"
                    f"  ({result.get('size_kb', '?')} KB)"
                )
            elif status == "skipped":
                print(f"  [{done:>4}/{total}] ~ SKIP  {title_short}")
            elif status == "no_pdf":
                print(f"  [{done:>4}/{total}] ✗ NO_PDF  {title_short}")
            else:
                err = result.get("error", "")[:80]
                print(f"  [{done:>4}/{total}] ✗ {status.upper()}  {title_short}  [{err}]")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:<15} {v:>5}")
    print(f"  {'TOTAL':<15} {total:>5}")
    print("=" * 60)
    print(f"\nPDFs saved to: {os.path.abspath(OUTPUT_DIR)}/")


if __name__ == "__main__":
    main() 