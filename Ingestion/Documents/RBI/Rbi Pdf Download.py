"""
RBI PDF Downloader — Multi-JSON, 7-Core Parallel
-------------------------------------------------
• Reads 4 JSON files, each mapped to its own output folder
• Uses 7 parallel workers (processes), each with its own headless Chrome
• Each worker: opens the notification page → scrapes PDF URL → downloads via requests

Requirements:
    pip install selenium requests webdriver-manager
"""

import json
import os
import time
import logging
import requests
import multiprocessing as mp
from pathlib import Path
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

try:
    from webdriver_manager.chrome import ChromeDriverManager
    _WDM_AVAILABLE = True
except ImportError:
    _WDM_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent

# Map each JSON file → output folder (all paths relative to this script)
JSON_CONFIGS = [
    {
        "json":   _SCRIPT_DIR / "rbi_aml_links.json",
        "folder": _SCRIPT_DIR / "pdfs" / "AML",
    },
    {
        "json":   _SCRIPT_DIR / "rbi_kyc_links.json",
        "folder": _SCRIPT_DIR / "pdfs" / "KYC",
    },
    {
        "json":   _SCRIPT_DIR / "rbi_beneficial_links.json",
        "folder": _SCRIPT_DIR / "pdfs" / "Beneficial_Ownership",
    },
    {
        "json":   _SCRIPT_DIR / "rbi_shell_links.json",
        "folder": _SCRIPT_DIR / "pdfs" / "shell",
    },
]

NUM_WORKERS       = 7          # parallel Chrome processes
PAGE_TIMEOUT      = 15         # seconds to wait for PDF link on page
DOWNLOAD_TIMEOUT  = 60         # seconds for requests PDF download
POLITE_DELAY      = 1.5        # seconds between requests per worker

PDF_SELECTOR = "a[href$='.PDF'], a[href$='.pdf']"
# ─────────────────────────────────────────────────────────────────────────────


# ── Logging (each process writes its own prefix) ───────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [W%(process)d] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Per-process global driver (initialised once per worker) ───────────────
_driver: webdriver.Chrome | None = None


def _init_worker():
    """Called once when each worker process starts — creates its Chrome driver."""
    global _driver
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1280,900")
    options.add_argument("--log-level=3")           # suppress Chrome noise
    options.add_experimental_option("prefs", {
        "download.prompt_for_download": False,
        "plugins.always_open_pdf_externally": True,
    })
    if _WDM_AVAILABLE:
        service = Service(ChromeDriverManager().install())
    else:
        service = Service()
    _driver = webdriver.Chrome(service=service, options=options)
    log.info("Chrome driver ready.")


def _quit_worker():
    """Gracefully quit the driver when the worker exits."""
    global _driver
    if _driver:
        try:
            _driver.quit()
        except Exception:
            pass


# ── Helpers ────────────────────────────────────────────────────────────────

def sanitize(text: str, max_len: int = 90) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_(). ")
    return "".join(c if c in allowed else "_" for c in text).strip()[:max_len]


def scrape_pdf_url(page_url: str) -> str | None:
    """Use the worker's Chrome instance to find the PDF href on a page."""
    global _driver
    try:
        _driver.get(page_url)
        WebDriverWait(_driver, PAGE_TIMEOUT).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, PDF_SELECTOR))
        )
        links = _driver.find_elements(By.CSS_SELECTOR, PDF_SELECTOR)
        return links[0].get_attribute("href") if links else None
    except TimeoutException:
        return None
    except Exception as exc:
        log.warning(f"Selenium error on {page_url}: {exc}")
        return None


def download_pdf(pdf_url: str, dest: Path) -> bool:
    """Download a PDF via requests. Returns True on success."""
    try:
        resp = requests.get(
            pdf_url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=DOWNLOAD_TIMEOUT,
            stream=True,
        )
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=65536):
                fh.write(chunk)
        return True
    except Exception as exc:
        log.warning(f"Download failed {pdf_url}: {exc}")
        return False


# ── Worker task (one entry per call) ──────────────────────────────────────

def process_task(task: dict) -> dict:
    """
    Executed in a worker process.
    task = {"index": int, "total": int, "title": str,
             "url": str, "date": str, "dest_dir": str}
    Returns a result dict.
    """
    title    = task["title"]
    url      = task["url"]
    date_str = task["date"].replace(" ", "_").replace(",", "")
    dest_dir = Path(task["dest_dir"])
    idx      = task["index"]
    total    = task["total"]

    prefix = f"[{idx:>4}/{total}]"

    if not url:
        log.info(f"{prefix} SKIP (no URL)  {title[:60]}")
        return {"status": "skip", "title": title}

    # Build destination path
    filename = f"{date_str}_{sanitize(title)}.pdf" if date_str else f"{sanitize(title)}.pdf"
    dest = dest_dir / filename

    if dest.exists():
        log.info(f"{prefix} EXIST          {dest.name}")
        return {"status": "exists", "title": title}

    log.info(f"{prefix} Scraping       {url}")
    pdf_url = scrape_pdf_url(url)

    if not pdf_url:
        log.warning(f"{prefix} NO PDF LINK    {url}")
        return {"status": "no_pdf", "title": title}

    log.info(f"{prefix} Downloading    {pdf_url}")
    ok = download_pdf(pdf_url, dest)

    time.sleep(POLITE_DELAY)

    if ok:
        size_kb = dest.stat().st_size // 1024
        log.info(f"{prefix} OK {size_kb:>5} KB   {dest.name}")
        return {"status": "ok", "title": title, "path": str(dest)}
    else:
        return {"status": "fail", "title": title}


# ── Load all JSON files → flat task list ──────────────────────────────────

def load_all_tasks() -> list[dict]:
    tasks = []
    for cfg in JSON_CONFIGS:
        json_path = Path(cfg["json"])
        dest_dir  = Path(cfg["folder"])

        if not json_path.exists():
            log.warning(f"JSON not found, skipping: {json_path}")
            continue

        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        entries = data.get("results", [])
        log.info(f"Loaded {len(entries):>4} entries from {json_path.name} → {dest_dir.name}/")
        dest_dir.mkdir(parents=True, exist_ok=True)

        for entry in entries:
            tasks.append({
                "title":    entry.get("title", "untitled"),
                "url":      entry.get("url", ""),
                "date":     entry.get("date", ""),
                "dest_dir": str(dest_dir),
            })

    # Attach global index after combining all sources
    total = len(tasks)
    for i, t in enumerate(tasks, 1):
        t["index"] = i
        t["total"] = total

    return tasks


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    start = datetime.now()
    log.info("=" * 60)
    log.info("RBI PDF Downloader  |  workers=%d", NUM_WORKERS)
    log.info("=" * 60)

    tasks = load_all_tasks()
    if not tasks:
        log.error("No tasks loaded. Check your JSON file paths in JSON_CONFIGS.")
        return

    log.info(f"Total tasks: {len(tasks)}  |  Starting {NUM_WORKERS} workers …\n")

    # multiprocessing pool — each process gets its own Chrome via _init_worker
    ctx = mp.get_context("spawn")   # 'spawn' is safest on Windows
    with ctx.Pool(
        processes=NUM_WORKERS,
        initializer=_init_worker,
    ) as pool:
        results = pool.map(process_task, tasks, chunksize=1)

    # ── Summary ───────────────────────────────────────────────────────────
    counts = {"ok": 0, "exists": 0, "skip": 0, "no_pdf": 0, "fail": 0}
    for r in results:
        counts[r.get("status", "fail")] += 1

    elapsed = datetime.now() - start
    log.info("\n" + "=" * 60)
    log.info("DONE in %s", str(elapsed).split(".")[0])
    log.info("  ✓ Downloaded : %d", counts["ok"])
    log.info("  ↩ Already existed: %d", counts["exists"])
    log.info("  ✗ Failed     : %d", counts["fail"])
    log.info("  ⚠ No PDF link: %d", counts["no_pdf"])
    log.info("  – Skipped    : %d", counts["skip"])
    log.info("=" * 60)

    # Save failed entries for retry
    failed = [r["title"] for r in results if r.get("status") in ("fail", "no_pdf")]
    if failed:
        fail_log = _SCRIPT_DIR / "failed_downloads.txt"
        fail_log.write_text("\n".join(failed), encoding="utf-8")
        log.info(f"Failed titles saved to: {fail_log}")


if __name__ == "__main__":
    main()