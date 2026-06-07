"""
SEBI Legal Section Scraper  (v3)
Handles two page types:
  A) Pages with "Updated List" / "Historical Data" tabs  (e.g. Regulations)
  B) Pages where search form is directly inside a collapsible Search accordion
     (e.g. Acts, Rules, Guidelines)
"""

import time
import json
import csv
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import (
    TimeoutException, NoSuchElementException, StaleElementReferenceException
)


# ─── Configuration ────────────────────────────────────────────────────────────

BASE_URL = (
    "https://www.sebi.gov.in/sebiweb/home/HomeAction.do"
    "?doListing=yes&sid=1&ssid={ssid}&smid=0"
)

FROM_DATE = "01-01-2025"
TO_DATE   = datetime.today().strftime("%d-%m-%Y")

SECTIONS = {
    "Acts":        "1",
    "Rules":       "2",
    "Regulations": "3",
    "Guidelines":  "5",
}

SHORT_WAIT  = 10   # seconds — for optional elements
LONG_WAIT   = 30   # seconds — for mandatory elements / table results
POST_JS_SLEEP = 4  # seconds after any JS action


# ─── Browser ──────────────────────────────────────────────────────────────────

def build_driver(headless: bool = True) -> webdriver.Chrome:
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
    driver = webdriver.Chrome(options=opts)
    driver.implicitly_wait(3)
    return driver


# ─── Step helpers ─────────────────────────────────────────────────────────────

def _js_set(driver, elem_id: str, value: str):
    """Set an input value via JS and fire change + input events."""
    script = f"""
        var el = document.getElementById('{elem_id}');
        el.value = '{value}';
        el.dispatchEvent(new Event('input',  {{bubbles: true}}));
        el.dispatchEvent(new Event('change', {{bubbles: true}}));
    """
    driver.execute_script(script)


def ensure_search_form_visible(driver: webdriver.Chrome) -> bool:
    """
    Make the fromDate input visible by one of two mechanisms:
      1. Click the 'Historical Data' tab  (Regulations-style pages)
      2. Click the 'Search' accordion toggle  (Acts/Rules-style pages)
    Returns True if fromDate is now visible.
    """
    # Check if fromDate is already visible
    try:
        el = driver.find_element(By.ID, "fromDate")
        if el.is_displayed():
            print("  [OK] Search form already visible.")
            return True
    except NoSuchElementException:
        pass

    # --- Strategy 1: Historical Data tab ---
    try:
        tab = WebDriverWait(driver, SHORT_WAIT).until(
            EC.element_to_be_clickable(
                (By.XPATH, "//a[contains(@href,'showHistory')]")
            )
        )
        driver.execute_script("arguments[0].click();", tab)
        time.sleep(POST_JS_SLEEP)
        WebDriverWait(driver, SHORT_WAIT).until(
            EC.visibility_of_element_located((By.ID, "fromDate"))
        )
        print("  [OK] Historical Data tab clicked → search form visible.")
        return True
    except (TimeoutException, NoSuchElementException):
        pass

    # --- Strategy 2: Search accordion (expand_collapse_search) ---
    try:
        toggle = WebDriverWait(driver, SHORT_WAIT).until(
            EC.element_to_be_clickable(
                (By.XPATH,
                 "//a[contains(@href, \"expand_collapse_search\")]")
            )
        )
        driver.execute_script("arguments[0].click();", toggle)
        time.sleep(POST_JS_SLEEP)
        WebDriverWait(driver, SHORT_WAIT).until(
            EC.visibility_of_element_located((By.ID, "fromDate"))
        )
        print("  [OK] Search accordion expanded → search form visible.")
        return True
    except (TimeoutException, NoSuchElementException):
        pass

    # --- Strategy 3: Force-show via JS (last resort) ---
    try:
        driver.execute_script(
            "document.getElementById('fromDate').closest"
            "('.search_main, #2').style.display='block';"
        )
        time.sleep(1)
        print("  [WARN] Forced search section visible via JS.")
        return True
    except Exception:
        pass

    print("  [ERROR] Could not make search form visible.")
    return False


def fill_dates(driver: webdriver.Chrome):
    _js_set(driver, "fromDate", FROM_DATE)
    _js_set(driver, "toDate",   TO_DATE)
    print(f"  [OK] Dates set: {FROM_DATE} → {TO_DATE}")


def select_subsection(driver: webdriver.Chrome, ssid: str):
    try:
        sel = Select(WebDriverWait(driver, SHORT_WAIT).until(
            EC.presence_of_element_located((By.ID, "ssid"))
        ))
        sel.select_by_value(ssid)
        time.sleep(1)
        print(f"  [OK] Sub-section dropdown set to ssid={ssid}.")
    except (TimeoutException, NoSuchElementException):
        print("  [WARN] ssid dropdown not found — skipping.")


def click_go(driver: webdriver.Chrome) -> bool:
    """Click GO and wait for results. Returns True on success."""
    try:
        go_btn = WebDriverWait(driver, LONG_WAIT).until(
            EC.element_to_be_clickable(
                (By.CSS_SELECTOR, "a.go-search.go_search")
            )
        )
        driver.execute_script("arguments[0].click();", go_btn)
        print("  [OK] GO button clicked — waiting for results…")
        time.sleep(POST_JS_SLEEP)
        return True
    except TimeoutException:
        print("  [WARN] GO button not found.")
        return False


def wait_for_table(driver: webdriver.Chrome, timeout: int = LONG_WAIT) -> bool:
    try:
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "table#sample_1 tbody tr")
            )
        )
        return True
    except TimeoutException:
        return False


def get_pagination_info(driver: webdriver.Chrome):
    """Return (total_pages, total_records)."""
    try:
        text = driver.find_element(
            By.CSS_SELECTOR, ".pagination_inner p"
        ).text.strip()
        # " 1 to 25 of 155 records"
        parts = text.split()
        total  = int(parts[-2])
        per_pg = int(parts[2])
        pages  = (total + per_pg - 1) // per_pg
        return pages, total
    except Exception:
        return 1, 0


def extract_links(driver: webdriver.Chrome, section: str) -> list[dict]:
    rows = driver.find_elements(By.CSS_SELECTOR, "table#sample_1 tbody tr")
    out  = []
    for row in rows:
        try:
            cells = row.find_elements(By.TAG_NAME, "td")
            if len(cells) < 2:
                continue
            date    = cells[0].text.strip()
            link_el = cells[1].find_element(By.TAG_NAME, "a")
            title   = link_el.text.strip()
            url     = link_el.get_attribute("href")
            if url:
                out.append({"section": section, "date": date,
                            "title": title, "url": url})
        except (NoSuchElementException, StaleElementReferenceException):
            continue
    return out


def go_to_page(driver: webdriver.Chrome, page_index: int):
    """Navigate to a numbered result page (page_index = page_num - 2)."""
    driver.execute_script(f"searchFormNewsList('n', '{page_index}');")
    time.sleep(POST_JS_SLEEP)
    wait_for_table(driver)


# ─── Per-section scraper ──────────────────────────────────────────────────────

def scrape_section(driver: webdriver.Chrome,
                   section_name: str,
                   ssid: str) -> list[dict]:

    print(f"\n{'='*60}")
    print(f"  Section : {section_name}  (ssid={ssid})")
    print(f"  Period  : {FROM_DATE}  →  {TO_DATE}")
    print(f"{'='*60}")

    # ── 1. Load page ──────────────────────────────────────────────────────────
    driver.get(BASE_URL.format(ssid=ssid))
    time.sleep(3)

    # ── 2. Make search form visible ───────────────────────────────────────────
    if not ensure_search_form_visible(driver):
        print(f"  [SKIP] Cannot access search form for {section_name}.")
        return []

    # ── 3. Fill dates & subsection ────────────────────────────────────────────
    fill_dates(driver)
    select_subsection(driver, ssid)

    # ── 4. Click GO ───────────────────────────────────────────────────────────
    if not click_go(driver):
        return []

    # ── 5. Wait for results ───────────────────────────────────────────────────
    if not wait_for_table(driver, timeout=LONG_WAIT):
        print(f"  [INFO] No results for {section_name} in date range.")
        return []

    total_pages, total_records = get_pagination_info(driver)
    print(f"  Found {total_records} records across {total_pages} page(s).")

    # ── 6. Collect page 1 ────────────────────────────────────────────────────
    all_recs = extract_links(driver, section_name)
    print(f"  Page 1/{total_pages} — {len(all_recs)} links")

    # ── 7. Subsequent pages ───────────────────────────────────────────────────
    for page_num in range(2, total_pages + 1):
        try:
            go_to_page(driver, page_num - 2)
            recs = extract_links(driver, section_name)
            all_recs.extend(recs)
            print(f"  Page {page_num}/{total_pages} — {len(recs)} links")
        except Exception as exc:
            print(f"  [WARN] Page {page_num} failed: {exc}")
            break

    print(f"  ✓ {section_name}: {len(all_recs)} total links")
    return all_recs


# ─── Output ───────────────────────────────────────────────────────────────────

def save_csv(records: list[dict], path: str):
    if not records:
        print("  No records to save.")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(
            f, fieldnames=["section", "date", "title", "url"]
        ).writeheader()
        csv.DictWriter(
            f, fieldnames=["section", "date", "title", "url"]
        ).writerows(records)
    print(f"\n✓ CSV  → {path}  ({len(records)} rows)")


def save_json(records: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"✓ JSON → {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"\nSEBI Legal Scraper  (v3)")
    print(f"Period   : {FROM_DATE}  to  {TO_DATE}")
    print(f"Sections : {', '.join(SECTIONS.keys())}\n")

    driver      = build_driver(headless=True)
    all_results = []

    try:
        for section_name, ssid in SECTIONS.items():
            try:
                records = scrape_section(driver, section_name, ssid)
                all_results.extend(records)
            except Exception as exc:
                print(f"[ERROR] {section_name}: {exc}")
    finally:
        driver.quit()

    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path  = f"sebi_legal_links_{ts}.csv"
    json_path = f"sebi_legal_links_{ts}.json"

    # Fix CSV: write header + rows in one pass
    if all_results:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["section","date","title","url"])
            w.writeheader()
            w.writerows(all_results)
        print(f"\n✓ CSV  → {csv_path}  ({len(all_results)} rows)")

    save_json(all_results, json_path)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    counts = {}
    for r in all_results:
        counts[r["section"]] = counts.get(r["section"], 0) + 1
    for sec, cnt in counts.items():
        print(f"  {sec:<15} {cnt:>5} links")
    print(f"  {'TOTAL':<15} {len(all_results):>5} links")
    print("="*60)
    return all_results


if __name__ == "__main__":
    main()