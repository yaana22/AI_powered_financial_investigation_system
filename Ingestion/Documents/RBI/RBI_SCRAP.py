"""
RBI AML Scraper — Selenium version (drives a real Chrome browser)
=================================================================
Use this when the requests-based script gets 403 Forbidden.

Requirements:
    pip install selenium webdriver-manager

Run:
    python rbi_aml_scraper_selenium.py
"""

import csv
import json
import time
import sys
from datetime import datetime

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait, Select
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    from selenium.webdriver.chrome.options import Options
except ImportError:
    print("Selenium not installed. Run:  pip install selenium webdriver-manager")
    sys.exit(1)

try:
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.chrome.service import Service
    USE_WDM = True
except ImportError:
    USE_WDM = False

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_URL    = "https://rbi.org.in/scripts/SearchResults.aspx?search=beneficial+ownership"
SEARCH_TERM = "beneficial ownership"
FROM_DATE   = "01/01/2025"
TO_DATE     = datetime.today().strftime("%d/%m/%Y")
SECTIONS    = ["Notifications", "Reports", "Publications", "AnnualReport"]
OUTPUT_CSV  = "rbi_beneficial_links.csv"
HEADLESS    = False   # Set True to run without browser window
PAGE_WAIT   = 3       # seconds to wait after each action

# ── Browser setup ─────────────────────────────────────────────────────────────

def make_driver() -> webdriver.Chrome:
    opts = Options()
    if HEADLESS:
        opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1400,900")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    )

    if USE_WDM:
        service = Service(ChromeDriverManager().install())
        driver  = webdriver.Chrome(service=service, options=opts)
    else:
        driver = webdriver.Chrome(options=opts)

    driver.execute_script(
        "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"
    )
    return driver


# ── Helpers ───────────────────────────────────────────────────────────────────

def wait_for_results(driver, timeout=15):
    """Wait until at least one result div is present."""
    WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div#annual2 a.sub_title2_link"))
    )


def parse_page(driver) -> list:
    """Extract links from the current page."""
    results = []
    divs = driver.find_elements(By.CSS_SELECTOR, "div#annual2")
    for div in divs:
        try:
            a   = div.find_element(By.CSS_SELECTOR, "a.sub_title2_link")
            title = a.text.strip()
            href  = a.get_attribute("href") or ""
            try:
                date = div.find_element(
                    By.CSS_SELECTOR, "div.sub_title2_detail"
                ).text.strip()
            except NoSuchElementException:
                date = ""
            if title and href:
                results.append({"title": title, "url": href, "date": date})
        except NoSuchElementException:
            continue
    return results


def get_total_pages(driver) -> int:
    try:
        pagination = driver.find_element(By.CSS_SELECTOR, "div.pagination")
        lis = pagination.find_elements(By.TAG_NAME, "li")
        nums = []
        for li in lis:
            try:
                nums.append(int(li.text.strip()))
            except ValueError:
                pass
        return max(nums) if nums else 1
    except NoSuchElementException:
        return 1


def get_total_records(driver) -> int:
    try:
        return int(driver.find_element(By.ID, "lblTotalRec").text.strip())
    except Exception:
        return 0


def set_date_field(driver, field_id: str, value: str):
    """Clear and type a date into a datepicker field via JS (avoids keyboard issues)."""
    driver.execute_script(
        f"document.getElementById('{field_id}').value = '{value}';"
    )


def set_select(driver, select_id: str, value: str):
    sel = Select(driver.find_element(By.ID, select_id))
    sel.select_by_value(value)


def click_section_button(driver, section: str):
    """Click the filter button for a section (All / Notifications / …)."""
    btn_id = f"btn_{section}" if section else "btn_All"
    btn = driver.find_element(By.ID, btn_id)
    driver.execute_script("arguments[0].click();", btn)


def click_update(driver):
    btn = driver.find_element(By.ID, "btnUpdate")
    driver.execute_script("arguments[0].click();", btn)


def click_next_page(driver, page: int):
    """Click the numbered pagination link for `page`."""
    pagination = driver.find_element(By.CSS_SELECTOR, "div.pagination")
    links = pagination.find_elements(By.TAG_NAME, "a")
    for lnk in links:
        if lnk.text.strip() == str(page):
            driver.execute_script("arguments[0].click();", lnk)
            return
    # Fallback: use Next button
    try:
        nxt = pagination.find_element(By.CSS_SELECTOR, "div.next a")
        driver.execute_script("arguments[0].click();", nxt)
    except NoSuchElementException:
        pass


# ── Core scraper ──────────────────────────────────────────────────────────────

def scrape_section(driver, section: str) -> list:
    print(f"\n{'='*60}")
    print(f"  Section : {section}")
    print(f"{'='*60}")

    all_results = []

    # ── Set date range ────────────────────────────────────────────────────────
    set_select(driver, "ddlYearRange", "0")          # custom range
    set_date_field(driver, "txtFromDate", FROM_DATE)
    set_date_field(driver, "txtToDate",   TO_DATE)

    # ── Click section filter ──────────────────────────────────────────────────
    click_section_button(driver, section)
    time.sleep(PAGE_WAIT)

    try:
        wait_for_results(driver)
    except TimeoutException:
        print("  No results or timeout on page 1.")
        return all_results

    total = get_total_records(driver)
    pages = get_total_pages(driver)
    print(f"  Records : {total}  |  Pages : {pages}")

    page1 = parse_page(driver)
    for r in page1:
        r["section"] = section
    all_results.extend(page1)
    print(f"  Page 1  → {len(page1)} links")

    for page in range(2, pages + 1):
        click_next_page(driver, page)
        time.sleep(PAGE_WAIT)
        try:
            wait_for_results(driver)
        except TimeoutException:
            print(f"  Timeout on page {page}, skipping.")
            break
        page_res = parse_page(driver)
        for r in page_res:
            r["section"] = section
        all_results.extend(page_res)
        print(f"  Page {page}  → {len(page_res)} links")

    return all_results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  RBI AML Scraper (Selenium / Chrome)")
    print("=" * 60)
    print(f"  Search : {SEARCH_TERM}")
    print(f"  Range  : {FROM_DATE}  →  {TO_DATE}")
    print(f"  Tabs   : {', '.join(SECTIONS)}")

    driver = make_driver()
    all_links = []

    try:
        print(f"\n[Step 1] Opening RBI search page...")
        driver.get(BASE_URL)
        time.sleep(PAGE_WAIT)

        for section in SECTIONS:
            try:
                results = scrape_section(driver, section)
                all_links.extend(results)
            except Exception as exc:
                print(f"  [ERROR] Section '{section}': {exc}")

    finally:
        driver.quit()

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["section", "title", "date", "url"])
        writer.writeheader()
        writer.writerows(all_links)

    json_path = OUTPUT_CSV.replace(".csv", ".json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(
            {"meta": {"search": SEARCH_TERM, "from": FROM_DATE,
                      "to": TO_DATE, "scraped_at": datetime.now().isoformat()},
             "results": all_links},
            jf, indent=2, ensure_ascii=False
        )

    print(f"\n{'='*60}")
    print(f"  Saved {len(all_links)} links")
    print(f"  CSV  → {OUTPUT_CSV}")
    print(f"  JSON → {json_path}")
    print(f"{'='*60}")

    by_section = {}
    for r in all_links:
        by_section.setdefault(r["section"], []).append(r)
    print("\nSummary:")
    for sec, items in by_section.items():
        print(f"  {sec:20s} → {len(items)} links")

    print("\nSample links (first 2 per section):")
    for sec, items in by_section.items():
        print(f"\n  [{sec}]")
        for item in items[:2]:
            print(f"    • {item['date']:<14} {item['title'][:65]}")
            print(f"      {item['url']}")


if __name__ == "__main__":
    main()