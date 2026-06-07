from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

BASE_DIR   = Path.cwd()
print(f"BASE_DIR: {BASE_DIR}")
INPUT_DIR  = BASE_DIR / "ingestion" / "Documents"
OUTPUT_DIR = BASE_DIR / "ingestion" / "chunks_jsonl"
LOG_DIR    = BASE_DIR / "ingestion" / "logs"

print(f"BASE_DIR: {BASE_DIR}"
      f"\nINPUT_DIR: {INPUT_DIR}"
      f"\nOUTPUT_DIR: {OUTPUT_DIR}"
      f"\nLOG_DIR: {LOG_DIR}")

CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150
SEPARATORS    = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

# ── Runtime ───────────────────────────────────────────────────────────────────
MAX_WORKERS    = min(8, (os.cpu_count() or 4))
MAX_RETRIES    = 2          # Extraction retries per PDF
MIN_PAGE_CHARS = 50         # Skip pages shorter than this (headers, blank pages)
JSONL_BUFFER   = 500        # Flush JSONL to disk every N records (memory control)

def setup_logging() -> logging.Logger:
    """
    Two-handler setup:
      • File (DEBUG) — full detail for post-mortem analysis
      • Console (INFO) — clean progress for human operators

    One timestamped log file per pipeline run prevents log pollution
    and makes it easy to diff runs in CI/CD or scheduled jobs.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = LOG_DIR / f"ingestion_{run_ts}.log"

    fmt = logging.Formatter(
        fmt     = "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger("pdf_ingestion")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

@dataclass
class ChunkRecord:
    """
    One chunk — maps 1-to-1 to a Pinecone vector upsert.

    Pinecone upsert shape:
        index.upsert(vectors=[{
            "id":       record.id,
            "values":   embed(record.text),   # your embedding step
            "metadata": record.metadata,
        }])
    """
    id:       str    # Deterministic SHA-256
    text:     str    # Content sent to embedding model
    metadata: dict   # Returned in Pinecone query results


@dataclass
class ProcessingResult:
    """Structured result from each worker process. Never raises."""
    file_name:         str
    source_path:       str
    success:           bool
    chunks_written:    int   = 0
    pages_processed:   int   = 0
    pages_skipped:     int   = 0
    total_chars:       int   = 0
    processing_time_s: float = 0.0
    error: Optional[str]     = None

def is_valid_pdf(file_path: Path) -> bool:
    """
    Magic-byte check before attempting full PdfReader parse.

    Fast-fails on:
      • Renamed non-PDF files (e.g. HTML saved as .pdf)
      • Truncated / zero-byte files
      • Files with wrong extension

    Avoids paying the PdfReader parse cost on invalid inputs.
    """
    try:
        with open(file_path, "rb") as f:
            return f.read(5) == b"%PDF-"
    except OSError:
        return False

def generate_chunk_id(
    source: str,
    page: int,
    chunk_index: int,
    text: str,
) -> str:
    """
    Deterministic, collision-resistant chunk ID.

    Composite key: source + page + chunk_index + text
      • Same PDF re-ingested → identical IDs → Pinecone upsert is idempotent
      • Different content    → different ID  → no false deduplication
      • No dependency on run order or timestamps

    Why SHA-256 over UUID4:
      • UUID4 is random — re-ingestion creates duplicate vectors
      • SHA-256 of content is reproducible — safe to re-run pipeline on
        the same corpus without duplicating your index
    """
    raw = f"{source}::{page}::{chunk_index}::{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def extract_pages(file_path: Path) -> tuple[list[tuple[int, str]], int]:
    """
    Extract text page-by-page. Never concatenates the whole PDF.

    Returns:
        pages       — list of (1-indexed page_number, page_text) for non-empty pages
        total_pages — total page count including blank/unextractable pages

    Why page-by-page matters:
      • Chunk-to-page mapping becomes exact — critical for citations in RAG answers
      • Page-level metadata enables range-filtered retrieval
        e.g. "search only in pages 1–10" (abstract/intro) vs "search appendices"
      • Avoids loading the entire PDF text into a single string in RAM
      • Failed pages are isolated — one unreadable page doesn't corrupt the rest
    """
    reader      = PdfReader(str(file_path))
    total_pages = len(reader.pages)
    pages: list[tuple[int, str]] = []

    for page_num, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
            text = raw.strip()
            if text:
                pages.append((page_num, text))
        except Exception:
            # Individual page extraction failure → skip page, continue document
            pass

    return pages, total_pages

def build_splitter() -> RecursiveCharacterTextSplitter:
    """
    RecursiveCharacterTextSplitter configured for RAG retrieval quality.

    Built once per worker process and reused across all pages of a PDF.
    Instantiating a splitter is cheap, but no reason to reconstruct it
    per-page in a tight loop.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size        = CHUNK_SIZE,
        chunk_overlap     = CHUNK_OVERLAP,
        separators        = SEPARATORS,
        length_function   = len,
        is_separator_regex= False,
    )

def chunk_page(
    page_text:           str,
    splitter:            RecursiveCharacterTextSplitter,
    source:              str,
    source_path:         str,
    page_number:         int,
    total_pages:         int,
    total_chars:         int,
    ingestion_timestamp: str,
) -> list[ChunkRecord]:
    """
    Chunk a single page and attach full retrieval metadata to each record.

    Metadata design notes
    ─────────────────────
    page_position (0.0–1.0):
        Normalized location of the page in the document.
        Enables "retrieve from early/late in document" filters without
        knowing the exact page count in the query layer.

    is_first_page / is_last_page:
        Pinecone metadata filters can pin retrieval to intro or conclusion
        sections, useful for summarization tasks.

    chunk_char_length:
        Lets you filter out very short chunks in the query layer
        (e.g. discard chunks < 100 chars that are likely headers).

    chunk_size_config / chunk_overlap_config:
        Provenance fields. If you re-chunk with different settings,
        you can tell apart old vs new vectors in a hybrid index.
    """
    raw_chunks = splitter.split_text(page_text)
    records: list[ChunkRecord] = []

    for chunk_index, chunk_text in enumerate(raw_chunks):
        chunk_text = chunk_text.strip()
        if not chunk_text:
            continue

        chunk_id = generate_chunk_id(source, page_number, chunk_index, chunk_text)

        metadata = {
            # ── Source tracing ─────────────────────────────────────────────
            "source":              source,          # PDF stem name
            "source_path":         source_path,     # relative path from INPUT_DIR

            # ── Page location ──────────────────────────────────────────────
            "page":                page_number,
            "total_pages":         total_pages,
            "page_position":       round(page_number / total_pages, 4),

            # ── Chunk location ─────────────────────────────────────────────
            "chunk_id":            chunk_id,        # also the Pinecone vector ID
            "chunk_index":         chunk_index,     # index within this page
            "chunk_char_length":   len(chunk_text),

            # ── Document stats ─────────────────────────────────────────────
            "document_size_chars": total_chars,

            # ── Retrieval hints ────────────────────────────────────────────
            "is_first_page":       page_number == 1,
            "is_last_page":        page_number == total_pages,

            # ── Pipeline provenance ────────────────────────────────────────
            "ingestion_timestamp":  ingestion_timestamp,
            "chunk_size_config":    CHUNK_SIZE,
            "chunk_overlap_config": CHUNK_OVERLAP,
        }

        records.append(ChunkRecord(id=chunk_id, text=chunk_text, metadata=metadata))

    return records

def save_jsonl(records: list[ChunkRecord], output_path: Path) -> None:
    """
    Write chunk records as JSONL — one JSON object per line.

    Why JSONL instead of 31,000 .txt files
    ───────────────────────────────────────
    ① Structured: metadata and text are co-located — no separate lookup table
    ② Streamable: can be read line-by-line without loading all records into RAM,
       which is critical when loading 1M+ chunks for batch embedding
    ③ Pinecone-ready: each line maps 1-to-1 to an index.upsert() call
    ④ Debuggable: human-readable, grep-able, diff-able in any text editor
    ⑤ Partial re-ingestion: one file per PDF means you can re-process a single
       document without touching the rest of the output directory
    ⑥ Filesystem-friendly: 151 .jsonl files vs 31,446 .txt files — inodes, 
       directory listing, and backup tools all perform dramatically better
    ⑦ Schema evolution: adding a new metadata field doesn't change the file
       structure — old files stay valid, new files gain the new field
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            line = json.dumps(
                {
                    "id":       record.id,
                    "text":     record.text,
                    "metadata": record.metadata,
                },
                ensure_ascii=False,   # preserve non-ASCII chars (citations, names)
            )
            f.write(line + "\n")

def process_pdf(args: tuple[Path, Path, Path]) -> ProcessingResult:
    """
    End-to-end processing of a single PDF.

    Runs inside a worker process from ProcessPoolExecutor.

    Fault-tolerance contract:
      • NEVER raises an exception to the caller
      • Every failure path returns ProcessingResult(success=False, error=...)
      • A single PDF failing does not affect any other PDF
      • Per-page failures are tolerated — the document is still written
        with the successfully extracted pages

    Retry logic:
      • PDF extraction is retried up to MAX_RETRIES times with exponential
        back-off — handles transient I/O errors on network-mounted volumes
      • Single-page failures are not retried — they are silently skipped
        and counted in pages_skipped
    """
    file_path, input_dir, output_dir = args
    start_time = time.perf_counter()

    file_name   = file_path.name
    source_path = str(file_path.relative_to(input_dir))
    source      = file_path.stem
    ingestion_ts = datetime.now(timezone.utc).isoformat()

    # ── 1. Magic-byte validation ───────────────────────────────────────────────
    if not is_valid_pdf(file_path):
        return ProcessingResult(
            file_name   = file_name,
            source_path = source_path,
            success     = False,
            error       = "Invalid PDF header — file is not a PDF",
        )

    # ── 2. Extract pages (with retry) ─────────────────────────────────────────
    pages: list[tuple[int, str]] = []
    total_pages = 0
    last_error: Optional[str] = None

    for attempt in range(1, MAX_RETRIES + 2):          # 1, 2, 3 attempts total
        try:
            pages, total_pages = extract_pages(file_path)
            last_error = None
            break
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt <= MAX_RETRIES:
                time.sleep(0.3 * attempt)              # 0.3s, 0.6s back-off

    if last_error:
        return ProcessingResult(
            file_name   = file_name,
            source_path = source_path,
            success     = False,
            error       = f"Extraction failed after {MAX_RETRIES + 1} attempts: {last_error}",
        )

    if not pages:
        return ProcessingResult(
            file_name   = file_name,
            source_path = source_path,
            success     = False,
            error       = f"No extractable text found (scanned/image-only PDF? total_pages={total_pages})",
        )

    # ── 3. Document-level stats ────────────────────────────────────────────────
    total_chars   = sum(len(text) for _, text in pages)
    pages_skipped = total_pages - len(pages)           # blank/unreadable pages

    # ── 4. Chunk page-by-page ─────────────────────────────────────────────────
    splitter = build_splitter()
    all_records: list[ChunkRecord] = []

    for page_number, page_text in pages:
        if len(page_text) < MIN_PAGE_CHARS:
            pages_skipped += 1
            continue

        try:
            records = chunk_page(
                page_text           = page_text,
                splitter            = splitter,
                source              = source,
                source_path         = source_path,
                page_number         = page_number,
                total_pages         = total_pages,
                total_chars         = total_chars,
                ingestion_timestamp = ingestion_ts,
            )
            all_records.extend(records)
        except Exception as exc:
            # Isolated page failure — skip page, continue document
            pages_skipped += 1

    if not all_records:
        return ProcessingResult(
            file_name   = file_name,
            source_path = source_path,
            success     = False,
            error       = "Chunking produced zero records after processing all pages",
        )

    # ── 5. Write JSONL ────────────────────────────────────────────────────────
    # Build a filesystem-safe output filename from the relative path
    safe_stem   = source_path.replace("\\", "__").replace("/", "__")
    safe_stem   = Path(safe_stem).stem
    output_path = output_dir / f"{safe_stem}.jsonl"

    try:
        save_jsonl(all_records, output_path)
    except OSError as exc:
        return ProcessingResult(
            file_name   = file_name,
            source_path = source_path,
            success     = False,
            error       = f"JSONL write failed: {exc}",
        )

    elapsed = time.perf_counter() - start_time

    return ProcessingResult(
        file_name         = file_name,
        source_path       = source_path,
        success           = True,
        chunks_written    = len(all_records),
        pages_processed   = len(pages) - max(0, pages_skipped - (total_pages - len(pages))),
        pages_skipped     = pages_skipped,
        total_chars       = total_chars,
        processing_time_s = round(elapsed, 3),
    )

def run() -> None:
    """
    Main pipeline entry point.

    Workers: ProcessPoolExecutor
      For CPU-bound PDF parsing + chunking, multiprocessing beats threading
      because it bypasses the GIL. Each worker gets its own Python interpreter
      and memory space, which also provides fault isolation — a worker crash
      doesn't take down the main process.

    Scalability ceiling of this approach:
      ~10,000 PDFs — ProcessPoolExecutor handles this well
      ~100,000 PDFs — consider Celery + Redis or Ray for distributed work queues
      ~1M+ chunks — JSONL streaming + async Pinecone upsert batches

    See SCALABILITY NOTE at the bottom of this file.
    """
    logger = setup_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_DIR.exists():
        logger.critical(f"Input directory not found: {INPUT_DIR}")
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")

    # ── Discover PDFs ──────────────────────────────────────────────────────────
    pdf_files  = sorted(INPUT_DIR.rglob("*.pdf"))
    total_pdfs = len(pdf_files)

    if total_pdfs == 0:
        logger.warning("No PDF files found in input directory. Nothing to do.")
        return

    logger.info("=" * 65)
    logger.info("PDF INGESTION PIPELINE — START")
    logger.info("=" * 65)
    logger.info(f"Input  directory : {INPUT_DIR}")
    logger.info(f"Output directory : {OUTPUT_DIR}")
    logger.info(f"PDFs discovered  : {total_pdfs}")
    logger.info(f"Workers          : {MAX_WORKERS}")
    logger.info(f"Chunk size       : {CHUNK_SIZE} chars (~{CHUNK_SIZE // 4} tokens)")
    logger.info(f"Chunk overlap    : {CHUNK_OVERLAP} chars")
    logger.info("=" * 65)

    # ── Dispatch workers ───────────────────────────────────────────────────────
    args_list = [(pdf, INPUT_DIR, OUTPUT_DIR) for pdf in pdf_files]

    results: list[ProcessingResult] = []
    pipeline_start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_pdf, args): args[0]
            for args in args_list
        }

        completed = 0
        for future in as_completed(futures):
            completed += 1

            try:
                result = future.result()
            except Exception as exc:
                # Defensive guard — worker is designed never to raise
                pdf_path = futures[future]
                result = ProcessingResult(
                    file_name   = pdf_path.name,
                    source_path = str(pdf_path.relative_to(INPUT_DIR)),
                    success     = False,
                    error       = f"Unhandled worker exception: {type(exc).__name__}: {exc}",
                )

            results.append(result)
            pct = f"{completed / total_pdfs * 100:.1f}%"

            if result.success:
                logger.info(
                    f"[{completed:>4}/{total_pdfs}] ({pct}) ✅  {result.file_name} | "
                    f"pages={result.pages_processed}  "
                    f"skipped={result.pages_skipped}  "
                    f"chunks={result.chunks_written}  "
                    f"time={result.processing_time_s}s"
                )
            else:
                logger.warning(
                    f"[{completed:>4}/{total_pdfs}] ({pct}) ❌  {result.file_name} | "
                    f"error={result.error}"
                )

    # ── Aggregate metrics ──────────────────────────────────────────────────────
    pipeline_elapsed = time.perf_counter() - pipeline_start

    successes = [r for r in results if r.success]
    failures  = [r for r in results if not r.success]

    total_chunks       = sum(r.chunks_written  for r in successes)
    total_chars        = sum(r.total_chars     for r in successes)
    total_pages_ok     = sum(r.pages_processed for r in successes)
    total_pages_skip   = sum(r.pages_skipped   for r in successes)
    avg_time           = (
        sum(r.processing_time_s for r in successes) / len(successes)
        if successes else 0.0
    )
    chunks_per_second  = (
        round(total_chunks / pipeline_elapsed, 1)
        if pipeline_elapsed > 0 else 0.0
    )

    summary = {
        "pipeline_duration_s":   round(pipeline_elapsed, 2),
        "pdfs_found":            total_pdfs,
        "pdfs_succeeded":        len(successes),
        "pdfs_failed":           len(failures),
        "total_pages_processed": total_pages_ok,
        "total_pages_skipped":   total_pages_skip,
        "total_chunks":          total_chunks,
        "approx_tokens":         total_chars // 4,
        "avg_pdf_time_s":        round(avg_time, 3),
        "chunks_per_second":     chunks_per_second,
        "workers_used":          MAX_WORKERS,
        "chunk_size_config":     CHUNK_SIZE,
        "chunk_overlap_config":  CHUNK_OVERLAP,
    }

    # ── Write ingestion_summary.json ───────────────────────────────────────────
    summary_path = OUTPUT_DIR / "ingestion_summary.json"
    failed_list  = [
        {"file": r.file_name, "path": r.source_path, "error": r.error}
        for r in failures
    ]
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {"summary": summary, "failures": failed_list},
            f, indent=2, ensure_ascii=False,
        )

    # ── Final log ──────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 65)
    logger.info("INGESTION COMPLETE")
    logger.info("=" * 65)
    for k, v in summary.items():
        val = f"{v:,}" if isinstance(v, int) else str(v)
        logger.info(f"  {k:<30} {val}")

    if failures:
        logger.warning(f"\nFailed PDFs ({len(failures)}) — see {summary_path}")
        for r in failures:
            logger.warning(f"  ✗  {r.file_name}: {r.error}")

    logger.info(f"\nSummary written to : {summary_path}")
    logger.info("Ingestion complete. Ready for embedding + Pinecone upsert.")
    logger.info("=" * 65)

def stream_chunks_for_upsert(output_dir: Path = OUTPUT_DIR) -> Iterator[dict]:
    """
    Lazy generator — streams JSONL records one at a time.

    Memory footprint: O(1) regardless of corpus size.
    Suitable for 1M+ chunks — never loads all records into RAM.

    Usage example (OpenAI + Pinecone):

        from openai import OpenAI
        import pinecone

        client  = OpenAI()
        index   = pinecone.Index("your-index-name")

        BATCH   = 100
        batch   = []

        for record in stream_chunks_for_upsert():
            embedding = client.embeddings.create(
                input = record["text"],
                model = "text-embedding-3-small",
            ).data[0].embedding

            batch.append({
                "id":       record["id"],
                "values":   embedding,
                "metadata": record["metadata"],
            })

            if len(batch) >= BATCH:
                index.upsert(vectors=batch)
                batch.clear()

        if batch:
            index.upsert(vectors=batch)

    The metadata dict passes through unchanged — no transformation required.
    Pinecone metadata filters map directly to the keys already present:
        index.query(
            vector      = query_embedding,
            filter      = {"source": "chapter_3", "page": {"$gte": 5}},
            top_k       = 10,
            include_metadata = True,
        )
    """
    for jsonl_file in sorted(output_dir.glob("*.jsonl")):
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


# ══════════════════════════════════════════════════════════════════════════════
# SCALABILITY NOTE
# ══════════════════════════════════════════════════════════════════════════════
#
# Current architecture (ProcessPoolExecutor) — appropriate for:
#   ≤ 10,000 PDFs   on a single multi-core machine
#
# 10,000 → 100,000 PDFs
#   Bottleneck: single-machine RAM and CPU
#   Solution  : Celery + Redis task queue. Each worker node runs this same
#               process_pdf() function. The main process only dispatches.
#               Scale horizontally by adding worker nodes.
#
# 100,000 PDFs / 1M+ chunks
#   Bottleneck: disk I/O on a single output directory
#   Solution  : Partition output by document hash prefix (e.g. chunks_jsonl/ab/).
#               Use object storage (S3/GCS) instead of local disk.
#               Use Ray for distributed in-process parallelism.
#
# Embedding bottleneck (always)
#   OpenAI rate limit: ~3,000 RPM on text-embedding-3-small
#   For 1M chunks: use batch embedding API (up to 50k tokens per request)
#   or run a local model (e2-small, nomic-embed-text) for uncapped throughput.
#
# Pinecone upsert bottleneck
#   Pinecone recommends batches of 100 vectors per upsert call.
#   For 1M+ vectors, use async upserts with asyncio + aiohttp or the
#   official Pinecone async client.
#
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run()


