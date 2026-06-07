
import json
import logging
import os
import random
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIStatusError
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

load_dotenv()

BASE_DIR            = Path(__file__).resolve().parent
JSONL_DIR           = BASE_DIR / "chunks_jsonl"
LOG_DIR             = BASE_DIR / "logs"

OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "sk-...")
EMBEDDING_MODEL     = "text-embedding-3-small"
EMBEDDING_DIM       = 1536

PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY", "pcsk_...")
PINECONE_INDEX_NAME = "financial-crime-rag"
PINECONE_METRIC     = "cosine"

EMBED_BATCH_SIZE    = 100
UPSERT_BATCH_SIZE   = 100
NUM_EMBED_WORKERS   = 8
MIN_TEXT_LENGTH     = 10

MAX_RETRIES         = 6
BASE_BACKOFF        = 1.0
MAX_BACKOFF         = 60.0

TPM_LIMIT           = 900_000
TOKENS_PER_CHUNK    = 200

_tpm_lock           = threading.Lock()
_tokens_this_minute = 0
_minute_start       = time.monotonic()


def _throttle(n_chunks: int) -> None:
    """
    Token-per-minute gate.

    Pattern: check under lock → release lock → sleep outside lock → retry.
    Never holds the lock while sleeping so all 8 workers remain unblocked
    during the wait window.
    """
    global _tokens_this_minute, _minute_start
    tokens_needed = n_chunks * TOKENS_PER_CHUNK

    while True:
        with _tpm_lock:
            now     = time.monotonic()
            elapsed = now - _minute_start

            if elapsed >= 60.0:
                _tokens_this_minute = 0
                _minute_start       = now
                elapsed             = 0.0

            if _tokens_this_minute + tokens_needed <= TPM_LIMIT:
                _tokens_this_minute += tokens_needed
                return                          # lock released here, no sleep

            sleep_for = 60.0 - elapsed + 0.5   # how long until window resets
        # ── lock is released before sleeping ──────────────────────────────────
        time.sleep(sleep_for)
        # loop back and re-check under lock (another thread may have reset the window)


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = LOG_DIR / f"embedding_{run_ts}.log"
    fmt     = logging.Formatter(
        fmt     = "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("embed_upsert")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


def stream_records(jsonl_dir: Path) -> Iterator[dict]:
    for jsonl_file in sorted(jsonl_dir.glob("*.jsonl")):
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def count_total_chunks(jsonl_dir: Path) -> int:
    total = 0
    for jsonl_file in jsonl_dir.glob("*.jsonl"):
        with open(jsonl_file, encoding="utf-8") as f:
            total += sum(1 for line in f if line.strip())
    return total


def batched(iterable: Iterator, size: int) -> Iterator[list]:
    it = iter(iterable)
    while True:
        batch = list(islice(it, size))
        if not batch:
            break
        yield batch


def _sanitize_batch(records: list[dict]) -> tuple[list[dict], list[str], int]:
    valid_records: list[dict] = []
    valid_texts:   list[str]  = []
    skipped = 0
    for r in records:
        raw = r.get("text", "")
        if not isinstance(raw, str):
            raw = str(raw) if raw is not None else ""
        cleaned = raw.strip()
        if len(cleaned) >= MIN_TEXT_LENGTH:
            valid_records.append(r)
            valid_texts.append(cleaned)
        else:
            skipped += 1
    return valid_records, valid_texts, skipped


def embed_batch_with_retry(
    client: OpenAI,
    records: list[dict],
    batch_index: int,
) -> tuple[int, list[dict] | None, list[list[float]] | None, int, str | None]:
    valid_records, texts, skipped = _sanitize_batch(records)

    if not texts:
        return batch_index, [], [], len(records), None

    _throttle(len(texts))
    backoff = BASE_BACKOFF

    for attempt in range(1, MAX_RETRIES + 2):
        try:
            response = client.embeddings.create(
                input = texts,
                model = EMBEDDING_MODEL,
            )
            vectors = [item.embedding for item in response.data]
            return batch_index, valid_records, vectors, skipped, None
        except RateLimitError as exc:
            retry_after = None
            if hasattr(exc, "response") and exc.response is not None:
                retry_after = exc.response.headers.get("retry-after")
            wait = float(retry_after) if retry_after else min(backoff + random.uniform(0, 1), MAX_BACKOFF)
            if attempt <= MAX_RETRIES:
                time.sleep(wait)
                backoff = min(backoff * 2, MAX_BACKOFF)
            else:
                return batch_index, None, None, skipped, f"RateLimitError after {MAX_RETRIES + 1} attempts: {exc}"
        except APIStatusError as exc:
            if exc.status_code in (500, 502, 503, 529) and attempt <= MAX_RETRIES:
                wait = min(backoff + random.uniform(0, 1), MAX_BACKOFF)
                time.sleep(wait)
                backoff = min(backoff * 2, MAX_BACKOFF)
            else:
                return batch_index, None, None, skipped, f"APIStatusError {exc.status_code}: {exc}"
        except Exception as exc:
            return batch_index, None, None, skipped, f"Unexpected error: {exc}"

    return batch_index, None, None, skipped, "Max retries exhausted"


def upsert_batch(index, records: list[dict], vectors: list[list[float]]) -> int:
    pinecone_vectors = [
        {"id": r["id"], "values": v, "metadata": r["metadata"]}
        for r, v in zip(records, vectors)
    ]
    index.upsert(vectors=pinecone_vectors)
    return len(pinecone_vectors)


def get_or_create_index(pc: Pinecone, logger: logging.Logger):
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        logger.info(f"Creating index '{PINECONE_INDEX_NAME}' ({EMBEDDING_DIM} dims, {PINECONE_METRIC})")
        pc.create_index(
            name      = PINECONE_INDEX_NAME,
            dimension = EMBEDDING_DIM,
            metric    = PINECONE_METRIC,
            spec      = ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(2)
        logger.info("Index created and ready.")
    else:
        logger.info(f"Using existing index '{PINECONE_INDEX_NAME}'")
    return pc.Index(PINECONE_INDEX_NAME)


def run() -> None:
    logger = setup_logging()

    if not JSONL_DIR.exists():
        raise FileNotFoundError(f"Run ingest_pipeline.py first. Not found: {JSONL_DIR}")

    jsonl_files = list(JSONL_DIR.glob("*.jsonl"))
    if not jsonl_files:
        logger.warning("No JSONL files found.")
        return

    logger.info("=" * 65)
    logger.info("EMBEDDING + PINECONE UPSERT PIPELINE — START")
    logger.info("=" * 65)
    logger.info(f"JSONL directory  : {JSONL_DIR}")
    logger.info(f"JSONL files      : {len(jsonl_files)}")
    logger.info(f"Embedding model  : {EMBEDDING_MODEL} ({EMBEDDING_DIM} dims)")
    logger.info(f"Workers          : {NUM_EMBED_WORKERS}")
    logger.info(f"Embed batch size : {EMBED_BATCH_SIZE}")
    logger.info(f"Min text length  : {MIN_TEXT_LENGTH} chars")
    logger.info(f"TPM cap          : {TPM_LIMIT:,}")
    logger.info(f"Pinecone index   : {PINECONE_INDEX_NAME}")
    logger.info("=" * 65)

    logger.info("Counting chunks...")
    total_chunks = count_total_chunks(JSONL_DIR)
    logger.info(f"Total chunks: {total_chunks:,}")

    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    pc            = Pinecone(api_key=PINECONE_API_KEY)
    index         = get_or_create_index(pc, logger)

    start_time     = time.perf_counter()
    total_upserted = 0
    total_errors   = 0
    total_skipped  = 0

    all_batches   = list(enumerate(batched(stream_records(JSONL_DIR), EMBED_BATCH_SIZE)))
    total_batches = len(all_batches)

    with tqdm(total=total_chunks, unit="chunk", desc="Embedding & upserting") as pbar:
        with ThreadPoolExecutor(max_workers=NUM_EMBED_WORKERS) as executor:

            future_to_batch = {
                executor.submit(embed_batch_with_retry, openai_client, batch, idx): (idx, batch)
                for idx, batch in all_batches
            }

            for future in as_completed(future_to_batch):
                orig_idx, orig_batch = future_to_batch[future]
                batch_num, valid_records, vectors, skipped, error = future.result()

                total_skipped += skipped

                if error:
                    logger.error(f"Batch {batch_num:>4} failed: {error}")
                    total_errors += len(orig_batch) - skipped
                    pbar.update(len(orig_batch))
                    continue

                if not valid_records:
                    pbar.update(len(orig_batch))
                    continue

                for i in range(0, len(valid_records), UPSERT_BATCH_SIZE):
                    sub_records = valid_records[i : i + UPSERT_BATCH_SIZE]
                    sub_vectors = vectors[i : i + UPSERT_BATCH_SIZE]
                    try:
                        total_upserted += upsert_batch(index, sub_records, sub_vectors)
                    except Exception as exc:
                        logger.error(f"Upsert failed for batch {batch_num}: {exc}")
                        total_errors += len(sub_records)

                pbar.update(len(orig_batch))
                pbar.set_postfix(upserted=f"{total_upserted:,}", skipped=total_skipped, errors=total_errors)
                logger.debug(
                    f"Batch {batch_num:>4}/{total_batches} | "
                    f"valid={len(valid_records)} skipped={skipped} | "
                    f"upserted_total={total_upserted:,}"
                )

    elapsed     = time.perf_counter() - start_time
    index_stats = index.describe_index_stats()

    logger.info("")
    logger.info("=" * 65)
    logger.info("EMBEDDING + UPSERT COMPLETE")
    logger.info("=" * 65)
    logger.info(f"  Total chunks     : {total_chunks:,}")
    logger.info(f"  Upserted         : {total_upserted:,}")
    logger.info(f"  Skipped (bad txt): {total_skipped:,}")
    logger.info(f"  Errors           : {total_errors:,}")
    logger.info(f"  Duration         : {elapsed:.1f}s")
    logger.info(f"  Throughput       : {total_upserted / elapsed:.0f} chunks/s")
    logger.info(f"  Pinecone vectors : {index_stats.total_vector_count:,}")
    logger.info("=" * 65)


if __name__ == "__main__":
    run()
