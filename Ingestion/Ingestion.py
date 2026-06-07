import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# PATH SETUP
BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "Documents"
CHUNK_DIR = BASE_DIR / "chunks"

CHUNK_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_DIR.exists():
    raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")


# VALID PDF CHECK
def is_valid_pdf(file_path):
    try:
        with open(file_path, "rb") as f:
            return f.read(5) == b"%PDF-"
    except:
        return False


# PDF LOADER
def load_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        return text.strip()

    except Exception:
        return None


# CHUNKING
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)


# SAVE CHUNKS
def save_chunks(file_path, chunks):
    # unique filename to avoid overwrite
    safe_name = str(file_path.relative_to(INPUT_DIR))
    safe_name = safe_name.replace("\\", "_").replace("/", "_")
    safe_name = Path(safe_name).stem

    for i, chunk in enumerate(chunks):
        out_file = CHUNK_DIR / f"{safe_name}_chunk_{i}.txt"
        out_file.write_text(chunk, encoding="utf-8")


# PROCESS ONE PDF
def process_pdf(file_path):
    try:
        if not is_valid_pdf(file_path):
            return None

        text = load_pdf(file_path)

        if not text or len(text) < 50:
            return None

        chunks = chunk_text(text)
        save_chunks(file_path, chunks)

        return {
            "file": file_path.name,
            "chunks": len(chunks),
            "chars": len(text)
        }

    except Exception:
        return None


# MAIN
def run():
    pdf_files = list(INPUT_DIR.rglob("*.pdf"))

    print(f"\n📁 Total PDFs found: {len(pdf_files)}\n")

    workers = 6

    processed_pdfs = 0
    total_chunks = 0
    total_chars = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_pdf, pdf) for pdf in pdf_files]

        for future in as_completed(futures):
            result = future.result()

            if result is None:
                continue

            processed_pdfs += 1
            total_chunks += result["chunks"]
            total_chars += result["chars"]

            print(
                f"✅ {result['file']} -> "
                f"{result['chunks']} chunks"
            )

    print("\n" + "=" * 50)
    print("FINAL SUMMARY")
    print("=" * 50)
    print(f"PDFs Processed : {processed_pdfs}")
    print(f"Total Chunks   : {total_chunks}")
    print(f"Approx Tokens  : {total_chars // 4:,}")

    print("\n🚀 Chunking complete (NO embeddings yet)")


if __name__ == "__main__":
    run()