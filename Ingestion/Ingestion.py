import os
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# PATH SETUP

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "Documents"
CHUNK_DIR = BASE_DIR / "chunks"

CHUNK_DIR.mkdir(parents=True, exist_ok=True)

# 🔴 CHECK FOLDER EXISTS
if not INPUT_DIR.exists():
    raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")



#  VALID PDF CHECK

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

    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
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
    safe_name = file_path.stem

    for i, chunk in enumerate(chunks):
        out_file = CHUNK_DIR / f"{safe_name}_chunk_{i}.txt"
        out_file.write_text(chunk, encoding="utf-8")


# MAIN PIPELINE

def run():
    # 🔥 recursive scan (RBI + SCBI folders)
    pdf_files = list(INPUT_DIR.rglob("*.pdf"))

    print(f"📁 Total PDFs found: {len(pdf_files)}\n")

    for file_path in pdf_files:
        print(f"📄 Processing: {file_path.relative_to(INPUT_DIR)}")

        # skip invalid PDFs
        if not is_valid_pdf(file_path):
            print("❌ Skipped (not a real PDF)")
            continue

        text = load_pdf(file_path)
        if not text or len(text) < 50:
            print("⚠️ Skipped (empty/invalid content)")
            continue

        chunks = chunk_text(text)
        save_chunks(file_path, chunks)

        print(f"✅ Chunks created: {len(chunks)}\n")

    print("\n🚀 Chunking complete (NO embeddings yet)")


if __name__ == "__main__":
    run()