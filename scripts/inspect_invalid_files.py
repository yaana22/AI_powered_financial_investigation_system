# from pathlib import Path

# pdf_dir = Path("ingestion/Documents")

# print("Documents exists:", pdf_dir.exists())

# pdfs = list(pdf_dir.rglob("*.pdf"))

# print("Total PDFs found:", len(pdfs))



from pathlib import Path

pdf_dir = Path("ingestion/Documents")

shown = 0

for pdf_file in pdf_dir.rglob("*.pdf"):

    with open(pdf_file, "rb") as f:
        header = f.read(10)

    if not header.startswith(b"%PDF"):

        print("\n" + "=" * 80)
        print(f"FILE: {pdf_file}")
        print(f"HEADER: {header}")

        with open(pdf_file, "rb") as f:
            content = f.read(500)

        print(content.decode("utf-8", errors="ignore"))

        shown += 1

        if shown >= 5:
            break