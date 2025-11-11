import os
import json
import time
from src.ingest.extract_pdf import extract_any
from src.preprocess.clean_text import clean_text

RAW_DIR = "data/raw/"
OUT_FILE = "data/cleaned/corpus.jsonl"

def process_documents():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    files = []
    valid_ext = ('.pdf', '.docx', '.txt', '.json')
    for root, _, filenames in os.walk(RAW_DIR):
        for filename in filenames:
            if filename.lower().endswith(valid_ext):
                files.append(os.path.join(root, filename))

    if not files:
        print(f"[INFO] No files found in {RAW_DIR}. Add some documents first.")
        return
    
    print(f"[INFO] Found {len(files)} files to process.\n")

    with open(OUT_FILE, "w", encoding="utf-8") as out:
        for path in files:
            start = time.time()
            raw = extract_any(path)
            clean = clean_text(raw)
            duration = round(time.time() - start, 2)

            record = {
                "doc_id": os.path.basename(path),
                "clean_text": clean,
                "processing_time": duration
            }
            out.write(json.dumps(record) + "\n")
            print(f"[DONE] {os.path.basename(path)} processed in {duration}s")

    print(f"\n✅ Processing complete. Results saved to {OUT_FILE}")

if __name__ == "__main__":
    process_documents()
