from src.ingest.extract_pdf import extract_any

path = "data/raw/sample.txt"
text = extract_any(path)
print("Extracted text:\n", text[:400])
