from src.ingestion.parser import DocumentParser

path = "data/raw/resume/10001727.pdf"
parser = DocumentParser()
text = parser.parse_document(path)
print(text)
