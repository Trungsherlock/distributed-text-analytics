from src.ingestion.clean_text import clean_text

sample = """The quick brown foxes were jumping over lazy dogs near the riverbank.
However, the foxes didn’t know it was a trap!"""

print("Cleaned text:")
print(clean_text(sample))
