import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def _ensure_nltk_resources():
    """Download required NLTK resources if they are missing."""
    resources = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
    }
    for package, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


_ensure_nltk_resources()

lemmatizer = WordNetLemmatizer()
stops = set(stopwords.words("english"))

def clean_text(raw_text: str) -> str:
    """
    Cleans and normalizes text using NLTK.
    Steps:
      1. Lowercase & remove non-letter characters
      2. Tokenize
      3. Remove stopwords
      4. Lemmatize
    """
    # Lowercase and strip punctuation/numbers
    text = re.sub(r'[^a-zA-Z\s]', '', raw_text.lower())

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords and short tokens, then lemmatize
    lemmas = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stops and len(token) > 2
    ]

    return " ".join(lemmas)
