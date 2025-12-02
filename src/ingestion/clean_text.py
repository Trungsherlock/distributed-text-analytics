import re


def clean_text(raw_text: str, simple: bool = True) -> str:
    """
    Cleans and normalizes text.

    Args:
        raw_text: Raw text input
        simple: If True, only basic cleaning (recommended for distributed systems demo)
                If False, apply advanced preprocessing (lemmatization, stopwords)

    Simple mode (default):
      1. Lowercase
      2. Remove non-letter characters
      3. Remove extra whitespace

    Advanced mode:
      1. Lowercase & remove non-letter characters
      2. Tokenize
      3. Remove stopwords
      4. Lemmatize

    Note: Simple mode is recommended per project strategy - complex preprocessing
    adds computation time but doesn't help demonstrate distributed systems concepts.
    """
    if simple:
        # Basic text extraction - fast and sufficient for clustering
        text = re.sub(r'[^a-zA-Z\s]', '', raw_text.lower())
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    else:
        # Advanced preprocessing (optional)
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        from nltk.tokenize import word_tokenize

        # Ensure NLTK resources
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

        lemmatizer = WordNetLemmatizer()
        stops = set(stopwords.words("english"))

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
