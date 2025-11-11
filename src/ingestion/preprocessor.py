# src/ingestion/preprocessor.py

import re
import string
from typing import List, Set
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

class TextPreprocessor:
    """
    Text normalization and preprocessing
    """
    
    def __init__(self, remove_stopwords=True, lowercase=True, remove_punct=True):
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        self.remove_punct = remove_punct
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
    
    def preprocess(self, text: str) -> str:
        """
        Full preprocessing pipeline
        """
        # Basic cleaning
        text = self._clean_text(text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation
        if self.remove_punct:
            text = self._remove_punctuation(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]
        
        return ' '.join(tokens)
    
    def _clean_text(self, text: str) -> str:
        """Remove special characters and normalize whitespace"""
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove non-ASCII characters
        text = text.encode('ascii', 'ignore').decode('ascii')
        return text.strip()
    
    def _remove_punctuation(self, text: str) -> str:
        """Remove punctuation while preserving word boundaries"""
        translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        return text.translate(translator)