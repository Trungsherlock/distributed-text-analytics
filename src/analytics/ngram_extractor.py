# src/analytics/ngram_extractor.py

from collections import Counter
from typing import List, Tuple, Dict, Set
import itertools

class NgramExtractor:
    """
    Extract n-grams and top-frequency terms from documents
    """
    
    def __init__(self, n_range=(1, 3), top_k=20):
        """
        Args:
            n_range: Tuple of (min_n, max_n) for n-gram extraction
            top_k: Number of top terms to extract
        """
        self.n_range = n_range
        self.top_k = top_k
    
    def extract_ngrams(self, text: str, n: int) -> List[Tuple[str, ...]]:
        """
        Extract n-grams from text
        """
        tokens = text.split()
        ngrams = []
        
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams.append(ngram)
        
        return ngrams
    
    def extract_all_ngrams(self, text: str) -> Dict[int, List[Tuple[str, ...]]]:
        """
        Extract all n-grams within the specified range
        """
        ngrams_dict = {}
        
        for n in range(self.n_range[0], self.n_range[1] + 1):
            ngrams_dict[n] = self.extract_ngrams(text, n)
        
        return ngrams_dict
    
    def get_top_ngrams(self, text: str) -> Dict[int, List[Tuple[Tuple[str, ...], int]]]:
        """
        Get top k n-grams by frequency for each n
        """
        ngrams_dict = self.extract_all_ngrams(text)
        top_ngrams = {}
        
        for n, ngrams in ngrams_dict.items():
            counter = Counter(ngrams)
            top_ngrams[n] = counter.most_common(self.top_k)
        
        return top_ngrams
    
    def get_vocabulary(self, texts: List[str]) -> Set[str]:
        """
        Build vocabulary from multiple documents
        """
        vocab = set()
        for text in texts:
            vocab.update(text.split())
        return vocab