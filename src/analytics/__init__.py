# src/analytics/__init__.py

"""
Text Analytics Module
Provides n-gram extraction, TF-IDF computation, and similarity measures
"""

from .ngram_extractor import NgramExtractor
from .tfidf_engine import SparkTFIDFEngine
from .similarity import SimilarityCalculator

__all__ = [
    'NgramExtractor',
    'SparkTFIDFEngine', 
    'SimilarityCalculator'
]

# Default configurations
DEFAULT_NGRAM_RANGE = (1, 3)
DEFAULT_NUM_FEATURES = 10000
DEFAULT_TOP_K_TERMS = 20