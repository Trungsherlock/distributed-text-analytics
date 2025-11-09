# src/ingestion/__init__.py

"""
Document Ingestion Module
Handles parsing and preprocessing of multiple document formats
"""

from .parser import DocumentParser
from .preprocessor import TextPreprocessor

__all__ = ['DocumentParser', 'TextPreprocessor']

# Module-level configuration
SUPPORTED_FORMATS = {'.pdf', '.docx', '.txt', '.json'}
DEFAULT_BATCH_SIZE = 32
MAX_FILE_SIZE_MB = 100