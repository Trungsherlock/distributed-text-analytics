# src/embedding/__init__.py

"""
Embedding Module
Background embedding generation and vector database management
"""

from .embedding_engine import BackgroundEmbeddingEngine
from .vector_store import VectorStore

__all__ = [
    'BackgroundEmbeddingEngine',
    'VectorStore'
]

# Embedding configurations
DEFAULT_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
DEFAULT_BATCH_SIZE = 32
EMBEDDING_DIMENSION = 384  # For MiniLM