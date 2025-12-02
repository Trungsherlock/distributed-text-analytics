# src/api/query.py

import numpy as np
from typing import Optional
import time
from sentence_transformers import SentenceTransformer


class QueryEmbedder:
    """
    Converts query text to embedding vectors
    Uses same model as document embeddings for consistency
    """

    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize query embedder

        Args:
            model_name: HuggingFace model name (must match document embedding model)
        """
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed_query(self, query_text: str) -> tuple[np.ndarray, float]:
        """
        Convert query text to embedding vector

        Args:
            query_text: Query string

        Returns:
            Tuple of (embedding vector, embedding time in seconds)
        """
        start_time = time.time()
        embedding = self.model.encode(
            query_text,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        embedding_time = time.time() - start_time

        return embedding, embedding_time

    def embed_queries_batch(self, queries: list[str]) -> tuple[np.ndarray, float]:
        """
        Convert multiple queries to embeddings (batch processing)

        Args:
            queries: List of query strings

        Returns:
            Tuple of (embedding array, total embedding time in seconds)
        """
        start_time = time.time()
        embeddings = self.model.encode(
            queries,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=32
        )
        embedding_time = time.time() - start_time

        return embeddings, embedding_time
