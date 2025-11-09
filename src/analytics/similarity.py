# src/analytics/similarity.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict

class SimilarityCalculator:
    """
    Calculate document similarity using various metrics
    """
    
    @staticmethod
    def cosine_similarity_matrix(tfidf_matrix: np.ndarray) -> np.ndarray:
        """
        Compute pairwise cosine similarity for all documents
        """
        return cosine_similarity(tfidf_matrix)
    
    @staticmethod
    def find_similar_documents(
        doc_idx: int, 
        similarity_matrix: np.ndarray, 
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Find top k most similar documents to a given document
        """
        # Get similarity scores for the document
        similarities = similarity_matrix[doc_idx]
        
        # Sort indices by similarity (excluding the document itself)
        similar_indices = np.argsort(similarities)[::-1]
        
        # Filter out the document itself and get top k
        similar_docs = []
        for idx in similar_indices:
            if idx != doc_idx and len(similar_docs) < top_k:
                similar_docs.append((idx, similarities[idx]))
        
        return similar_docs
    
    @staticmethod
    def compute_similarity_threshold_groups(
        similarity_matrix: np.ndarray, 
        threshold: float = 0.7
    ) -> List[List[int]]:
        """
        Group documents that have similarity above threshold
        """
        n_docs = similarity_matrix.shape[0]
        groups = []
        visited = set()
        
        for i in range(n_docs):
            if i in visited:
                continue
            
            group = [i]
            visited.add(i)
            
            for j in range(i + 1, n_docs):
                if j not in visited and similarity_matrix[i][j] >= threshold:
                    group.append(j)
                    visited.add(j)
            
            if len(group) > 1:
                groups.append(group)
        
        return groups