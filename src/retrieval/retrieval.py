# src/api/retrieval.py

import numpy as np
from typing import Optional, Dict, List, Tuple
import time
from sklearn.metrics.pairwise import cosine_similarity


class ClusterAwareRetrieval:
    """
    Cluster-aware document retrieval
    Searches only within the most relevant cluster for faster queries
    """

    def __init__(
        self,
        cluster_centroids: np.ndarray,
        vector_store,
        cluster_assignments: Dict[int, int]
    ):
        """
        Initialize cluster-aware retrieval

        Args:
            cluster_centroids: Array of cluster centroid vectors (n_clusters x embedding_dim)
            vector_store: VectorStore instance with indexed documents
            cluster_assignments: Dict mapping doc_id -> cluster_id
        """
        self.cluster_centroids = cluster_centroids
        self.vector_store = vector_store
        self.cluster_assignments = cluster_assignments
        self.n_clusters = len(cluster_centroids)

    def find_relevant_cluster(self, query_embedding: np.ndarray) -> Tuple[int, float]:
        """
        Find the cluster most similar to the query

        Args:
            query_embedding: Query embedding vector

        Returns:
            Tuple of (cluster_id, similarity_score)
        """
        start_time = time.time()

        # Calculate cosine similarity between query and all centroids
        query_reshaped = query_embedding.reshape(1, -1)
        similarities = cosine_similarity(query_reshaped, self.cluster_centroids)[0]

        # Find cluster with highest similarity
        best_cluster_id = int(np.argmax(similarities))
        best_similarity = float(similarities[best_cluster_id])

        cluster_selection_time = time.time() - start_time

        return best_cluster_id, best_similarity, cluster_selection_time

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> Dict:
        """
        Perform cluster-aware retrieval

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return

        Returns:
            Dictionary with results and timing breakdown
        """
        timing = {}
        start_time = time.time()

        # Step 1: Find relevant cluster
        cluster_id, similarity, cluster_time = self.find_relevant_cluster(query_embedding)
        timing['cluster_selection_ms'] = round(cluster_time * 1000, 2)

        # Step 2: Search within cluster
        search_start = time.time()
        results = self.vector_store.search(
            query_embedding=query_embedding,
            cluster_id=cluster_id,
            k=k
        )
        timing['faiss_search_ms'] = round((time.time() - search_start) * 1000, 2)

        # Calculate documents searched
        docs_in_cluster = sum(1 for cid in self.cluster_assignments.values() if cid == cluster_id)
        timing['documents_searched'] = docs_in_cluster
        timing['total_latency_ms'] = round((time.time() - start_time) * 1000, 2)

        return {
            'results': results,
            'cluster_id': cluster_id,
            'cluster_similarity': similarity,
            'timing': timing
        }


class FlatRetrieval:
    """
    Baseline flat retrieval
    Searches all documents without clustering
    """

    def __init__(self, vector_store, total_documents: int):
        """
        Initialize flat retrieval

        Args:
            vector_store: VectorStore instance with indexed documents
            total_documents: Total number of documents in corpus
        """
        self.vector_store = vector_store
        self.total_documents = total_documents

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> Dict:
        """
        Perform flat retrieval across all documents

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return

        Returns:
            Dictionary with results and timing breakdown
        """
        timing = {}
        start_time = time.time()

        # Search all documents
        search_start = time.time()
        results = self.vector_store.search(
            query_embedding=query_embedding,
            cluster_id=None,  # Search all clusters
            k=k
        )
        timing['faiss_search_ms'] = round((time.time() - search_start) * 1000, 2)

        timing['documents_searched'] = self.total_documents
        timing['total_latency_ms'] = round((time.time() - start_time) * 1000, 2)

        return {
            'results': results,
            'timing': timing
        }


class RetrievalComparator:
    """
    Compares cluster-aware vs flat retrieval performance
    """

    def __init__(
        self,
        cluster_aware_retrieval: ClusterAwareRetrieval,
        flat_retrieval: FlatRetrieval
    ):
        """
        Initialize comparator

        Args:
            cluster_aware_retrieval: ClusterAwareRetrieval instance
            flat_retrieval: FlatRetrieval instance
        """
        self.cluster_aware = cluster_aware_retrieval
        self.flat = flat_retrieval

    def compare(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> Dict:
        """
        Run both retrieval methods and compare

        Args:
            query_embedding: Query embedding vector
            k: Number of results

        Returns:
            Comparison results with both methods' outputs
        """
        cluster_result = self.cluster_aware.search(query_embedding, k)
        flat_result = self.flat.search(query_embedding, k)

        # Calculate speedup
        speedup = flat_result['timing']['total_latency_ms'] / cluster_result['timing']['total_latency_ms']

        # Calculate search space reduction
        search_reduction = 1 - (
            cluster_result['timing']['documents_searched'] /
            flat_result['timing']['documents_searched']
        )

        return {
            'cluster_aware': cluster_result,
            'flat': flat_result,
            'comparison': {
                'speedup': round(speedup, 2),
                'search_space_reduction_pct': round(search_reduction * 100, 1),
                'latency_difference_ms': round(
                    flat_result['timing']['total_latency_ms'] -
                    cluster_result['timing']['total_latency_ms'],
                    2
                )
            }
        }
