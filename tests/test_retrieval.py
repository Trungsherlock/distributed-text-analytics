#!/usr/bin/env python3
# tests/test_retrieval.py

"""
Unit tests for Stage 5: Query & Retrieval
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
import numpy as np
from retrieval.query import QueryEmbedder
from retrieval.retrieval import ClusterAwareRetrieval, FlatRetrieval, RetrievalComparator
from embedding.vector_store import VectorStore


class TestQueryEmbedder(unittest.TestCase):
    """Test query embedding"""

    def setUp(self):
        self.embedder = QueryEmbedder()

    def test_embed_single_query(self):
        """Test embedding a single query"""
        query = "test query"
        embedding, time = self.embedder.embed_query(query)

        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(len(embedding.shape), 1)  # 1D vector
        self.assertGreater(len(embedding), 0)
        self.assertGreater(time, 0)  # Should take some time

    def test_embed_batch_queries(self):
        """Test embedding multiple queries"""
        queries = ["query 1", "query 2", "query 3"]
        embeddings, time = self.embedder.embed_queries_batch(queries)

        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(embeddings.shape[0], 3)
        self.assertGreater(time, 0)

    def test_embedding_consistency(self):
        """Test that same query produces similar embeddings"""
        query = "machine learning"
        emb1, _ = self.embedder.embed_query(query)
        emb2, _ = self.embedder.embed_query(query)

        # Should be very close (cosine similarity near 1.0)
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        self.assertGreater(similarity, 0.99)


class TestRetrieval(unittest.TestCase):
    """Test retrieval systems"""

    def setUp(self):
        """Create mock data for testing"""
        # Create mock documents with embeddings
        np.random.seed(42)
        n_docs = 100
        n_clusters = 5
        embedding_dim = 384

        # Generate clustered embeddings
        self.embeddings = {}
        self.cluster_assignments = {}
        self.cluster_centroids = np.random.randn(n_clusters, embedding_dim)

        for i in range(n_docs):
            cluster_id = i % n_clusters
            # Generate embedding close to cluster centroid
            noise = np.random.randn(embedding_dim) * 0.1
            self.embeddings[i] = self.cluster_centroids[cluster_id] + noise
            self.cluster_assignments[i] = cluster_id

        # Create vector store
        self.vector_store = VectorStore(store_type='faiss')
        self.vector_store.add_embeddings(self.embeddings, self.cluster_assignments)

        # Create retrieval systems
        self.cluster_aware = ClusterAwareRetrieval(
            cluster_centroids=self.cluster_centroids,
            vector_store=self.vector_store,
            cluster_assignments=self.cluster_assignments
        )

        self.flat = FlatRetrieval(
            vector_store=self.vector_store,
            total_documents=n_docs
        )

    def test_cluster_aware_search(self):
        """Test cluster-aware retrieval"""
        # Query similar to cluster 0 centroid
        query_embedding = self.cluster_centroids[0] + np.random.randn(384) * 0.05

        result = self.cluster_aware.search(query_embedding, k=5)

        # Check result structure
        self.assertIn('results', result)
        self.assertIn('cluster_id', result)
        self.assertIn('timing', result)

        # Should find cluster 0
        self.assertEqual(result['cluster_id'], 0)

        # Should return 5 results
        self.assertEqual(len(result['results']), 5)

        # Check timing data
        self.assertIn('cluster_selection_ms', result['timing'])
        self.assertIn('faiss_search_ms', result['timing'])
        self.assertIn('total_latency_ms', result['timing'])
        self.assertGreater(result['timing']['total_latency_ms'], 0)

    def test_flat_search(self):
        """Test flat retrieval"""
        query_embedding = self.cluster_centroids[0]

        result = self.flat.search(query_embedding, k=5)

        # Check result structure
        self.assertIn('results', result)
        self.assertIn('timing', result)

        # Should return 5 results
        self.assertEqual(len(result['results']), 5)

        # Should search all documents
        self.assertEqual(result['timing']['documents_searched'], 100)

    def test_cluster_aware_searches_fewer_docs(self):
        """Test that cluster-aware searches fewer documents"""
        query_embedding = self.cluster_centroids[0]

        cluster_result = self.cluster_aware.search(query_embedding, k=5)
        flat_result = self.flat.search(query_embedding, k=5)

        # Cluster-aware should search fewer documents
        cluster_docs = cluster_result['timing']['documents_searched']
        flat_docs = flat_result['timing']['documents_searched']

        self.assertLess(cluster_docs, flat_docs)
        self.assertEqual(flat_docs, 100)

    def test_retrieval_comparator(self):
        """Test retrieval comparator"""
        comparator = RetrievalComparator(self.cluster_aware, self.flat)

        query_embedding = self.cluster_centroids[0]
        result = comparator.compare(query_embedding, k=5)

        # Check structure
        self.assertIn('cluster_aware', result)
        self.assertIn('flat', result)
        self.assertIn('comparison', result)

        # Check comparison metrics
        comparison = result['comparison']
        self.assertIn('speedup', comparison)
        self.assertIn('search_space_reduction_pct', comparison)

        # Speedup should be positive
        self.assertGreater(comparison['speedup'], 0)

        # Search space should be reduced
        self.assertGreater(comparison['search_space_reduction_pct'], 0)


class TestVectorStore(unittest.TestCase):
    """Test vector store functionality"""

    def test_faiss_index_creation(self):
        """Test FAISS index creation"""
        vector_store = VectorStore(store_type='faiss')

        # Add embeddings
        embeddings = {
            0: np.random.randn(384),
            1: np.random.randn(384),
            2: np.random.randn(384)
        }
        cluster_assignments = {0: 0, 1: 0, 2: 1}

        vector_store.add_embeddings(embeddings, cluster_assignments)

        # Check indices created
        self.assertIn(0, vector_store.indices)
        self.assertIn(1, vector_store.indices)

    def test_faiss_search_within_cluster(self):
        """Test searching within a specific cluster"""
        vector_store = VectorStore(store_type='faiss')

        # Create distinct embeddings for two clusters
        cluster_0_embedding = np.ones(384)
        cluster_1_embedding = -np.ones(384)

        embeddings = {
            0: cluster_0_embedding,
            1: cluster_0_embedding + 0.1,
            2: cluster_1_embedding,
            3: cluster_1_embedding + 0.1
        }
        cluster_assignments = {0: 0, 1: 0, 2: 1, 3: 1}

        vector_store.add_embeddings(embeddings, cluster_assignments)

        # Search cluster 0
        results = vector_store.search(cluster_0_embedding, cluster_id=0, k=2)

        # Should only return docs from cluster 0
        doc_ids = [doc_id for doc_id, _ in results]
        self.assertIn(0, doc_ids)
        self.assertIn(1, doc_ids)
        self.assertNotIn(2, doc_ids)
        self.assertNotIn(3, doc_ids)


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
