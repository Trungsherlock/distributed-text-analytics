# src/embedding/vector_store.py

import numpy as np
from typing import List, Dict, Optional, Tuple
import faiss
import os
import pickle

class VectorStore:
    """
    FAISS-based vector database for document embeddings

    Why FAISS (from project strategy):
    - Fast similarity search - optimized for vector operations
    - Simple to use - no server setup, just a Python library
    - Industry standard - used in production systems
    - Good enough for this project scale
    """

    def __init__(self, persist_directory: str = './data/embeddings'):
        """
        Initialize FAISS vector store

        Args:
            persist_directory: Directory to persist embeddings
        """
        self.persist_directory = persist_directory
        self.dimension = None
        self.indices = {}  # One index per cluster
        self.id_map = {}   # Map from doc_id to index position
        os.makedirs(persist_directory, exist_ok=True)
    
    def add_embeddings(
        self,
        embeddings: Dict[int, np.ndarray],
        cluster_assignments: Dict[int, int],
        metadata: Optional[Dict[int, Dict]] = None
    ):
        """
        Add embeddings to FAISS vector store

        Args:
            embeddings: Dictionary of doc_id -> embedding
            cluster_assignments: Dictionary of doc_id -> cluster_id
            metadata: Optional metadata (not used in FAISS, kept for compatibility)
        """
        # Set dimension if not set
        if self.dimension is None:
            self.dimension = next(iter(embeddings.values())).shape[0]

        # Group by cluster
        clusters = {}
        for doc_id, embedding in embeddings.items():
            cluster_id = cluster_assignments.get(doc_id, -1)
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append((doc_id, embedding))

        # Add to indices
        for cluster_id, docs in clusters.items():
            if cluster_id not in self.indices:
                # Create new FAISS index for this cluster
                # Using IndexFlatL2 for exact L2 distance search
                self.indices[cluster_id] = faiss.IndexFlatL2(self.dimension)
                self.id_map[cluster_id] = []

            index = self.indices[cluster_id]
            id_list = self.id_map[cluster_id]

            # Add embeddings to index
            embeddings_array = np.array([emb for _, emb in docs]).astype('float32')
            index.add(embeddings_array)

            # Update ID map
            id_list.extend([doc_id for doc_id, _ in docs])
    
    def search(
        self,
        query_embedding: np.ndarray,
        cluster_id: Optional[int] = None,
        k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Search for similar documents using FAISS

        Args:
            query_embedding: Query embedding vector
            cluster_id: Optional cluster to search within (None = search all)
            k: Number of results to return

        Returns:
            List of (doc_id, distance) tuples sorted by distance
        """
        query = query_embedding.reshape(1, -1).astype('float32')

        if cluster_id is not None:
            # Search within specific cluster
            if cluster_id not in self.indices:
                return []

            index = self.indices[cluster_id]
            id_list = self.id_map[cluster_id]

            # FAISS search
            distances, indices = index.search(query, min(k, len(id_list)))

            # Format results
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx != -1:  # Valid result
                    results.append((id_list[idx], float(dist)))

            return results

        else:
            # Search across all clusters (flat retrieval)
            all_results = []

            for cid, index in self.indices.items():
                id_list = self.id_map[cid]
                distances, indices = index.search(query, min(k, len(id_list)))

                for dist, idx in zip(distances[0], indices[0]):
                    if idx != -1:
                        all_results.append((id_list[idx], float(dist)))

            # Sort by distance and return top k
            all_results.sort(key=lambda x: x[1])
            return all_results[:k]

    def save(self, filepath: Optional[str] = None):
        """
        Save FAISS indices to disk

        Args:
            filepath: Base filepath (will save multiple files)
        """
        if filepath is None:
            filepath = os.path.join(self.persist_directory, 'vector_store')

        # Save each index
        for cluster_id, index in self.indices.items():
            index_path = f"{filepath}_cluster_{cluster_id}.index"
            faiss.write_index(index, index_path)

        # Save ID mappings
        id_map_path = f"{filepath}_id_map.pkl"
        with open(id_map_path, 'wb') as f:
            pickle.dump({
                'id_map': self.id_map,
                'dimension': self.dimension
            }, f)

    def load(self, filepath: Optional[str] = None):
        """
        Load FAISS indices from disk

        Args:
            filepath: Base filepath (will load multiple files)
        """
        if filepath is None:
            filepath = os.path.join(self.persist_directory, 'vector_store')

        # Load ID mappings
        id_map_path = f"{filepath}_id_map.pkl"
        if not os.path.exists(id_map_path):
            raise FileNotFoundError(f"ID map not found: {id_map_path}")

        with open(id_map_path, 'rb') as f:
            data = pickle.load(f)
            self.id_map = data['id_map']
            self.dimension = data['dimension']

        # Load each index
        self.indices = {}
        for cluster_id in self.id_map.keys():
            index_path = f"{filepath}_cluster_{cluster_id}.index"
            if os.path.exists(index_path):
                self.indices[cluster_id] = faiss.read_index(index_path)

    def get_stats(self) -> Dict:
        """Get vector store statistics"""
        total_docs = sum(len(id_list) for id_list in self.id_map.values())
        return {
            'total_documents': total_docs,
            'num_clusters': len(self.indices),
            'dimension': self.dimension,
            'cluster_sizes': {cid: len(id_list) for cid, id_list in self.id_map.items()}
        }