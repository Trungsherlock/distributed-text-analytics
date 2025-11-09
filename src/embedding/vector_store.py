# src/embedding/vector_store.py

import numpy as np
from typing import List, Dict, Optional, Tuple
import chromadb
from chromadb.config import Settings
import faiss

class VectorStore:
    """
    Vector database integration for embeddings
    Supports both Chroma and FAISS
    """
    
    def __init__(
        self, 
        store_type: str = 'chroma',
        persist_directory: str = './data/embeddings'
    ):
        """
        Initialize vector store
        
        Args:
            store_type: 'chroma' or 'faiss'
            persist_directory: Directory to persist embeddings
        """
        self.store_type = store_type
        self.persist_directory = persist_directory
        
        if store_type == 'chroma':
            self._init_chroma()
        elif store_type == 'faiss':
            self._init_faiss()
        else:
            raise ValueError(f"Unknown store type: {store_type}")
    
    def _init_chroma(self):
        """Initialize Chroma database"""
        self.chroma_client = chromadb.Client(
            Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=self.persist_directory
            )
        )
        
        # Create collection for each cluster
        self.collections = {}
    
    def _init_faiss(self):
        """Initialize FAISS index"""
        self.dimension = None
        self.indices = {}  # One index per cluster
        self.id_map = {}   # Map from doc_id to index position
    
    def add_embeddings(
        self, 
        embeddings: Dict[int, np.ndarray],
        cluster_assignments: Dict[int, int],
        metadata: Optional[Dict[int, Dict]] = None
    ):
        """
        Add embeddings to vector store
        
        Args:
            embeddings: Dictionary of doc_id -> embedding
            cluster_assignments: Dictionary of doc_id -> cluster_id
            metadata: Optional metadata for each document
        """
        if self.store_type == 'chroma':
            self._add_to_chroma(embeddings, cluster_assignments, metadata)
        else:
            self._add_to_faiss(embeddings, cluster_assignments)
    
    def _add_to_chroma(
        self, 
        embeddings: Dict[int, np.ndarray],
        cluster_assignments: Dict[int, int],
        metadata: Optional[Dict[int, Dict]]
    ):
        """Add embeddings to Chroma"""
        # Group by cluster
        clusters = {}
        for doc_id, embedding in embeddings.items():
            cluster_id = cluster_assignments.get(doc_id, -1)
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append((doc_id, embedding))
        
        # Add to collections
        for cluster_id, docs in clusters.items():
            collection_name = f"cluster_{cluster_id}"
            
            if collection_name not in self.collections:
                self.collections[collection_name] = \
                    self.chroma_client.create_collection(collection_name)
            
            collection = self.collections[collection_name]
            
            # Prepare data
            ids = [str(doc_id) for doc_id, _ in docs]
            embeddings_list = [emb.tolist() for _, emb in docs]
            metadatas = []
            
            for doc_id, _ in docs:
                doc_metadata = metadata.get(doc_id, {}) if metadata else {}
                doc_metadata['cluster_id'] = cluster_id
                metadatas.append(doc_metadata)
            
            # Add to collection
            collection.add(
                embeddings=embeddings_list,
                metadatas=metadatas,
                ids=ids
            )
    
    def _add_to_faiss(
        self, 
        embeddings: Dict[int, np.ndarray],
        cluster_assignments: Dict[int, int]
    ):
        """Add embeddings to FAISS"""
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
                # Create new index
                self.indices[cluster_id] = faiss.IndexFlatL2(self.dimension)
                self.id_map[cluster_id] = []
            
            index = self.indices[cluster_id]
            id_list = self.id_map[cluster_id]
            
            # Add embeddings
            embeddings_array = np.array([emb for _, emb in docs])
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
        Search for similar documents
        
        Args:
            query_embedding: Query embedding vector
            cluster_id: Optional cluster to search within
            k: Number of results
            
        Returns:
            List of (doc_id, distance) tuples
        """
        if self.store_type == 'chroma':
            return self._search_chroma(query_embedding, cluster_id, k)
        else:
            return self._search_faiss(query_embedding, cluster_id, k)
    
    def _search_chroma(
        self, 
        query_embedding: np.ndarray,
        cluster_id: Optional[int],
        k: int
    ) -> List[Tuple[int, float]]:
        """Search in Chroma"""
        if cluster_id is not None:
            collection_name = f"cluster_{cluster_id}"
            if collection_name not in self.collections:
                return []
            
            collection = self.collections[collection_name]
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k
            )
            
            # Format results
            doc_ids = [int(id_) for id_ in results['ids'][0]]
            distances = results['distances'][0]
            
            return list(zip(doc_ids, distances))
        
        else:
            # Search all collections
            all_results = []
            for collection in self.collections.values():
                results = collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=k
                )
                
                for id_, dist in zip(results['ids'][0], results['distances'][0]):
                    all_results.append((int(id_), dist))
            
            # Sort by distance and return top k
            all_results.sort(key=lambda x: x[1])
            return all_results[:k]
    
    def _search_faiss(
        self, 
        query_embedding: np.ndarray,
        cluster_id: Optional[int],
        k: int
    ) -> List[Tuple[int, float]]:
        """Search in FAISS"""
        query = query_embedding.reshape(1, -1)
        
        if cluster_id is not None:
            if cluster_id not in self.indices:
                return []
            
            index = self.indices[cluster_id]
            id_list = self.id_map[cluster_id]
            
            distances, indices = index.search(query, min(k, len(id_list)))
            
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx != -1:  # Valid result
                    results.append((id_list[idx], float(dist)))
            
            return results
        
        else:
            # Search all indices
            all_results = []
            
            for cluster_id, index in self.indices.items():
                id_list = self.id_map[cluster_id]
                distances, indices = index.search(query, min(k, len(id_list)))
                
                for dist, idx in zip(distances[0], indices[0]):
                    if idx != -1:
                        all_results.append((id_list[idx], float(dist)))
            
            # Sort by distance and return top k
            all_results.sort(key=lambda x: x[1])
            return all_results[:k]