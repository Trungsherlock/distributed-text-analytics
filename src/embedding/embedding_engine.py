# src/embedding/embedding_engine.py

import numpy as np
from typing import List, Dict, Optional
import threading
import queue
from sentence_transformers import SentenceTransformer
import torch
import time

class BackgroundEmbeddingEngine:
    """
    Asynchronous embedding generation engine
    """
    
    def __init__(
        self, 
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        batch_size: int = 32
    ):
        """
        Initialize embedding engine
        
        Args:
            model_name: HuggingFace model name
            batch_size: Batch size for encoding
        """
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.embedding_queue = queue.Queue()
        self.results = {}
        self.is_running = False
        
        # Check for GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
    def start(self):
        """Start background processing thread"""
        if not self.is_running:
            self.is_running = True
            self.worker_thread = threading.Thread(
                target=self._process_embeddings, 
                daemon=True
            )
            self.worker_thread.start()
    
    def stop(self):
        """Stop background processing"""
        self.is_running = False
        if hasattr(self, 'worker_thread'):
            self.worker_thread.join(timeout=5)
    
    def add_documents(self, documents: List[Dict]):
        """
        Add documents to embedding queue
        
        Args:
            documents: List of document dictionaries with 'id' and 'text'
        """
        for doc in documents:
            self.embedding_queue.put(doc)
    
    def _process_embeddings(self):
        """Background worker to process embeddings"""
        batch = []
        
        while self.is_running:
            try:
                # Collect batch
                while len(batch) < self.batch_size:
                    doc = self.embedding_queue.get(timeout=1)
                    batch.append(doc)
                
            except queue.Empty:
                # Process partial batch if available
                if batch:
                    self._encode_batch(batch)
                    batch = []
                continue
            
            # Process full batch
            if len(batch) >= self.batch_size:
                self._encode_batch(batch)
                batch = []
    
    def _encode_batch(self, batch: List[Dict]):
        """
        Encode a batch of documents
        """
        texts = [doc['text'] for doc in batch]
        ids = [doc['id'] for doc in batch]
        cluster_ids = [doc.get('cluster_id', -1) for doc in batch]
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Store results
        for doc_id, cluster_id, embedding in zip(ids, cluster_ids, embeddings):
            self.results[doc_id] = {
                'embedding': embedding,
                'cluster_id': cluster_id,
                'timestamp': time.time()
            }
    
    def get_embedding(self, doc_id: int) -> Optional[np.ndarray]:
        """Get embedding for a document"""
        if doc_id in self.results:
            return self.results[doc_id]['embedding']
        return None
    
    def get_all_embeddings(self) -> Dict[int, np.ndarray]:
        """Get all computed embeddings"""
        return {
            doc_id: data['embedding'] 
            for doc_id, data in self.results.items()
        }