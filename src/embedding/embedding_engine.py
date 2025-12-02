# src/embedding/embedding_engine.py

import numpy as np
from typing import List, Dict, Optional, Callable
import threading
import queue
from sentence_transformers import SentenceTransformer
import torch
import time
import logging

class BackgroundEmbeddingEngine:
    """
    Asynchronous embedding generation engine

    This engine runs in the background and doesn't block the main system.
    The system can work with TF-IDF-based retrieval while embeddings are being
    generated asynchronously, then switch to embedding-based retrieval once ready.
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
        self.worker_thread = None

        # Performance tracking
        self.total_processed = 0
        self.total_time = 0
        self.start_time = None

        # Callbacks
        self.on_progress_callback = None
        self.on_complete_callback = None

        # Check for GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        self.logger = logging.getLogger("embedding")

    def start(self, on_progress: Optional[Callable] = None, on_complete: Optional[Callable] = None):
        """
        Start background processing thread

        Args:
            on_progress: Callback function(num_processed, total) called periodically
            on_complete: Callback function() called when all documents are processed
        """
        if not self.is_running:
            self.is_running = True
            self.start_time = time.time()
            self.on_progress_callback = on_progress
            self.on_complete_callback = on_complete

            self.worker_thread = threading.Thread(
                target=self._process_embeddings,
                daemon=True,
                name="EmbeddingWorker"
            )
            self.worker_thread.start()
            self.logger.info(f"Background embedding engine started on {self.device}")

    def stop(self, wait: bool = True):
        """
        Stop background processing

        Args:
            wait: If True, wait for current batch to finish
        """
        self.is_running = False
        if wait and hasattr(self, 'worker_thread') and self.worker_thread:
            self.worker_thread.join(timeout=10)
        self.logger.info("Background embedding engine stopped")

    def add_documents(self, documents: List[Dict], total_count: Optional[int] = None):
        """
        Add documents to embedding queue

        Args:
            documents: List of document dictionaries with 'id' and 'text'
            total_count: Total number of documents (for progress tracking)
        """
        for doc in documents:
            self.embedding_queue.put(doc)

        if total_count:
            self.total_to_process = total_count

        self.logger.info(f"Added {len(documents)} documents to embedding queue")

    def _process_embeddings(self):
        """Background worker to process embeddings"""
        batch = []
        last_progress_update = time.time()

        while self.is_running or not self.embedding_queue.empty():
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

                # Check if we're done
                if not self.is_running and self.embedding_queue.empty():
                    break

                continue

            # Process full batch
            if len(batch) >= self.batch_size:
                self._encode_batch(batch)
                batch = []

            # Progress callback
            if self.on_progress_callback and time.time() - last_progress_update > 5:
                self.on_progress_callback(self.total_processed, self.get_queue_size())
                last_progress_update = time.time()

        # Process any remaining documents
        if batch:
            self._encode_batch(batch)

        # Calculate final stats
        self.total_time = time.time() - self.start_time if self.start_time else 0

        # Completion callback
        if self.on_complete_callback:
            self.on_complete_callback()

        self.logger.info(f"Embedding generation complete: {self.total_processed} documents in {self.total_time:.2f}s")

    def _encode_batch(self, batch: List[Dict]):
        """
        Encode a batch of documents
        """
        batch_start = time.time()

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

        self.total_processed += len(batch)
        batch_time = time.time() - batch_start

        self.logger.debug(f"Processed batch of {len(batch)} documents in {batch_time:.2f}s")

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

    def get_queue_size(self) -> int:
        """Get number of documents waiting to be processed"""
        return self.embedding_queue.qsize()

    def get_progress(self) -> Dict:
        """Get current progress statistics"""
        return {
            'processed': self.total_processed,
            'queued': self.get_queue_size(),
            'elapsed_time': time.time() - self.start_time if self.start_time else 0,
            'is_running': self.is_running,
            'device': self.device
        }

    def is_complete(self) -> bool:
        """Check if all documents have been processed"""
        return self.embedding_queue.empty() and not self.is_running

    def wait_for_completion(self, timeout: Optional[float] = None):
        """
        Wait for all embeddings to be processed

        Args:
            timeout: Maximum time to wait in seconds (None = wait forever)
        """
        if self.worker_thread:
            self.worker_thread.join(timeout=timeout)