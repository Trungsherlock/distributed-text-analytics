# src/clustering/cluster_metadata.py

import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
import re

class ClusterMetadataGenerator:
    """
    Generate metadata and labels for clusters
    """
    
    def __init__(self, top_terms_count: int = 10):
        self.top_terms_count = top_terms_count
    
    def generate_cluster_metadata(
        self,
        cluster_id: int,
        document_indices: List[int],
        documents: List[Dict],
        tfidf_matrix: np.ndarray,
        vocabulary: List[str]
    ) -> Dict:
        """
        Generate comprehensive metadata for a cluster
        
        Args:
            cluster_id: Cluster identifier
            document_indices: Indices of documents in this cluster
            documents: Original document data
            tfidf_matrix: TF-IDF feature matrix
            vocabulary: Feature vocabulary
            
        Returns:
            Dictionary containing cluster metadata
        """
        # Extract cluster documents
        cluster_docs = [documents[idx] for idx in document_indices]
        cluster_vectors = tfidf_matrix[document_indices]
        
        # Calculate cluster centroid
        centroid = np.mean(cluster_vectors, axis=0)
        
        # Get top terms based on centroid
        top_term_indices = np.argsort(centroid)[-self.top_terms_count:][::-1]
        top_terms = [
            (vocabulary[idx] if idx < len(vocabulary) else f"term_{idx}", 
             centroid[idx])
            for idx in top_term_indices if centroid[idx] > 0
        ]
        
        # Generate cluster label
        label = self._generate_cluster_label(top_terms, cluster_docs)
        
        # Get document formats
        formats = Counter([doc.get('format', 'unknown') for doc in cluster_docs])
        
        # Calculate average document length
        avg_length = np.mean([
            doc.get('word_count', 0) for doc in cluster_docs
        ])
        
        # Get representative snippets
        snippets = self._extract_representative_snippets(
            cluster_docs, top_terms
        )
        
        metadata = {
            'cluster_id': cluster_id,
            'label': label,
            'size': len(document_indices),
            'document_indices': document_indices,
            'top_terms': top_terms,
            'formats': dict(formats),
            'average_doc_length': avg_length,
            'representative_snippets': snippets,
            'centroid_norm': np.linalg.norm(centroid)
        }
        
        return metadata
    
    def _generate_cluster_label(
        self, 
        top_terms: List[Tuple[str, float]], 
        documents: List[Dict]
    ) -> str:
        """
        Generate human-readable cluster label
        """
        # Use top 3 terms for label
        if len(top_terms) >= 3:
            label_terms = [term[0] for term in top_terms[:3]]
            label = ' / '.join(label_terms).title()
        else:
            label = 'Cluster'
        
        # Try to identify common document types
        if any('financial' in str(doc.get('text', '')).lower() 
               for doc in documents[:5]):
            label = f"Financial Documents - {label}"
        elif any('clinical' in str(doc.get('text', '')).lower() 
                 or 'medical' in str(doc.get('text', '')).lower() 
                 for doc in documents[:5]):
            label = f"Medical/Clinical - {label}"
        elif any('legal' in str(doc.get('text', '')).lower() 
                 for doc in documents[:5]):
            label = f"Legal Documents - {label}"
        
        return label
    
    def _extract_representative_snippets(
        self, 
        documents: List[Dict], 
        top_terms: List[Tuple[str, float]],
        max_snippets: int = 3
    ) -> List[str]:
        """
        Extract representative text snippets from cluster
        """
        snippets = []
        term_words = [term[0] for term in top_terms[:5]]
        
        for doc in documents[:10]:  # Check first 10 documents
            text = doc.get('text', '')
            if not text:
                continue
            
            # Find sentences containing top terms
            sentences = re.split(r'[.!?]+', text)
            for sentence in sentences:
                sentence = sentence.strip()
                if any(term in sentence.lower() for term in term_words):
                    if 20 < len(sentence.split()) < 50:  # Reasonable length
                        snippets.append(sentence)
                        if len(snippets) >= max_snippets:
                            return snippets
        
        return snippets