# src/api/models.py

"""
API Data Models
Request and response schemas for the REST API
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any

@dataclass
class DocumentUploadRequest:
    """Schema for document upload requests"""
    files: List[str]
    batch_name: Optional[str] = None
    auto_cluster: bool = True

@dataclass
class ClusteringRequest:
    """Schema for clustering requests"""
    n_clusters: Optional[int] = None
    max_iterations: int = 100
    document_ids: Optional[List[int]] = None

@dataclass
class SearchRequest:
    """Schema for search requests"""
    query: str
    cluster_id: Optional[int] = None
    top_k: int = 10
    use_embeddings: bool = True

@dataclass
class ApiResponse:
    """Standard API response format"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    message: Optional[str] = None
    
    def to_dict(self):
        return {
            'success': self.success,
            'data': self.data,
            'error': self.error,
            'message': self.message
        }