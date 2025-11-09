# src/api/__init__.py

"""
API Module
REST API endpoints and request/response models
"""

from .routes import app
from .models import (
    DocumentUploadRequest,
    ClusteringRequest,
    SearchRequest,
    ApiResponse
)

__all__ = [
    'app',
    'DocumentUploadRequest',
    'ClusteringRequest',
    'SearchRequest',
    'ApiResponse'
]

# API Configuration
API_VERSION = "v1"
DEFAULT_PAGE_SIZE = 20
MAX_UPLOAD_SIZE_MB = 100