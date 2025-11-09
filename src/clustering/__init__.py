# src/clustering/__init__.py

"""
Clustering Module
Distributed K-means clustering with metadata generation
"""

from .kmeans_cluster import SparkKMeansClustering
from .cluster_metadata import ClusterMetadataGenerator

__all__ = [
    'SparkKMeansClustering',
    'ClusterMetadataGenerator'
]

# Clustering parameters
DEFAULT_N_CLUSTERS = 5
DEFAULT_MAX_ITER = 100
MIN_CLUSTER_SIZE = 3