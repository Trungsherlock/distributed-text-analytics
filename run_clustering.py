#!/usr/bin/env python3
# run_clustering.py

"""
Stage 3: K-Means Clustering with Spark

Reads TF-IDF matrix and performs distributed K-Means clustering.
Generates cluster assignments and metadata.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from clustering.kmeans_cluster import SparkKMeansClustering
from clustering.cluster_metadata import ClusterMetadataGenerator


def run_clustering(
    documents_file: str = 'data/processed/documents.json',
    tfidf_file: str = 'data/processed/tfidf_matrix.npz',
    output_dir: str = 'data/processed',
    n_clusters: int = 10,
    num_workers: int = None
):
    """
    Run K-Means clustering on TF-IDF vectors

    Args:
        documents_file: Path to documents JSON file
        tfidf_file: Path to TF-IDF matrix (.npz file)
        output_dir: Output directory for cluster results
        n_clusters: Number of clusters
        num_workers: Number of Spark workers (None = auto)

    Returns:
        True if successful
    """
    print("=" * 60)
    print("STAGE 3: K-MEANS CLUSTERING WITH SPARK")
    print("=" * 60)

    # Load documents
    print(f"\n1. Loading documents from {documents_file}...")
    if not os.path.exists(documents_file):
        print(f"❌ Error: Documents file not found: {documents_file}")
        return False

    with open(documents_file, 'r') as f:
        documents = json.load(f)

    print(f"   ✓ Loaded {len(documents)} documents")

    # Load TF-IDF matrix
    print(f"\n2. Loading TF-IDF matrix from {tfidf_file}...")
    if not os.path.exists(tfidf_file):
        print(f"❌ Error: TF-IDF matrix not found: {tfidf_file}")
        print("\nPlease run feature extraction first:")
        print("  python run_feature_extraction.py")
        return False

    data = np.load(tfidf_file)
    tfidf_matrix = data['matrix']
    print(f"   ✓ Loaded TF-IDF matrix: {tfidf_matrix.shape}")

    # Run K-Means clustering
    print(f"\n3. Running K-Means clustering (k={n_clusters})...")
    if num_workers:
        print(f"   Using {num_workers} Spark worker(s)")

    clustering_engine = SparkKMeansClustering(
        n_clusters=min(n_clusters, len(documents)),
        max_iter=100,
        seed=42,
        num_workers=num_workers
    )

    cluster_assignments, perf_metrics = clustering_engine.fit_predict(tfidf_matrix)

    print(f"   ✓ Clustering complete!")
    print(f"   ✓ Time: {perf_metrics['clustering_time_seconds']:.2f}s")
    print(f"   ✓ Iterations: {perf_metrics['num_iterations']}")
    print(f"   ✓ Workers: {perf_metrics['num_workers']}")
    print(f"   ✓ Memory: {perf_metrics['memory_usage_mb']:.1f} MB")

    # Evaluate clustering
    print(f"\n4. Evaluating clustering quality...")
    silhouette_score = clustering_engine.evaluate_clustering()
    cluster_stats = clustering_engine.get_cluster_statistics(cluster_assignments)

    print(f"   ✓ Silhouette score: {silhouette_score:.3f}")
    print(f"   ✓ Number of clusters: {len(cluster_stats)}")

    # Print cluster distribution
    print(f"\n   Cluster distribution:")
    for cluster_id, stats in sorted(cluster_stats.items()):
        print(f"     Cluster {cluster_id}: {stats['size']} docs ({stats['percentage']:.1f}%)")

    # Generate cluster metadata
    print(f"\n5. Generating cluster metadata...")
    metadata_gen = ClusterMetadataGenerator()

    # Create simple vocabulary for labeling (since HashingTF doesn't provide one)
    vocabulary = [f"feature_{i}" for i in range(tfidf_matrix.shape[1])]

    cluster_data = {
        'clusters': {},
        'silhouette_score': silhouette_score,
        'total_documents': len(documents),
        'num_clusters': clustering_engine.n_clusters,
        'performance_metrics': perf_metrics
    }

    for cluster_id, stats in cluster_stats.items():
        metadata = metadata_gen.generate_cluster_metadata(
            cluster_id,
            stats['document_indices'],
            documents,
            tfidf_matrix,
            vocabulary
        )

        # Extract top keywords for simple label
        top_keywords = [term[0] for term in metadata['top_terms'][:5]]

        cluster_data['clusters'][cluster_id] = {
            **metadata,
            'simple_label': ' | '.join(top_keywords[:3]).upper(),
            'percentage': stats['percentage']
        }

    print(f"   ✓ Generated metadata for {len(cluster_data['clusters'])} clusters")

    # Save results
    print(f"\n6. Saving results...")
    os.makedirs(output_dir, exist_ok=True)

    # Save clusters
    clusters_path = os.path.join(output_dir, 'clusters.json')
    with open(clusters_path, 'w') as f:
        json.dump(cluster_data, f, indent=2)
    print(f"   ✓ Saved clusters to {clusters_path}")

    # Save cluster assignments (simple array)
    assignments_path = os.path.join(output_dir, 'cluster_assignments.npy')
    np.save(assignments_path, cluster_assignments)
    print(f"   ✓ Saved assignments to {assignments_path}")

    # Cleanup Spark
    clustering_engine.close()

    print("\n" + "=" * 60)
    print("STAGE 3 COMPLETE")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  - {clusters_path}")
    print(f"  - {assignments_path}")
    print(f"\nNext step:")
    print(f"  Run retrieval experiments:")
    print(f"    python run_retrieval_experiments.py")
    print(f"\n  OR run Spark scalability test:")
    print(f"    python test_spark_scalability.py")
    print("=" * 60 + "\n")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Stage 3: K-Means Clustering with Spark'
    )
    parser.add_argument(
        '--documents',
        type=str,
        default='data/processed/documents.json',
        help='Input documents JSON file'
    )
    parser.add_argument(
        '--tfidf',
        type=str,
        default='data/processed/tfidf_matrix.npz',
        help='Input TF-IDF matrix file (.npz)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed',
        help='Output directory for cluster results'
    )
    parser.add_argument(
        '--clusters',
        type=int,
        default=10,
        help='Number of clusters (K)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of Spark workers (default: auto-detect)'
    )

    args = parser.parse_args()

    success = run_clustering(
        documents_file=args.documents,
        tfidf_file=args.tfidf,
        output_dir=args.output,
        n_clusters=args.clusters,
        num_workers=args.workers
    )

    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
