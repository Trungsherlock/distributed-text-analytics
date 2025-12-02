#!/usr/bin/env python3
# prepare_experiment_data.py

"""
Prepare data for Stage 5 retrieval experiments
Creates documents.json and clusters.json from processed data
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def prepare_data_from_raw(
    raw_dir: str = 'data/raw',
    output_dir: str = 'data/processed',
    n_clusters: int = 10
):
    """
    Process raw documents and create clustering data

    Args:
        raw_dir: Directory with raw documents
        output_dir: Output directory for processed files
        n_clusters: Number of clusters to create
    """
    print("=" * 60)
    print("PREPARING DATA FOR RETRIEVAL EXPERIMENTS")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Import modules
    from ingestion.parser import DocumentParser
    from ingestion.preprocessor import TextPreprocessor
    from analytics.ngram_extractor import NgramExtractor
    from analytics.tfidf_engine import SparkTFIDFEngine
    from clustering.kmeans_cluster import SparkKMeansClustering
    from clustering.cluster_metadata import ClusterMetadataGenerator

    parser = DocumentParser()
    preprocessor = TextPreprocessor()
    ngram_extractor = NgramExtractor()

    # Step 1: Parse and preprocess documents
    print("\n1. Parsing and preprocessing documents...")
    documents = []
    doc_id = 0

    raw_path = Path(raw_dir)
    file_count = 0

    for file_path in raw_path.rglob('*'):
        if file_path.is_file():
            extension = file_path.suffix.lower()
            if extension in DocumentParser.SUPPORTED_FORMATS:
                result = parser.parse_document(str(file_path))

                if result['success']:
                    processed_text = preprocessor.preprocess(result['text'])
                    top_ngrams = ngram_extractor.get_top_ngrams(processed_text)

                    documents.append({
                        'id': doc_id,
                        'file_name': result['file_name'],
                        'format': result['format'],
                        'text': processed_text,
                        'original_text': result.get('text', ''),
                        'preview_text': result.get('text', '')[:500],
                        'ngrams': top_ngrams,
                        'metadata': result['metadata'],
                        'word_count': result['word_count']
                    })

                    doc_id += 1
                    file_count += 1

                    if file_count % 10 == 0:
                        print(f"   Processed {file_count} documents...")

    print(f"   ✓ Processed {len(documents)} documents")

    if len(documents) < 2:
        print("\n❌ Error: Need at least 2 documents for clustering")
        return False

    # Step 2: Compute TF-IDF
    print("\n2. Computing TF-IDF features...")
    tfidf_engine = SparkTFIDFEngine()
    docs_for_tfidf = [
        {'id': doc['id'], 'text': doc['text']}
        for doc in documents
    ]
    tfidf_matrix, vocabulary = tfidf_engine.compute_tfidf(docs_for_tfidf)
    print(f"   ✓ TF-IDF matrix shape: {tfidf_matrix.shape}")

    # Step 3: Perform clustering
    print("\n3. Performing K-Means clustering...")
    clustering_engine = SparkKMeansClustering(
        n_clusters=min(n_clusters, len(documents))
    )
    cluster_assignments, perf_metrics = clustering_engine.fit_predict(tfidf_matrix)
    silhouette_score = clustering_engine.evaluate_clustering()
    cluster_stats = clustering_engine.get_cluster_statistics(cluster_assignments)

    print(f"   ✓ Created {len(cluster_stats)} clusters")
    print(f"   ✓ Silhouette score: {silhouette_score:.3f}")

    # Step 4: Generate cluster metadata
    print("\n4. Generating cluster metadata...")
    metadata_gen = ClusterMetadataGenerator()
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

        top_keywords = [term[0] for term in metadata['top_terms'][:5]]

        cluster_data['clusters'][cluster_id] = {
            **metadata,
            'simple_label': ' | '.join(top_keywords[:3]).upper(),
            'percentage': stats['percentage']
        }

    # Step 5: Save data
    print("\n5. Saving data...")

    # Save documents
    docs_path = os.path.join(output_dir, 'documents.json')
    with open(docs_path, 'w') as f:
        json.dump(documents, f, indent=2)
    print(f"   ✓ Saved {len(documents)} documents to {docs_path}")

    # Save clusters
    clusters_path = os.path.join(output_dir, 'clusters.json')
    with open(clusters_path, 'w') as f:
        json.dump(cluster_data, f, indent=2)
    print(f"   ✓ Saved {len(cluster_data['clusters'])} clusters to {clusters_path}")

    # Cleanup
    clustering_engine.close()

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  - {docs_path}")
    print(f"  - {clusters_path}")
    print(f"\nNext step:")
    print(f"  Run retrieval experiments:")
    print(f"    python run_retrieval_experiments.py")
    print("=" * 60 + "\n")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Prepare data for Stage 5 retrieval experiments'
    )
    parser.add_argument(
        '--raw-dir',
        type=str,
        default='data/raw',
        help='Directory with raw documents'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed',
        help='Output directory for processed files'
    )
    parser.add_argument(
        '--clusters',
        type=int,
        default=10,
        help='Number of clusters to create'
    )

    args = parser.parse_args()

    success = prepare_data_from_raw(
        raw_dir=args.raw_dir,
        output_dir=args.output,
        n_clusters=args.clusters
    )

    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
