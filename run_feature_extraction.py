#!/usr/bin/env python3
# run_feature_extraction.py

"""
Stage 2: TF-IDF Feature Extraction

Reads parsed documents and computes TF-IDF feature vectors using Spark.
Outputs a TF-IDF matrix ready for clustering.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from analytics.tfidf_engine import SparkTFIDFEngine


def run_tfidf(
    input_file: str = 'data/processed/documents.json',
    output_dir: str = 'data/processed',
    num_features: int = 5000
):
    """
    Run TF-IDF feature extraction on parsed documents

    Args:
        input_file: Path to documents JSON file
        output_dir: Output directory for TF-IDF matrix
        num_features: Number of TF-IDF features (HashingTF size)

    Returns:
        True if successful
    """
    print("=" * 60)
    print("STAGE 2: TF-IDF FEATURE EXTRACTION")
    print("=" * 60)

    # Load documents
    print(f"\n1. Loading documents from {input_file}...")
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file not found: {input_file}")
        print("\nPlease run document parsing first:")
        print("  python run_parser.py")
        return False

    with open(input_file, 'r') as f:
        documents = json.load(f)

    print(f"   ✓ Loaded {len(documents)} documents")

    if len(documents) < 2:
        print("\n❌ Error: Need at least 2 documents for TF-IDF")
        return False

    # Prepare documents for TF-IDF
    print(f"\n2. Computing TF-IDF features (num_features={num_features})...")
    docs_for_tfidf = [
        {'id': doc['id'], 'text': doc['text']}
        for doc in documents
    ]

    # Run TF-IDF
    tfidf_engine = SparkTFIDFEngine(num_features=num_features)
    tfidf_matrix, vocabulary, tfidf_metrics = tfidf_engine.compute_tfidf(docs_for_tfidf)

    print(f"   ✓ TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"   ✓ Computation time: {tfidf_metrics['total_time_sec']:.2f}s")
    print(f"   ✓ Time per document: {tfidf_metrics['time_per_doc_sec']:.4f}s")

    # Close Spark session
    tfidf_engine.close()

    # Save TF-IDF matrix
    print(f"\n3. Saving TF-IDF matrix...")
    os.makedirs(output_dir, exist_ok=True)

    # Save matrix as .npz (compressed numpy format)
    matrix_path = os.path.join(output_dir, 'tfidf_matrix.npz')
    np.savez_compressed(matrix_path, matrix=tfidf_matrix)
    print(f"   ✓ Saved TF-IDF matrix to {matrix_path}")

    # Save metrics
    metrics_path = os.path.join(output_dir, 'tfidf_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(tfidf_metrics, f, indent=2)
    print(f"   ✓ Saved metrics to {metrics_path}")

    print("\n" + "=" * 60)
    print("STAGE 2 COMPLETE")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  - {matrix_path}")
    print(f"  - {metrics_path}")
    print(f"\nNext step:")
    print(f"  Run clustering:")
    print(f"    python run_clustering.py")
    print("=" * 60 + "\n")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Stage 2: TF-IDF Feature Extraction with Spark'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/processed/documents.json',
        help='Input documents JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed',
        help='Output directory for TF-IDF matrix'
    )
    parser.add_argument(
        '--features',
        type=int,
        default=5000,
        help='Number of TF-IDF features (HashingTF size)'
    )

    args = parser.parse_args()

    success = run_tfidf(
        input_file=args.input,
        output_dir=args.output,
        num_features=args.features
    )

    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
