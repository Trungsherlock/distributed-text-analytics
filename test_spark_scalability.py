#!/usr/bin/env python3
# test_spark_scalability.py

"""
Spark Scalability Experiment

Tests K-Means clustering performance with different numbers of Spark workers (1, 2, 4, 8).
Measures speedup and demonstrates distributed computing concepts.

This is the MOST IMPORTANT experiment for the grader according to the project strategy:
"Speedup with 1, 2, 4, 8 Spark workers (most important!)"
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import json
import time
import numpy as np
import pandas as pd
from typing import List, Dict
import matplotlib.pyplot as plt
from pathlib import Path

from analytics.tfidf_engine import SparkTFIDFEngine
from clustering.kmeans_cluster import SparkKMeansClustering


def load_documents(docs_path: str = 'data/processed/documents.json') -> List[Dict]:
    """Load processed documents"""
    print(f"Loading documents from {docs_path}...")
    with open(docs_path, 'r') as f:
        documents = json.load(f)
    print(f"Loaded {len(documents)} documents\n")
    return documents


def run_clustering_experiment(
    documents: List[Dict],
    num_workers: int,
    n_clusters: int = 10,
    num_features: int = 5000
) -> Dict:
    """
    Run clustering with specified number of workers

    Args:
        documents: List of document dictionaries
        num_workers: Number of Spark workers
        n_clusters: Number of clusters
        num_features: TF-IDF feature dimensions

    Returns:
        Performance metrics dictionary
    """
    print(f"\n{'='*60}")
    print(f"Running with {num_workers} worker(s)")
    print(f"{'='*60}")

    # Prepare documents for TF-IDF
    doc_data = [
        {'id': i, 'text': doc['text']}
        for i, doc in enumerate(documents)
    ]

    # TF-IDF computation
    print(f"  [1/2] Computing TF-IDF...")
    tfidf_start = time.time()
    tfidf_engine = SparkTFIDFEngine(num_features=num_features)
    tfidf_matrix, _, tfidf_metrics = tfidf_engine.compute_tfidf(doc_data)
    tfidf_time = time.time() - tfidf_start
    tfidf_engine.close()

    print(f"        TF-IDF: {tfidf_time:.2f}s")

    # K-Means clustering
    print(f"  [2/2] Running K-Means clustering...")
    clustering_start = time.time()
    clusterer = SparkKMeansClustering(
        n_clusters=n_clusters,
        max_iter=100,
        seed=42,
        num_workers=num_workers
    )
    cluster_assignments, cluster_metrics = clusterer.fit_predict(tfidf_matrix)
    clustering_time = time.time() - clustering_start
    clusterer.close()

    print(f"        K-Means: {clustering_time:.2f}s")

    # Total time
    total_time = tfidf_time + clustering_time

    # Compile results
    results = {
        'num_workers': num_workers,
        'num_documents': len(documents),
        'num_clusters': n_clusters,
        'num_features': num_features,
        'tfidf_time_sec': round(tfidf_time, 3),
        'clustering_time_sec': round(clustering_time, 3),
        'total_time_sec': round(total_time, 3),
        'iterations': cluster_metrics.get('num_iterations', 0),
        'memory_mb': cluster_metrics.get('memory_usage_mb', 0),
        'num_partitions': cluster_metrics.get('num_partitions', 0)
    }

    print(f"\n  Results:")
    print(f"    Total time: {total_time:.2f}s")
    print(f"    Iterations: {results['iterations']}")
    print(f"    Memory: {results['memory_mb']:.1f} MB")

    return results


def calculate_speedup(results: List[Dict]) -> pd.DataFrame:
    """
    Calculate speedup metrics relative to single worker baseline

    Args:
        results: List of experiment results

    Returns:
        DataFrame with speedup analysis
    """
    df = pd.DataFrame(results)

    # Baseline (1 worker)
    baseline_time = df[df['num_workers'] == 1]['total_time_sec'].values[0]

    # Calculate speedup
    df['speedup'] = baseline_time / df['total_time_sec']
    df['efficiency'] = df['speedup'] / df['num_workers']
    df['time_reduction_pct'] = (1 - df['total_time_sec'] / baseline_time) * 100

    return df


def generate_visualizations(df: pd.DataFrame, output_dir: str):
    """
    Generate visualization plots

    Args:
        df: Results DataFrame
        output_dir: Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Execution time vs workers
    plt.figure(figsize=(10, 6))
    plt.plot(df['num_workers'], df['total_time_sec'], 'o-', linewidth=2, markersize=8)
    plt.xlabel('Number of Workers', fontsize=12)
    plt.ylabel('Total Time (seconds)', fontsize=12)
    plt.title('Execution Time vs Number of Workers', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(df['num_workers'])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_vs_workers.png'), dpi=300)
    print(f"  Saved: {output_dir}/time_vs_workers.png")
    plt.close()

    # 2. Speedup vs workers
    plt.figure(figsize=(10, 6))
    plt.plot(df['num_workers'], df['speedup'], 'o-', linewidth=2, markersize=8, label='Actual Speedup')
    plt.plot(df['num_workers'], df['num_workers'], '--', linewidth=2, alpha=0.5, label='Linear Speedup (Ideal)')
    plt.xlabel('Number of Workers', fontsize=12)
    plt.ylabel('Speedup', fontsize=12)
    plt.title('Speedup vs Number of Workers', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(df['num_workers'])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speedup_vs_workers.png'), dpi=300)
    print(f"  Saved: {output_dir}/speedup_vs_workers.png")
    plt.close()

    # 3. Parallel efficiency
    plt.figure(figsize=(10, 6))
    plt.plot(df['num_workers'], df['efficiency'] * 100, 'o-', linewidth=2, markersize=8, color='green')
    plt.axhline(y=100, color='red', linestyle='--', linewidth=2, alpha=0.5, label='100% Efficiency (Ideal)')
    plt.xlabel('Number of Workers', fontsize=12)
    plt.ylabel('Parallel Efficiency (%)', fontsize=12)
    plt.title('Parallel Efficiency vs Number of Workers', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(df['num_workers'])
    plt.ylim(0, 110)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'efficiency_vs_workers.png'), dpi=300)
    print(f"  Saved: {output_dir}/efficiency_vs_workers.png")
    plt.close()

    # 4. Component breakdown (stacked bar)
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.5
    x = np.arange(len(df))

    ax.bar(x, df['tfidf_time_sec'], width, label='TF-IDF', alpha=0.8)
    ax.bar(x, df['clustering_time_sec'], width, bottom=df['tfidf_time_sec'], label='K-Means', alpha=0.8)

    ax.set_xlabel('Number of Workers', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Time Breakdown by Component', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['num_workers'])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'component_breakdown.png'), dpi=300)
    print(f"  Saved: {output_dir}/component_breakdown.png")
    plt.close()


def print_summary(df: pd.DataFrame):
    """Print summary table"""
    print(f"\n{'='*80}")
    print("SPARK SCALABILITY EXPERIMENT SUMMARY")
    print(f"{'='*80}\n")

    print("Results Table:")
    print("-" * 80)
    print(f"{'Workers':<10} {'Total Time':<15} {'Speedup':<15} {'Efficiency':<15} {'Time Saved':<15}")
    print("-" * 80)

    for _, row in df.iterrows():
        print(f"{int(row['num_workers']):<10} "
              f"{row['total_time_sec']:.2f}s{'':<10} "
              f"{row['speedup']:.2f}x{'':<11} "
              f"{row['efficiency']*100:.1f}%{'':<11} "
              f"{row['time_reduction_pct']:.1f}%")

    print("-" * 80)

    # Analysis
    print(f"\nKey Findings:")
    max_speedup = df['speedup'].max()
    max_speedup_workers = df.loc[df['speedup'].idxmax(), 'num_workers']
    print(f"  - Maximum speedup: {max_speedup:.2f}x with {int(max_speedup_workers)} worker(s)")

    best_efficiency = df['efficiency'].max()
    print(f"  - Best efficiency: {best_efficiency*100:.1f}%")

    baseline = df[df['num_workers'] == 1]['total_time_sec'].values[0]
    max_workers_time = df[df['num_workers'] == df['num_workers'].max()]['total_time_sec'].values[0]
    total_reduction = (1 - max_workers_time / baseline) * 100
    print(f"  - Total time reduction (1 → {int(df['num_workers'].max())} workers): {total_reduction:.1f}%")

    # Overhead analysis
    ideal_speedup_8 = 8
    actual_speedup_8 = df[df['num_workers'] == 8]['speedup'].values[0] if 8 in df['num_workers'].values else 0
    if actual_speedup_8 > 0:
        overhead = (ideal_speedup_8 - actual_speedup_8) / ideal_speedup_8 * 100
        print(f"  - Parallelism overhead (8 workers): {overhead:.1f}%")
        print(f"    (Demonstrates coordination costs in distributed computing)")

    print(f"\n{'='*80}\n")


def main():
    """Main experiment runner"""
    print("="*80)
    print("SPARK SCALABILITY EXPERIMENT")
    print("Testing K-Means clustering with 1, 2, 4, 8 Spark workers")
    print("="*80)

    # Configuration
    worker_counts = [1, 2, 4, 8]
    n_clusters = 10
    num_features = 5000
    output_dir = 'data/experiments/spark_scalability'

    # Load documents
    try:
        documents = load_documents()
    except FileNotFoundError:
        print("Error: documents.json not found. Please run document processing first.")
        sys.exit(1)

    # Run experiments
    all_results = []
    for num_workers in worker_counts:
        try:
            results = run_clustering_experiment(
                documents=documents,
                num_workers=num_workers,
                n_clusters=n_clusters,
                num_features=num_features
            )
            all_results.append(results)
        except Exception as e:
            print(f"\nError with {num_workers} worker(s): {e}")
            continue

    if not all_results:
        print("\nNo experiments completed successfully!")
        sys.exit(1)

    # Analyze results
    df = calculate_speedup(all_results)

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    # Save JSON
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to: {output_dir}/results.json")

    # Save CSV
    df.to_csv(os.path.join(output_dir, 'results.csv'), index=False)
    print(f"Saved results to: {output_dir}/results.csv")

    # Generate visualizations
    print(f"\nGenerating visualizations...")
    generate_visualizations(df, output_dir)

    # Print summary
    print_summary(df)

    print(f"All results saved to: {output_dir}/")
    print(f"  - results.json: Raw experiment data")
    print(f"  - results.csv: Results table with speedup analysis")
    print(f"  - *.png: Performance visualization plots")


if __name__ == '__main__':
    main()
