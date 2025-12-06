# src/api/evaluation.py

import numpy as np
import pandas as pd
import time
from typing import List, Dict, Optional
import json
import os
from pathlib import Path


class RetrievalEvaluator:
    """
    Runs experiments comparing cluster-aware vs flat retrieval
    Collects performance metrics and generates analysis
    """

    def __init__(
        self,
        query_embedder,
        cluster_aware_retrieval,
        flat_retrieval,
        document_store: List[Dict]
    ):
        """
        Initialize evaluator

        Args:
            query_embedder: QueryEmbedder instance
            cluster_aware_retrieval: ClusterAwareRetrieval instance
            flat_retrieval: FlatRetrieval instance
            document_store: List of document dictionaries
        """
        self.query_embedder = query_embedder
        self.cluster_aware = cluster_aware_retrieval
        self.flat = flat_retrieval
        self.document_store = document_store
        self.results = []

    def run_single_experiment(
        self,
        query_text: str,
        k: int = 5,
        num_runs: int = 10
    ) -> Dict:
        """
        Run a single query multiple times and average results

        Args:
            query_text: Query text
            k: Number of results to retrieve
            num_runs: Number of times to run the query

        Returns:
            Dictionary with averaged metrics
        """
        cluster_times = []
        flat_times = []
        embedding_times = []
        cluster_selection_times = []
        cluster_faiss_times = []
        flat_faiss_times = []

        cluster_results_list = []
        flat_results_list = []

        for _ in range(num_runs):
            # Embed query
            query_embedding, emb_time = self.query_embedder.embed_query(query_text)
            embedding_times.append(emb_time * 1000)  # Convert to ms

            # Cluster-aware search
            cluster_result = self.cluster_aware.search(query_embedding, k)
            cluster_times.append(cluster_result['timing']['total_latency_ms'])
            cluster_selection_times.append(cluster_result['timing'].get('cluster_selection_ms', 0))
            cluster_faiss_times.append(cluster_result['timing'].get('faiss_search_ms', 0))
            cluster_results_list.append(cluster_result)

            # Flat search
            flat_result = self.flat.search(query_embedding, k)
            flat_times.append(flat_result['timing']['total_latency_ms'])
            flat_faiss_times.append(flat_result['timing'].get('faiss_search_ms', 0))
            flat_results_list.append(flat_result)

        # Calculate averages
        avg_cluster_time = np.mean(cluster_times)
        avg_flat_time = np.mean(flat_times)
        avg_embedding_time = np.mean(embedding_times)
        avg_cluster_selection_time = np.mean(cluster_selection_times)
        avg_cluster_faiss_time = np.mean(cluster_faiss_times)
        avg_flat_faiss_time = np.mean(flat_faiss_times)

        speedup = avg_flat_time / avg_cluster_time if avg_cluster_time > 0 else 0

        # Use first run for result details
        cluster_first = cluster_results_list[0]
        flat_first = flat_results_list[0]

        search_space_reduction = 1 - (
            cluster_first['timing']['documents_searched'] /
            flat_first['timing']['documents_searched']
        )

        return {
            'query': query_text,
            'k': k,
            'num_runs': num_runs,
            'corpus_size': len(self.document_store),
            'metrics': {
                'avg_embedding_time_ms': round(avg_embedding_time, 2),
                'cluster_aware': {
                    'avg_total_latency_ms': round(avg_cluster_time, 2),
                    'avg_cluster_selection_ms': round(avg_cluster_selection_time, 2),
                    'avg_faiss_search_ms': round(avg_cluster_faiss_time, 2),
                    'std_latency_ms': round(np.std(cluster_times), 2),
                    'min_latency_ms': round(np.min(cluster_times), 2),
                    'max_latency_ms': round(np.max(cluster_times), 2),
                    'documents_searched': cluster_first['timing']['documents_searched']
                },
                'flat': {
                    'avg_total_latency_ms': round(avg_flat_time, 2),
                    'avg_faiss_search_ms': round(avg_flat_faiss_time, 2),
                    'std_latency_ms': round(np.std(flat_times), 2),
                    'min_latency_ms': round(np.min(flat_times), 2),
                    'max_latency_ms': round(np.max(flat_times), 2),
                    'documents_searched': flat_first['timing']['documents_searched']
                },
                'comparison': {
                    'speedup': round(speedup, 2),
                    'latency_difference_ms': round(avg_flat_time - avg_cluster_time, 2),
                    'search_space_reduction_pct': round(search_space_reduction * 100, 1)
                }
            },
            'cluster_aware_results': [r['results'] for r in cluster_results_list[:1]],
            'flat_results': [r['results'] for r in flat_results_list[:1]]
        }

    def run_batch_experiments(
        self,
        test_queries: List[str],
        k: int = 5,
        num_runs_per_query: int = 10
    ) -> List[Dict]:
        """
        Run experiments on multiple queries

        Args:
            test_queries: List of query strings
            k: Number of results per query
            num_runs_per_query: Number of runs per query

        Returns:
            List of experiment results
        """
        results = []

        for i, query in enumerate(test_queries):
            print(f"Running experiment {i+1}/{len(test_queries)}: '{query[:50]}...'")
            result = self.run_single_experiment(query, k, num_runs_per_query)
            results.append(result)
            self.results.append(result)

        return results

    def save_results(self, output_path: str):
        """
        Save experiment results to JSON file

        Args:
            output_path: Path to output file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {output_path}")

    def generate_summary_csv(self, output_path: str):
        """
        Generate CSV summary of all experiments

        Args:
            output_path: Path to output CSV file
        """
        rows = []

        for result in self.results:
            rows.append({
                'query': result['query'],
                'corpus_size': result['corpus_size'],
                'k': result['k'],
                'num_runs': result['num_runs'],
                'embedding_time_ms': result['metrics']['avg_embedding_time_ms'],
                'cluster_aware_latency_ms': result['metrics']['cluster_aware']['avg_total_latency_ms'],
                'cluster_selection_ms': result['metrics']['cluster_aware']['avg_cluster_selection_ms'],
                'cluster_faiss_search_ms': result['metrics']['cluster_aware']['avg_faiss_search_ms'],
                'cluster_aware_docs_searched': result['metrics']['cluster_aware']['documents_searched'],
                'flat_latency_ms': result['metrics']['flat']['avg_total_latency_ms'],
                'flat_faiss_search_ms': result['metrics']['flat']['avg_faiss_search_ms'],
                'flat_docs_searched': result['metrics']['flat']['documents_searched'],
                'speedup': result['metrics']['comparison']['speedup'],
                'latency_diff_ms': result['metrics']['comparison']['latency_difference_ms'],
                'search_space_reduction_pct': result['metrics']['comparison']['search_space_reduction_pct']
            })

        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"CSV summary saved to {output_path}")
        return df

    def print_summary(self):
        """Print summary statistics"""
        if not self.results:
            print("No results to summarize")
            return

        speedups = [r['metrics']['comparison']['speedup'] for r in self.results]
        cluster_latencies = [r['metrics']['cluster_aware']['avg_total_latency_ms'] for r in self.results]
        flat_latencies = [r['metrics']['flat']['avg_total_latency_ms'] for r in self.results]
        search_reductions = [r['metrics']['comparison']['search_space_reduction_pct'] for r in self.results]

        print("\n" + "="*60)
        print("RETRIEVAL PERFORMANCE EVALUATION SUMMARY")
        print("="*60)
        print(f"Total queries tested: {len(self.results)}")
        print(f"Corpus size: {self.results[0]['corpus_size']} documents")
        print()
        print("CLUSTER-AWARE RETRIEVAL:")
        print(f"  Average latency: {np.mean(cluster_latencies):.2f} ms")
        print(f"  Std deviation: {np.std(cluster_latencies):.2f} ms")
        print()
        print("FLAT RETRIEVAL:")
        print(f"  Average latency: {np.mean(flat_latencies):.2f} ms")
        print(f"  Std deviation: {np.std(flat_latencies):.2f} ms")
        print()
        print("PERFORMANCE COMPARISON:")
        print(f"  Average speedup: {np.mean(speedups):.2f}x")
        print(f"  Min speedup: {np.min(speedups):.2f}x")
        print(f"  Max speedup: {np.max(speedups):.2f}x")
        print(f"  Average search space reduction: {np.mean(search_reductions):.1f}%")
        print()
        print("CONCLUSION:")
        if np.mean(speedups) > 1.5:
            print(f"  Cluster-aware retrieval provides significant speedup ({np.mean(speedups):.2f}x)")
            print(f"  with {np.mean(search_reductions):.1f}% reduction in search space.")
        else:
            print(f"  Modest speedup ({np.mean(speedups):.2f}x) observed.")
        print("="*60 + "\n")


class AccuracyChecker:
    """
    Simple accuracy sanity check for retrieval results
    Manually verify a few queries return reasonable results
    """

    def __init__(self, document_store: List[Dict]):
        """
        Initialize accuracy checker

        Args:
            document_store: List of document dictionaries
        """
        self.document_store = document_store
        self.manual_labels = {}

    def add_relevant_docs(self, query: str, relevant_doc_ids: List[int]):
        """
        Manually label relevant documents for a query

        Args:
            query: Query text
            relevant_doc_ids: List of relevant document IDs
        """
        self.manual_labels[query] = set(relevant_doc_ids)

    def check_precision(
        self,
        query: str,
        retrieved_doc_ids: List[int],
        k: int = 5
    ) -> Dict:
        """
        Calculate precision for a query

        Args:
            query: Query text
            retrieved_doc_ids: Retrieved document IDs
            k: Number of results

        Returns:
            Dictionary with precision metrics
        """
        if query not in self.manual_labels:
            return {'error': 'No manual labels for this query'}

        relevant_set = self.manual_labels[query]
        retrieved_set = set(retrieved_doc_ids[:k])

        relevant_retrieved = relevant_set.intersection(retrieved_set)
        precision = len(relevant_retrieved) / k if k > 0 else 0

        return {
            'query': query,
            'k': k,
            'total_relevant': len(relevant_set),
            'retrieved': k,
            'relevant_retrieved': len(relevant_retrieved),
            'precision': round(precision, 3),
            'relevant_doc_ids': list(relevant_retrieved)
        }

    def compare_methods_accuracy(
        self,
        query: str,
        cluster_aware_results: List[int],
        flat_results: List[int],
        k: int = 5
    ) -> Dict:
        """
        Compare accuracy of both methods

        Args:
            query: Query text
            cluster_aware_results: Doc IDs from cluster-aware
            flat_results: Doc IDs from flat search
            k: Number of results

        Returns:
            Comparison dictionary
        """
        cluster_metrics = self.check_precision(query, cluster_aware_results, k)
        flat_metrics = self.check_precision(query, flat_results, k)

        return {
            'query': query,
            'cluster_aware_precision': cluster_metrics.get('precision', 0),
            'flat_precision': flat_metrics.get('precision', 0),
            'precision_difference': round(
                flat_metrics.get('precision', 0) - cluster_metrics.get('precision', 0),
                3
            )
        }
