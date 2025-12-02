# src/api/visualization.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import os


class RetrievalVisualizer:
    """
    Generate visualizations for retrieval performance comparison
    """

    def __init__(self, output_dir: str = 'data/visualizations'):
        """
        Initialize visualizer

        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_latency_comparison(
        self,
        results_df: pd.DataFrame,
        output_path: Optional[str] = None
    ):
        """
        Plot latency comparison between cluster-aware and flat retrieval

        Args:
            results_df: DataFrame with experiment results
            output_path: Optional custom output path
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'latency_comparison.png')

        fig, ax = plt.subplots(figsize=(10, 6))

        queries = [f"Q{i+1}" for i in range(len(results_df))]
        x = np.arange(len(queries))
        width = 0.35

        cluster_latencies = results_df['cluster_aware_latency_ms'].values
        flat_latencies = results_df['flat_latency_ms'].values

        ax.bar(x - width/2, cluster_latencies, width, label='Cluster-Aware', color='#2ecc71')
        ax.bar(x + width/2, flat_latencies, width, label='Flat Search', color='#e74c3c')

        ax.set_xlabel('Query', fontsize=12)
        ax.set_ylabel('Latency (ms)', fontsize=12)
        ax.set_title('Query Latency: Cluster-Aware vs Flat Retrieval', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(queries)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Latency comparison plot saved to {output_path}")

    def plot_speedup_chart(
        self,
        results_df: pd.DataFrame,
        output_path: Optional[str] = None
    ):
        """
        Plot speedup factors for each query

        Args:
            results_df: DataFrame with experiment results
            output_path: Optional custom output path
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'speedup_chart.png')

        fig, ax = plt.subplots(figsize=(10, 6))

        queries = [f"Q{i+1}" for i in range(len(results_df))]
        speedups = results_df['speedup'].values

        colors = ['#27ae60' if s > 1 else '#e67e22' for s in speedups]
        bars = ax.bar(queries, speedups, color=colors, alpha=0.7)

        # Add horizontal line at y=1 (no speedup)
        ax.axhline(y=1, color='red', linestyle='--', linewidth=2, label='No speedup')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}x',
                   ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Query', fontsize=12)
        ax.set_ylabel('Speedup Factor', fontsize=12)
        ax.set_title('Cluster-Aware Retrieval Speedup vs Flat Search', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Speedup chart saved to {output_path}")

    def plot_latency_breakdown(
        self,
        avg_cluster_aware: Dict[str, float],
        avg_flat: Dict[str, float],
        output_path: Optional[str] = None
    ):
        """
        Plot stacked bar chart showing latency breakdown

        Args:
            avg_cluster_aware: Dict with timing components for cluster-aware
            avg_flat: Dict with timing components for flat
            output_path: Optional custom output path
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'latency_breakdown.png')

        fig, ax = plt.subplots(figsize=(10, 6))

        methods = ['Cluster-Aware', 'Flat Search']

        # Prepare data
        embedding_time = [
            avg_cluster_aware.get('embedding', 0),
            avg_flat.get('embedding', 0)
        ]
        cluster_selection = [
            avg_cluster_aware.get('cluster_selection', 0),
            0  # Flat doesn't have cluster selection
        ]
        search_time = [
            avg_cluster_aware.get('search', 0),
            avg_flat.get('search', 0)
        ]

        width = 0.5
        x = np.arange(len(methods))

        # Create stacked bars
        p1 = ax.bar(x, embedding_time, width, label='Query Embedding', color='#3498db')
        p2 = ax.bar(x, cluster_selection, width, bottom=embedding_time,
                   label='Cluster Selection', color='#9b59b6')
        p3 = ax.bar(x, search_time, width,
                   bottom=np.array(embedding_time) + np.array(cluster_selection),
                   label='FAISS Search', color='#e67e22')

        ax.set_ylabel('Time (ms)', fontsize=12)
        ax.set_title('Latency Breakdown by Component', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Latency breakdown plot saved to {output_path}")

    def plot_search_space_reduction(
        self,
        results_df: pd.DataFrame,
        output_path: Optional[str] = None
    ):
        """
        Plot search space reduction visualization

        Args:
            results_df: DataFrame with experiment results
            output_path: Optional custom output path
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'search_space_reduction.png')

        fig, ax = plt.subplots(figsize=(10, 6))

        queries = [f"Q{i+1}" for i in range(len(results_df))]
        cluster_docs = results_df['cluster_aware_docs_searched'].values
        flat_docs = results_df['flat_docs_searched'].values

        x = np.arange(len(queries))
        width = 0.35

        ax.bar(x - width/2, cluster_docs, width, label='Cluster-Aware', color='#16a085')
        ax.bar(x + width/2, flat_docs, width, label='Flat Search', color='#c0392b')

        # Add reduction percentage labels
        for i in range(len(queries)):
            reduction = (1 - cluster_docs[i] / flat_docs[i]) * 100 if flat_docs[i] > 0 else 0
            ax.text(i, max(cluster_docs[i], flat_docs[i]) + 5,
                   f'-{reduction:.0f}%',
                   ha='center', fontsize=9, fontweight='bold')

        ax.set_xlabel('Query', fontsize=12)
        ax.set_ylabel('Number of Documents Searched', fontsize=12)
        ax.set_title('Search Space Reduction: Cluster-Aware vs Flat', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(queries)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Search space reduction plot saved to {output_path}")

    def plot_accuracy_vs_speed_tradeoff(
        self,
        cluster_aware_latency: float,
        flat_latency: float,
        cluster_aware_precision: float,
        flat_precision: float,
        output_path: Optional[str] = None
    ):
        """
        Plot accuracy vs speed trade-off scatter plot

        Args:
            cluster_aware_latency: Average latency for cluster-aware
            flat_latency: Average latency for flat
            cluster_aware_precision: Average precision for cluster-aware
            flat_precision: Average precision for flat
            output_path: Optional custom output path
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'accuracy_speed_tradeoff.png')

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot points
        ax.scatter([cluster_aware_latency], [cluster_aware_precision * 100],
                  s=300, c='#2ecc71', marker='o', label='Cluster-Aware',
                  edgecolors='black', linewidths=2, zorder=5)
        ax.scatter([flat_latency], [flat_precision * 100],
                  s=300, c='#e74c3c', marker='s', label='Flat Search',
                  edgecolors='black', linewidths=2, zorder=5)

        # Add annotations
        ax.annotate('Faster, Good Accuracy',
                   xy=(cluster_aware_latency, cluster_aware_precision * 100),
                   xytext=(cluster_aware_latency - 20, cluster_aware_precision * 100 - 5),
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='#2ecc71', alpha=0.3))
        ax.annotate('Slower, Best Accuracy',
                   xy=(flat_latency, flat_precision * 100),
                   xytext=(flat_latency + 10, flat_precision * 100 + 3),
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='#e74c3c', alpha=0.3))

        ax.set_xlabel('Query Latency (ms)', fontsize=12)
        ax.set_ylabel('Precision (%)', fontsize=12)
        ax.set_title('Accuracy vs Speed Trade-off', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Accuracy vs speed trade-off plot saved to {output_path}")

    def generate_all_plots(self, results_df: pd.DataFrame):
        """
        Generate all standard plots

        Args:
            results_df: DataFrame with experiment results
        """
        print("\nGenerating visualizations...")

        self.plot_latency_comparison(results_df)
        self.plot_speedup_chart(results_df)
        self.plot_search_space_reduction(results_df)

        # Calculate average latency breakdown for breakdown plot
        avg_cluster_aware = {
            'embedding': results_df['embedding_time_ms'].mean(),
            'cluster_selection': 2.0,  # Approximate
            'search': results_df['cluster_aware_latency_ms'].mean() - results_df['embedding_time_ms'].mean() - 2.0
        }
        avg_flat = {
            'embedding': results_df['embedding_time_ms'].mean(),
            'search': results_df['flat_latency_ms'].mean() - results_df['embedding_time_ms'].mean()
        }
        self.plot_latency_breakdown(avg_cluster_aware, avg_flat)

        print(f"\nAll visualizations saved to {self.output_dir}/")
