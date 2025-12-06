#!/usr/bin/env python3
# run_retrieval_experiments.py

"""
Stage 5: Query & Retrieval Experiments
Compares cluster-aware vs flat retrieval performance
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import argparse
from typing import List
import json

# Import components
from retrieval.query import QueryEmbedder
from retrieval.retrieval import ClusterAwareRetrieval, FlatRetrieval, RetrievalComparator
from retrieval.evaluation import RetrievalEvaluator
from retrieval.visualization import RetrievalVisualizer
from embedding.vector_store import VectorStore
from embedding.embedding_engine import BackgroundEmbeddingEngine


DEFAULT_TEST_QUERIES = [
    # HR & Talent (2)
    "talent acquisition specialist skilled in recruiting, interview scheduling, and offer negotiation for tech companies",
    # Banking & Finance (2)
    "digital payment systems architect with expertise in mobile wallets, online payment gateways, and transaction security",
    "financial analyst conducting portfolio analysis, investment risk scenarios, and quarterly reporting for institutional funds",
    # Healthcare (2)
    "medical administrator managing clinic operations, patient record systems, insurance claim processing, and compliance",
    "patient care coordinator overseeing multi-specialty appointments, insurance verification, and aftercare planning",
    # Art & Culture (2)
    "museum curator specializing in contemporary art exhibitions, artist liaison, and grant proposal writing",
    "digital art platform product manager responsible for user experience design, artist onboarding, and feature roadmapping",
    # Automotive (2)
    "automotive engineer developing electric powertrain systems, battery technologies, and vehicle safety features",
    "car designer skilled in CAD modeling, concept visualization, and aerodynamic optimization",
    # Food & Hospitality (2)
    "executive chef expert in French and Mediterranean cuisine, seasonal menu planning, and kitchen team leadership",
    "restaurant manager specializing in FOH operations, inventory control, and staff training programs",
    # Fitness & Wellness (2)
    "fitness coach creating custom training regimens, tracking client progress, and providing motivational support",
    "wellness program coordinator responsible for employee health screenings, mental wellness seminars, and activity challenges",
    # Sales & Marketing (2)
    "B2B technology sales executive using CRM tools for pipeline management, contract negotiation, and strategic account growth",
    "B2C product marketer specializing in lead generation campaigns, customer engagement, and social media analytics",
    # Software Engineering (2)
    "software engineer developing RESTful APIs, microservices architectures, and integration pipelines in Python and Go",
    "backend developer with experience in database design, distributed caching, and system scalability enhancements",
    # Cybersecurity (2)
    "cybersecurity analyst investigating intrusion attempts, configuring firewalls, and incident response planning",
    "cloud security architect designing IAM policies, encryption protocols, and continuous security monitoring solutions",
    # Environment (2)
    "environmental scientist modeling climate change impacts, conducting field sampling, and publishing peer-reviewed studies",
    "sustainability consultant advising clients on carbon reduction strategies, green certifications, and regulatory compliance",
    # Business & Analytics (2)
    "business process analyst streamlining workflow automation, requirements gathering, and system documentation",
    "data analytics lead developing BI dashboards, data pipelines, and predictive models for retail chains",
    # Logistics & Supply Chain (2)
    "logistics coordinator managing freight forwarding, customs documentation, and last-mile delivery optimization",
    "supply chain analyst monitoring inventory turnover rates, demand forecasting, and vendor rating systems",
]


def load_documents_and_clusters(
    docs_path: str = 'data/processed/documents.json',
    clusters_path: str = 'data/processed/clusters.json'
):
    """
    Load processed documents and cluster information

    Args:
        docs_path: Path to documents JSON
        clusters_path: Path to clusters JSON

    Returns:
        Tuple of (document_store, cluster_data)
    """
    print(f"Loading documents from {docs_path}...")
    with open(docs_path, 'r') as f:
        documents = json.load(f)

    print(f"Loading clusters from {clusters_path}...")
    with open(clusters_path, 'r') as f:
        cluster_data = json.load(f)

    print(f"Loaded {len(documents)} documents with {len(cluster_data.get('clusters', {}))} clusters")
    return documents, cluster_data


def initialize_retrieval_systems(documents: List, cluster_data: dict, tfidf_matrix_path: str = None):
    """
    Initialize all retrieval components

    Args:
        documents: List of document dictionaries
        cluster_data: Cluster information dictionary
        tfidf_matrix_path: Optional path to precomputed TF-IDF matrix

    Returns:
        Tuple of (query_embedder, cluster_aware_retrieval, flat_retrieval)
    """
    print("\nInitializing retrieval systems...")

    # Initialize query embedder
    print("  - Loading query embedding model...")
    query_embedder = QueryEmbedder()

    # Initialize vector store
    print("  - Creating FAISS vector store...")
    vector_store = VectorStore()

    # Generate document embeddings
    print("  - Generating document embeddings...")
    embedding_engine = BackgroundEmbeddingEngine()

    # Extract cluster assignments
    cluster_assignments = {}
    for cluster_id, cluster_info in cluster_data['clusters'].items():
        for doc_idx in cluster_info['document_indices']:
            cluster_assignments[doc_idx] = int(cluster_id)

    # Encode documents
    texts = [doc['text'] for doc in documents]
    embeddings_array = embedding_engine.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    # Create embeddings dictionary
    embeddings_dict = {i: emb for i, emb in enumerate(embeddings_array)}

    # Add to vector store
    print("  - Indexing embeddings in FAISS...")
    vector_store.add_embeddings(embeddings_dict, cluster_assignments)

    # Load cluster centroids (from TF-IDF clustering)
    # For simplicity, we'll compute centroids from embeddings
    print("  - Computing cluster centroids...")
    import numpy as np
    n_clusters = len(cluster_data['clusters'])
    embedding_dim = embeddings_array.shape[1]
    cluster_centroids = np.zeros((n_clusters, embedding_dim))

    for cluster_id, cluster_info in cluster_data['clusters'].items():
        doc_indices = cluster_info['document_indices']
        cluster_embeddings = embeddings_array[doc_indices]
        cluster_centroids[int(cluster_id)] = cluster_embeddings.mean(axis=0)

    # Initialize retrieval systems
    print("  - Creating cluster-aware retrieval...")
    cluster_aware_retrieval = ClusterAwareRetrieval(
        cluster_centroids=cluster_centroids,
        vector_store=vector_store,
        cluster_assignments=cluster_assignments
    )

    print("  - Creating flat retrieval baseline...")
    flat_retrieval = FlatRetrieval(
        vector_store=vector_store,
        total_documents=len(documents)
    )

    print("Initialization complete!\n")
    return query_embedder, cluster_aware_retrieval, flat_retrieval


def run_experiments(
    query_embedder,
    cluster_aware_retrieval,
    flat_retrieval,
    documents: List,
    test_queries: List[str],
    k: int = 5,
    num_runs: int = 10,
    output_dir: str = 'data/experiments'
):
    """
    Run retrieval experiments

    Args:
        query_embedder: QueryEmbedder instance
        cluster_aware_retrieval: ClusterAwareRetrieval instance
        flat_retrieval: FlatRetrieval instance
        documents: List of documents
        test_queries: List of test queries
        k: Number of results to retrieve
        num_runs: Number of runs per query
        output_dir: Output directory for results
    """
    print("="*60)
    print("RUNNING RETRIEVAL EXPERIMENTS")
    print("="*60)
    print(f"Corpus size: {len(documents)} documents")
    print(f"Test queries: {len(test_queries)}")
    print(f"Results per query (k): {k}")
    print(f"Runs per query: {num_runs}")
    print("="*60 + "\n")

    # Initialize evaluator
    evaluator = RetrievalEvaluator(
        query_embedder=query_embedder,
        cluster_aware_retrieval=cluster_aware_retrieval,
        flat_retrieval=flat_retrieval,
        document_store=documents
    )

    # Run experiments
    results = evaluator.run_batch_experiments(
        test_queries=test_queries,
        k=k,
        num_runs_per_query=num_runs
    )

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    evaluator.save_results(os.path.join(output_dir, 'experiment_results.json'))

    # Generate CSV summary
    results_df = evaluator.generate_summary_csv(os.path.join(output_dir, 'results_summary.csv'))

    # Print summary
    evaluator.print_summary()

    # Generate visualizations
    visualizer = RetrievalVisualizer(output_dir=os.path.join(output_dir, 'plots'))
    visualizer.generate_all_plots(results_df)

    print("\n" + "="*60)
    print("EXPERIMENTS COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_dir}/")
    print(f"  - experiment_results.json: Detailed results")
    print(f"  - results_summary.csv: Summary table")
    print(f"  - plots/: Performance visualizations")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run Stage 5 retrieval experiments comparing cluster-aware vs flat search'
    )
    parser.add_argument(
        '--docs',
        type=str,
        default='data/processed/documents.json',
        help='Path to documents JSON file'
    )
    parser.add_argument(
        '--clusters',
        type=str,
        default='data/processed/clusters.json',
        help='Path to clusters JSON file'
    )
    parser.add_argument(
        '--queries',
        type=str,
        nargs='+',
        help='Custom test queries (space-separated)'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=5,
        help='Number of results to retrieve per query'
    )
    parser.add_argument(
        '--runs',
        type=int,
        default=10,
        help='Number of runs per query for averaging'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/experiments',
        help='Output directory for results'
    )

    args = parser.parse_args()

    # Load data
    try:
        documents, cluster_data = load_documents_and_clusters(args.docs, args.clusters)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease run clustering first to generate the required files.")
        print("You can use the Flask API or run clustering directly.")
        sys.exit(1)

    # Initialize retrieval systems
    query_embedder, cluster_aware, flat = initialize_retrieval_systems(documents, cluster_data)

    # Determine test queries
    test_queries = args.queries if args.queries else DEFAULT_TEST_QUERIES

    # Run experiments
    run_experiments(
        query_embedder=query_embedder,
        cluster_aware_retrieval=cluster_aware,
        flat_retrieval=flat,
        documents=documents,
        test_queries=test_queries,
        k=args.k,
        num_runs=args.runs,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
