# Complete Pipeline Guide

This guide shows how to run the entire distributed text analytics pipeline in 4 separate stages.

## Overview

```
Stage 1: Document Ingestion         → run_parser.py
Stage 2: TF-IDF Feature Extraction  → run_feature_extraction.py
Stage 3: K-Means Clustering         → run_clustering.py
Stage 4: Embedding Generation       → (runs in run_retrieval_experiments.py)
Stage 5: Query & Retrieval          → run_retrieval_experiments.py
```

---

## Stage-by-Stage Execution

### Stage 1: Document Ingestion
**Purpose**: Parse PDF/DOCX/TXT files and extract text

```bash
python run_parser.py
```

**Outputs**:
- `data/cleaned/corpus.jsonl` - JSONL format (legacy)
- `data/processed/documents.json` - Structured JSON for pipeline

**What it does**:
- Scans `data/raw/` for documents
- Extracts raw text from each file
- Applies simple text cleaning (lowercase, remove punctuation)
- Stores metadata (filename, word count, format)

---

### Stage 2: TF-IDF Feature Extraction
**Purpose**: Convert text documents into numerical feature vectors

```bash
python run_feature_extraction.py
```

**Options**:
```bash
python run_feature_extraction.py --features 5000    # Adjust feature count
```

**Outputs**:
- `data/processed/tfidf_matrix.npz` - TF-IDF feature matrix
- `data/processed/tfidf_metrics.json` - Performance metrics

**What it does**:
- Loads documents from Stage 1
- Uses Spark MLlib HashingTF + IDF
- Computes TF-IDF vectors (5000 features by default)
- Tracks computation time and throughput

---

### Stage 3: K-Means Clustering
**Purpose**: Group similar documents using distributed K-Means

```bash
python run_clustering.py
```

**Options**:
```bash
python run_clustering.py --clusters 10              # Number of clusters
python run_clustering.py --workers 4                # Spark workers
```

**Outputs**:
- `data/processed/clusters.json` - Cluster metadata and assignments
- `data/processed/cluster_assignments.npy` - Cluster assignments array

**What it does**:
- Loads TF-IDF matrix from Stage 2
- Runs distributed K-Means with Spark (10 clusters by default)
- Generates cluster labels from top TF-IDF terms
- Evaluates clustering quality (silhouette score)
- Tracks performance metrics (time, memory, iterations)

---

### Stage 4 & 5: Embedding Generation + Retrieval
**Purpose**: Generate embeddings and test retrieval performance

```bash
python run_retrieval_experiments.py
```

**Outputs**:
- `data/experiments/experiment_results.json` - Detailed results
- `data/experiments/results_summary.csv` - Summary table
- `data/experiments/plots/*.png` - Visualization plots

**What it does**:
- Loads documents and clusters from previous stages
- Generates semantic embeddings (Stage 4 - runs in background)
- Builds FAISS vector index
- Compares cluster-aware vs flat retrieval
- Measures query latency breakdown

---

## Special Experiments

### Spark Scalability Test (MOST IMPORTANT!)
**Purpose**: Measure speedup with different numbers of Spark workers

```bash
python test_spark_scalability.py
```

**What it tests**:
- Runs K-Means clustering with 1, 2, 4, 8 workers
- Measures execution time for each
- Calculates speedup and parallel efficiency
- Demonstrates distributed computing concepts

**Outputs**:
- `data/experiments/spark_scalability/results.json`
- `data/experiments/spark_scalability/results.csv`
- `data/experiments/spark_scalability/*.png` - Plots showing:
  - Execution time vs workers
  - Speedup curve (actual vs ideal)
  - Parallel efficiency
  - Component breakdown (TF-IDF vs K-Means)

---

## Complete Workflow

### Option 1: Run all stages sequentially
```bash
# Stage 1
python run_parser.py

# Stage 2
python run_feature_extraction.py

# Stage 3
python run_clustering.py

# Stages 4-5
python run_retrieval_experiments.py

# Spark scalability test
python test_spark_scalability.py
```

### Option 2: Run with custom parameters
```bash
# Stage 1: Parse documents
python run_parser.py

# Stage 2: Extract 10000 features
python run_feature_extraction.py --features 10000

# Stage 3: Create 15 clusters with 4 workers
python run_clustering.py --clusters 15 --workers 4

# Stages 4-5: Test retrieval
python run_retrieval_experiments.py
```

---

## Data Flow

```
data/raw/                      # Input: Raw PDF/DOCX/TXT files
    ↓
[Stage 1: run_parser.py]
    ↓
data/processed/documents.json  # Cleaned, structured documents
    ↓
[Stage 2: run_feature_extraction.py]
    ↓
data/processed/tfidf_matrix.npz  # TF-IDF feature vectors
    ↓
[Stage 3: run_clustering.py]
    ↓
data/processed/clusters.json   # Cluster assignments + metadata
    ↓
[Stages 4-5: run_retrieval_experiments.py]
    ↓
data/experiments/              # Retrieval performance results
```

---

## Performance Metrics Collected

### Stage 1: Document Ingestion
- Total ingestion time
- Average time per document
- Throughput (documents/second)
- Time breakdown by file type

### Stage 2: TF-IDF
- TF-IDF computation time
- Time per document
- Matrix shape and memory usage
- Vocabulary size impact

### Stage 3: K-Means Clustering
- Clustering time
- Number of iterations
- Memory usage
- Speedup with multiple workers ⭐
- Silhouette score (quality)

### Stage 4: Embeddings
- Embedding generation time
- Throughput (documents/second)
- GPU vs CPU performance

### Stage 5: Retrieval
- End-to-end query latency
- Cluster selection time
- FAISS search time
- Cluster-aware vs flat comparison ⭐
- Speedup from clustering

---

## Tips

1. **Start small**: Test with 10-50 documents first
2. **Check outputs**: Each stage creates files needed by the next
3. **Monitor Spark**: Watch for worker utilization in Stage 3
4. **Run scalability test**: Most important for demonstrating distributed systems concepts

## Troubleshooting

**Error: "Documents file not found"**
- Make sure you ran the previous stage first
- Check that output directory exists

**Spark errors on Windows**
- Ensure Java is installed
- Set `JAVA_HOME` environment variable

**Out of memory**
- Reduce number of features: `--features 2000`
- Reduce batch size for embeddings
