# Distributed Text Analytics - Usage Guide

## Project Overview

This is a **distributed text analytics system** for CS532 that implements:
1. Document ingestion and preprocessing
2. TF-IDF feature extraction
3. Distributed K-means clustering (via PySpark)
4. **Stage 5: Cluster-aware retrieval** with performance evaluation

**No web UI or API** - Everything runs as CLI scripts.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Option A: If you have raw documents in `data/raw/`:
```bash
python prepare_experiment_data.py
```

Option B: If you already have processed data:
- Make sure `data/processed/documents.json` exists
- Make sure `data/processed/clusters.json` exists

### 3. Run Retrieval Experiments (Stage 5)

```bash
python run_retrieval_experiments.py
```

This will:
- Load documents and clusters
- Test 10 queries with both cluster-aware and flat retrieval
- Generate performance metrics
- Create visualization plots
- Save results to `data/experiments/`

---

## Project Structure

```
distributed-text-analytics/
├── src/
│   ├── analytics/          # TF-IDF, n-grams
│   ├── clustering/         # K-means clustering
│   ├── embedding/          # Vector store (FAISS)
│   ├── ingestion/          # Document parsing, cleaning
│   └── retrieval/          # Stage 5: Query & retrieval
│       ├── query.py
│       ├── retrieval.py
│       ├── evaluation.py
│       └── visualization.py
├── tests/
│   └── test_retrieval.py
├── data/
│   ├── raw/                # Input documents
│   ├── processed/          # documents.json, clusters.json
│   └── experiments/        # Experiment results
├── prepare_experiment_data.py   # Data preparation script
└── run_retrieval_experiments.py # Main experiment runner
```

---

## CLI Scripts

### prepare_experiment_data.py

Processes raw documents and creates clustering data.

```bash
# Basic usage
python prepare_experiment_data.py

# Custom options
python prepare_experiment_data.py \
  --raw-dir data/my_docs \
  --output data/processed \
  --clusters 15
```

**Output:**
- `data/processed/documents.json`
- `data/processed/clusters.json`

---

### run_retrieval_experiments.py

Runs Stage 5 retrieval performance experiments.

```bash
# Basic usage (uses 10 default queries)
python run_retrieval_experiments.py

# Custom queries
python run_retrieval_experiments.py \
  --queries "neural networks" "data mining" "distributed systems"

# Custom data paths
python run_retrieval_experiments.py \
  --docs data/my_documents.json \
  --clusters data/my_clusters.json

# More runs for better statistics
python run_retrieval_experiments.py --runs 20

# More results per query
python run_retrieval_experiments.py --k 10
```

**Output:**
- `data/experiments/experiment_results.json` - Detailed results
- `data/experiments/results_summary.csv` - Summary table
- `data/experiments/plots/` - Performance visualizations

---

## Stage 5: Retrieval System

### What It Does

Compares two retrieval approaches:

1. **Cluster-Aware Retrieval** (FAST)
   - Finds most relevant cluster
   - Searches only within that cluster
   - ~10% of documents searched

2. **Flat Retrieval** (BASELINE)
   - Searches all documents
   - 100% of documents searched

### Performance Metrics

- **Query Latency**: Total time per query
- **Speedup**: How much faster cluster-aware is
- **Search Space Reduction**: % of documents not searched
- **Scalability**: Performance vs corpus size

### Expected Results

With 1000 documents, 10 clusters:
- Cluster-aware: ~30ms
- Flat search: ~235ms
- **Speedup: ~8x**
- **Search reduction: ~90%**

---

## Example Workflow

```bash
# 1. Prepare your data
python prepare_experiment_data.py --raw-dir data/raw

# 2. Run experiments
python run_retrieval_experiments.py

# 3. View results
cat data/experiments/results_summary.csv
open data/experiments/plots/speedup_chart.png

# 4. Run tests
python -m pytest tests/test_retrieval.py -v
```

---

## Key Features

✅ **No server needed** - All CLI scripts
✅ **Distributed clustering** - PySpark K-means
✅ **Fast retrieval** - FAISS vector search
✅ **Performance evaluation** - Comprehensive metrics
✅ **Visualization** - Auto-generated charts
✅ **Reproducible** - Run multiple times, average results

---

## Dependencies

Core:
- `pyspark` - Distributed computing
- `sentence-transformers` - Semantic embeddings
- `faiss-cpu` - Fast similarity search
- `scikit-learn` - ML utilities
- `pandas` - Data analysis
- `matplotlib` - Visualizations

Document parsing:
- `beautifulsoup4` - HTML parsing
- `python-docx` - DOCX parsing
- `pdfminer.six` - PDF parsing

---

## Troubleshooting

### "FileNotFoundError: documents.json"
Run `prepare_experiment_data.py` first.

### "Need at least 2 documents for clustering"
Add more documents to `data/raw/`.

### Slow first run
Model loading takes 15-30 seconds initially. Subsequent queries are fast.

### Low speedup
Speedup increases with corpus size. Try more documents (500+).

---

## For Your Report

Use the generated visualizations:
1. **Latency Comparison** - Shows cluster-aware is faster
2. **Speedup Chart** - Shows speedup factors per query
3. **Search Space Reduction** - Shows why it's faster

Include metrics from `results_summary.csv`:
- Average speedup (e.g., 8.27x)
- Search space reduction (e.g., 89.8%)
- Latency comparison (e.g., 28ms vs 235ms)

---

## Testing

```bash
# Run unit tests
python -m pytest tests/test_retrieval.py -v

# Run specific test
python -m pytest tests/test_retrieval.py::TestQueryEmbedder -v
```

---

## Documentation

- **STAGE5_QUICKSTART.md** - Quick reference
- **STAGE5_RETRIEVAL.md** - Full documentation
- **REFACTORING_SUMMARY.md** - What changed (removed Flask/UI)
- **USAGE.md** - This file

---

## Summary

This project demonstrates distributed text analytics with:
- Document processing pipeline
- Distributed clustering (PySpark)
- Semantic search (embeddings + FAISS)
- **Performance-optimized retrieval** (cluster-aware vs flat)
- Quantitative evaluation (speedup, latency, search reduction)

**No Flask, no API, no UI** - Simple CLI tools for a distributed computing project! 🚀
