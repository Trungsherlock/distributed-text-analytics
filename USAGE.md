# Distributed Text Analytics - Usage Guide

## Project Overview

This is a **distributed text analytics system** for CS532 that implements a complete document processing and retrieval pipeline:

1. **Stage 1: Document Ingestion** - Parse PDF, DOCX, TXT files with text cleaning
2. **Stage 2: TF-IDF Feature Extraction** - Distributed computation via PySpark
3. **Stage 3: K-Means Clustering** - Distributed clustering with PySpark MLlib
4. **Stage 4: Cluster-Aware Retrieval** - Fast semantic search with performance evaluation

**No web UI or API** - Everything runs as CLI scripts for distributed computing focus.

---

## Complete Setup & Workflow

### Step 0: Setup Virtual Environment (IMPORTANT!)

Modern Linux systems require using a virtual environment to avoid conflicts with system packages.

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify activation (should show path to venv)
which python
```

**Important:** Always activate the virtual environment before running any scripts:
```bash
source venv/bin/activate
```

Your prompt should show `(venv)` at the beginning when activated.

To deactivate when done:
```bash
deactivate
```

---

### Step 1: Install Dependencies

With your virtual environment activated, install all required Python packages:

```bash
pip install -r requirements.txt
```

**If you get "externally-managed-environment" error:**
- Make sure you activated the virtual environment first: `source venv/bin/activate`
- Your prompt should show `(venv)` at the beginning

**Required packages include:**
- `pyspark>=3.4.0` - Distributed computing framework
- `sentence-transformers>=2.2.0` - Semantic embeddings for retrieval
- `faiss-cpu>=1.7.0` - Fast similarity search
- `torch>=2.0.0` - PyTorch for embeddings
- `scikit-learn`, `numpy`, `pandas` - Data processing
- `nltk>=3.8` - Text preprocessing
- `kagglehub>=0.2.0` - Kaggle dataset download
- Document parsers: `python-docx`, `pdfminer.six`

**Note:** First run will download models automatically (~400MB for sentence-transformers).

---

### Step 2: Download Dataset (Optional)

If you need sample data, download from Kaggle:

```bash
python download_data.py
```

**What it does:**
- Downloads the "resume-dataset" from Kaggle using kagglehub
- Extracts PDF, DOCX, TXT, JSON files
- Saves to `data/raw/resumes/`

**Requirements:**
- Kaggle API credentials in `~/.kaggle/kaggle.json`
- Get API key from https://www.kaggle.com/settings

**Manual alternative:**
- Place your own documents in `data/raw/` directory
- Supports: `.pdf`, `.docx`, `.txt` files
- Minimum: 10+ documents recommended for clustering

---

### Step 3: Stage 1 - Document Ingestion

Parse and clean raw documents:

```bash
python run_parser.py
```

**What it does:**
- Scans `data/raw/` directory recursively
- Parses PDF, DOCX, TXT files
- Performs simple text cleaning
- Creates structured JSON output
- Outputs both JSONL (backward compatibility) and JSON formats

**Output files:**
- `data/cleaned/corpus.jsonl` - JSONL format
- `data/processed/documents.json` - Structured JSON for pipeline

**Verification:**
```bash
# Count processed documents
python -c "import json; print(f'Documents: {len(json.load(open(\"data/processed/documents.json\")))}')"

# View first document
head -n 20 data/processed/documents.json
```

---

### Step 4: Stage 2 - TF-IDF Feature Extraction

Compute TF-IDF features using distributed Spark:

```bash
python run_feature_extraction.py
```

**What it does:**
- Loads parsed documents from `data/processed/documents.json`
- Computes TF-IDF features using PySpark
- Uses HashingTF (5000 features default) for efficient distributed computation
- Tracks performance metrics (computation time, time per document)
- Saves compressed feature matrix

**Custom options:**
```bash
# Custom input/output
python run_feature_extraction.py \
  --input data/processed/documents.json \
  --output data/processed

# Change number of features
python run_feature_extraction.py --features 10000
```

**Output files:**
- `data/processed/tfidf_matrix.npz` - Compressed TF-IDF matrix (numpy format)
- `data/processed/tfidf_metrics.json` - Performance metrics

**Verification:**
```bash
# Check matrix shape
python -c "import numpy as np; m=np.load('data/processed/tfidf_matrix.npz')['matrix']; print(f'TF-IDF shape: {m.shape}')"

# View metrics
cat data/processed/tfidf_metrics.json
```

---

### Step 5: Stage 3 - K-Means Clustering

Perform distributed K-Means clustering with Spark:

```bash
python run_clustering.py
```

**What it does:**
- Loads TF-IDF matrix and documents
- Runs distributed K-Means using PySpark MLlib
- Evaluates clustering quality (Silhouette score)
- Generates cluster metadata and statistics
- Tracks performance metrics:
  - Clustering time
  - Number of iterations
  - Number of Spark workers
  - Memory usage

**Custom options:**
```bash
# Change number of clusters
python run_clustering.py --clusters 15

# Specify Spark workers
python run_clustering.py --workers 4

# Custom input/output paths
python run_clustering.py \
  --documents data/processed/documents.json \
  --tfidf data/processed/tfidf_matrix.npz \
  --output data/processed
```

**Output files:**
- `data/processed/clusters.json` - Cluster assignments with metadata
- `data/processed/cluster_assignments.npy` - Simple numpy array of assignments

**Verification:**
```bash
# View cluster summary
python -c "import json; c=json.load(open('data/processed/clusters.json')); print(f'Clusters: {c[\"num_clusters\"]}, Silhouette: {c[\"silhouette_score\"]:.3f}')"

# Check cluster distribution
python -c "import json; c=json.load(open('data/processed/clusters.json')); print('\\n'.join([f'Cluster {k}: {v[\"document_count\"]} docs ({v[\"percentage\"]:.1f}%)' for k,v in c['clusters'].items()]))"
```

---

### Step 6: Stage 4 - Retrieval Experiments

Execute performance evaluation comparing cluster-aware vs flat retrieval:

```bash
python run_retrieval_experiments.py
```

**What it does:**
- Loads documents and cluster assignments
- Initializes FAISS vector indices for fast search
- Generates document embeddings using sentence-transformers
- Tests 10 predefined queries with both methods:
  1. **Cluster-Aware Retrieval** - Searches only relevant cluster (~10% of docs)
  2. **Flat Retrieval** - Searches all documents (baseline)
- Measures latency, speedup, search space reduction
- Generates visualizations and CSV reports

**Default test queries:**
- "machine learning algorithms and neural networks"
- "data processing and analytics pipelines"
- "distributed computing systems"
- "natural language processing techniques"
- "information retrieval and search"
- And 5 more...

**Custom options:**
```bash
# Test custom queries
python run_retrieval_experiments.py \
  --queries "machine learning" "data science" "neural networks"

# More runs for statistical accuracy
python run_retrieval_experiments.py --runs 20

# Return more results per query
python run_retrieval_experiments.py --k 10
```

**Output files:**
- `data/experiments/experiment_results.json` - Detailed per-query results
- `data/experiments/results_summary.csv` - Aggregated metrics table
- `data/experiments/plots/latency_comparison.png` - Bar chart
- `data/experiments/plots/speedup_chart.png` - Speedup factors
- `data/experiments/plots/search_space_reduction.png` - Efficiency gains

**Verification:**
```bash
cat data/experiments/results_summary.csv  # View summary
ls -lh data/experiments/plots/  # Check visualizations
```

---

## Complete End-to-End Example

Here's a complete workflow from scratch:

```bash
# 0. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 1. Install dependencies
pip install -r requirements.txt

# 2. Download sample dataset (optional)
python download_data.py

# 3. Stage 1: Parse documents
python run_parser.py

# 4. Stage 2: Extract TF-IDF features
python run_feature_extraction.py

# 5. Stage 3: Cluster documents
python run_clustering.py --clusters 10

# 6. Stage 4: Run retrieval experiments
python run_retrieval_experiments.py

# 7. View results
cat data/experiments/results_summary.csv
xdg-open data/experiments/plots/speedup_chart.png  # Linux
# or: open data/experiments/plots/speedup_chart.png  # macOS

# 8. Run tests (optional)
python -m pytest tests/ -v
```

**Expected output locations:**
```
data/
├── raw/                        # Your input documents
├── cleaned/
│   └── corpus.jsonl           # JSONL format
├── processed/
│   ├── documents.json         # Parsed documents
│   ├── tfidf_matrix.npz       # TF-IDF features
│   ├── tfidf_metrics.json     # Stage 2 metrics
│   ├── clusters.json          # Cluster metadata
│   └── cluster_assignments.npy # Cluster labels
└── experiments/
    ├── experiment_results.json
    ├── results_summary.csv
    └── plots/
        ├── latency_comparison.png
        ├── speedup_chart.png
        └── search_space_reduction.png
```

---

## Project Structure

```
distributed-text-analytics/
├── src/
│   ├── ingestion/          # Document parsing & text cleaning
│   │   ├── parser.py       # Multi-format parser (PDF/DOCX/TXT)
│   │   └── clean_text.py   # Text preprocessing
│   ├── analytics/          # Feature extraction
│   │   ├── tfidf_engine.py # PySpark TF-IDF computation
│   │   ├── ngram_extractor.py # N-gram extraction
│   │   └── similarity.py   # Document similarity
│   ├── clustering/         # Distributed clustering
│   │   ├── kmeans_cluster.py # PySpark K-means
│   │   └── cluster_metadata.py # Cluster summarization
│   ├── embedding/          # Vector embeddings
│   │   ├── embedding_engine.py # Sentence transformers
│   │   └── vector_store.py # FAISS index management
│   └── retrieval/          # Query & retrieval
│       ├── query.py        # Query embedding
│       ├── retrieval.py    # Cluster-aware search
│       ├── evaluation.py   # Performance metrics
│       └── visualization.py # Result plotting
├── tests/
│   └── test_retrieval.py   # Unit tests
├── data/
│   ├── raw/                # Input documents
│   ├── cleaned/            # Preprocessed corpus
│   ├── processed/          # Intermediate files
│   └── experiments/        # Results & visualizations
├── download_data.py        # Kaggle dataset downloader
├── run_parser.py           # Stage 1: Document parsing
├── run_feature_extraction.py # Stage 2: TF-IDF
├── run_clustering.py       # Stage 3: K-means
├── run_retrieval_experiments.py # Stage 4: Retrieval
├── prepare_experiment_data.py # Legacy: all-in-one script
└── requirements.txt        # Python dependencies
```

---

## CLI Scripts Reference

### 1. download_data.py

Downloads sample dataset from Kaggle.

```bash
python download_data.py
```

**Requires:** Kaggle API credentials (`~/.kaggle/kaggle.json`)

**Output:** `data/raw/resumes/` with PDF/DOCX/TXT files

---

### 2. run_parser.py (Stage 1)

Parses and cleans raw documents.

```bash
python run_parser.py
```

**Input:** `data/raw/` (recursively scans subdirectories)

**Output:**
- `data/cleaned/corpus.jsonl`
- `data/processed/documents.json`

**Processing:**
- PDF parsing (pdfminer.six)
- DOCX parsing (python-docx)
- Simple text cleaning

---

### 3. run_feature_extraction.py (Stage 2)

Creates TF-IDF features using PySpark.

```bash
# Basic usage
python run_feature_extraction.py

# Custom options
python run_feature_extraction.py \
  --input data/processed/documents.json \
  --output data/processed \
  --features 10000
```

**Input:** `data/processed/documents.json`

**Output:**
- `data/processed/tfidf_matrix.npz` - Feature matrix
- `data/processed/tfidf_metrics.json` - Performance metrics

**Processing:**
- HashingTF for feature hashing
- IDF computation
- Distributed via PySpark

---

### 4. run_clustering.py (Stage 3)

Performs distributed K-Means clustering.

```bash
# Basic usage
python run_clustering.py

# Custom options
python run_clustering.py \
  --documents data/processed/documents.json \
  --tfidf data/processed/tfidf_matrix.npz \
  --output data/processed \
  --clusters 15 \
  --workers 4
```

**Input:**
- `data/processed/documents.json`
- `data/processed/tfidf_matrix.npz`

**Output:**
- `data/processed/clusters.json` - Cluster metadata
- `data/processed/cluster_assignments.npy` - Assignment array

**Processing:**
- PySpark MLlib K-means
- Silhouette score evaluation
- Cluster statistics generation

---

### 5. run_retrieval_experiments.py (Stage 4)

Runs retrieval performance experiments.

```bash
# Basic usage
python run_retrieval_experiments.py

# Custom queries
python run_retrieval_experiments.py \
  --queries "neural networks" "data mining" "distributed systems"

# More runs for statistical accuracy
python run_retrieval_experiments.py --runs 20

# More results per query
python run_retrieval_experiments.py --k 10
```

**Input:**
- `data/processed/documents.json`
- `data/processed/clusters.json`

**Output:**
- `data/experiments/experiment_results.json`
- `data/experiments/results_summary.csv`
- `data/experiments/plots/*.png`

**Processing:**
- Sentence-transformers embeddings
- FAISS vector search
- Performance comparison

---

### 6. prepare_experiment_data.py (Legacy)

All-in-one script that combines Stages 1-3.

```bash
python prepare_experiment_data.py --clusters 10
```

**Note:** Use individual stage scripts for better control and monitoring.

---

## Understanding the Pipeline Stages

### Stage 1: Document Ingestion
- **Script:** `run_parser.py`
- **Purpose:** Extract and clean text from multi-format documents
- **Technologies:** pdfminer.six, python-docx
- **Output:** `data/processed/documents.json`

### Stage 2: TF-IDF Feature Extraction
- **Script:** `run_feature_extraction.py`
- **Purpose:** Convert text to numerical features
- **Technologies:** PySpark HashingTF, IDF
- **Output:** `data/processed/tfidf_matrix.npz`

### Stage 3: Distributed Clustering
- **Script:** `run_clustering.py`
- **Purpose:** Group similar documents using distributed computing
- **Technologies:** PySpark MLlib K-means
- **Output:** `data/processed/clusters.json`
- **Metrics:** Silhouette score, cluster statistics, performance metrics

### Stage 4: Cluster-Aware Retrieval
- **Script:** `run_retrieval_experiments.py`
- **Purpose:** Fast semantic search with performance evaluation
- **Technologies:** Sentence-transformers, FAISS
- **Methods:**
  1. **Cluster-Aware** - Searches only relevant cluster (~10% of docs)
  2. **Flat Retrieval** - Baseline search (100% of docs)
- **Output:** Performance metrics and visualizations

---

## Performance Metrics Explained

### Query Latency
Total time to process a query and return results (milliseconds).

**Expected values:**
- Cluster-aware: 20-40ms
- Flat search: 200-300ms (depends on corpus size)

### Speedup Factor
How many times faster cluster-aware is compared to flat search.

**Formula:** `speedup = flat_latency / cluster_aware_latency`

**Expected:** 5-10x speedup with 10 clusters

### Search Space Reduction
Percentage of documents NOT searched due to clustering.

**Formula:** `reduction = 100 * (1 - cluster_size / total_docs)`

**Expected:** 85-95% reduction with 10 clusters

### Scalability
Performance improves with larger corpora (more documents = higher speedup).

---

## Expected Results

### With 1000 documents, 10 clusters:
- **Cluster-aware latency:** ~30ms
- **Flat search latency:** ~235ms
- **Speedup:** ~8x faster
- **Search reduction:** ~90% fewer documents searched
- **Accuracy:** Similar relevance (cosine similarity scores)

### Interpretation:
- Cluster-aware retrieval is significantly faster
- Searches only 10% of documents on average
- Maintains search quality (finds relevant documents)
- Speedup increases with corpus size

---

## Troubleshooting

### "externally-managed-environment" error
**Cause:** Trying to install packages without a virtual environment on modern Linux systems

**Solution:** Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Verify activation:** Your prompt should show `(venv)` at the beginning

---

### "FileNotFoundError: documents.json"
**Cause:** Skipped a stage in the pipeline

**Solution:** Run stages in order:
```bash
python run_parser.py                    # Stage 1
python run_feature_extraction.py       # Stage 2
python run_clustering.py                # Stage 3
python run_retrieval_experiments.py    # Stage 4
```

---

### "Need at least 2 documents for clustering"
**Cause:** Insufficient documents in `data/raw/`

**Solution:** Add more documents or download sample data:
```bash
python download_data.py
```
Minimum recommended: 10+ documents

---

### "ModuleNotFoundError: No module named 'kagglehub'"
**Cause:** Missing optional dependency

**Solution:** Install kagglehub (only needed for download_data.py):
```bash
pip install kagglehub
```

**Alternative:** Skip download and add your own documents to `data/raw/`

---

### Slow first run
**Cause:** Sentence-transformer model downloading (~400MB)

**Expected behavior:** First run takes 30-60 seconds for model download. Subsequent runs are fast.

**Location:** Models cached in `~/.cache/torch/sentence_transformers/`

---

### Low speedup results
**Cause:** Small corpus size or too many clusters

**Solutions:**
- Use larger dataset (500+ documents recommended)
- Reduce number of clusters: `python run_clustering.py --clusters 8`
- Speedup scales with corpus size

---

### PySpark warnings
**Cause:** Java version or Spark configuration

**Safe to ignore:** Most PySpark warnings about local mode are informational

**Check Java:** Ensure Java 8 or 11 is installed:
```bash
java -version
```

---

### "Permission denied" when downloading data
**Cause:** Missing Kaggle API credentials

**Solution:**
1. Get API key from https://www.kaggle.com/settings
2. Place in `~/.kaggle/kaggle.json`
3. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

---

## Testing

Run the test suite to verify everything works:

```bash
# Run all tests with verbose output
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_retrieval.py -v

# Run specific test class
python -m pytest tests/test_retrieval.py::TestQueryEmbedder -v

# Show print statements
python -m pytest tests/ -v -s
```

**Test coverage:**
- Query embedding
- Vector store operations
- Retrieval methods (cluster-aware & flat)
- Performance evaluation

---

## Key Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Distributed Computing | **PySpark** | TF-IDF computation, K-means clustering |
| Document Parsing | **pdfminer.six, python-docx** | Multi-format extraction |
| Text Preprocessing | Simple cleaning | Basic text normalization |
| Embeddings | **Sentence-Transformers** | Semantic vector representations |
| Vector Search | **FAISS** | Fast similarity search |
| Clustering | **PySpark MLlib** | Distributed K-means |
| Visualization | **Matplotlib** | Performance charts |

---

## For Your Report

### Include These Visualizations:
1. **Latency Comparison** (`latency_comparison.png`)
   - Bar chart showing cluster-aware vs flat retrieval times

2. **Speedup Chart** (`speedup_chart.png`)
   - Speedup factors for each query

3. **Search Space Reduction** (`search_space_reduction.png`)
   - Shows efficiency gains from clustering

### Key Metrics to Report:
From `results_summary.csv`:
- **Average speedup:** e.g., 8.27x
- **Search space reduction:** e.g., 89.8%
- **Latency comparison:** e.g., 28ms vs 235ms
- **Silhouette score:** Cluster quality metric
- **Performance metrics:** Clustering time, workers, memory

### Discussion Points:
- How cluster-aware retrieval reduces computational cost
- Distributed computing benefits (PySpark for TF-IDF and clustering)
- Tradeoff between speed and completeness
- Scalability with larger document collections
- Impact of cluster quality on retrieval performance

---

## Project Highlights

✅ **Complete Pipeline** - End-to-end from raw docs to retrieval
✅ **Modular Stages** - Each stage can be run independently
✅ **Distributed Processing** - PySpark for TF-IDF and K-means
✅ **Multi-Format Support** - PDF, DOCX, TXT parsing
✅ **Fast Retrieval** - FAISS + cluster-aware search
✅ **Performance Evaluation** - Quantitative speedup analysis
✅ **Performance Tracking** - Metrics at each stage
✅ **Visualization** - Auto-generated charts
✅ **Reproducible** - CLI scripts, no manual steps
✅ **No Server/UI** - Pure distributed computing focus

---

## Quick Reference Commands

```bash
# Setup (first time only)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Complete workflow (with venv activated)
source venv/bin/activate  # Always activate first!
python download_data.py
python run_parser.py
python run_feature_extraction.py
python run_clustering.py
python run_retrieval_experiments.py

# View results
cat data/experiments/results_summary.csv
ls -lh data/experiments/plots/

# Run tests
python -m pytest tests/ -v

# Clean up data (be careful!)
rm -rf data/processed/* data/experiments/*

# Deactivate virtual environment when done
deactivate
```

---

## Summary

This distributed text analytics system demonstrates:

1. **Scalable document processing** with multi-format parsing
2. **Distributed feature extraction** using PySpark TF-IDF
3. **Distributed clustering** with PySpark K-means
4. **Efficient retrieval** via cluster-aware search
5. **Performance optimization** achieving 5-10x speedup
6. **Quantitative evaluation** with metrics and visualizations
7. **Modular architecture** with independent pipeline stages
8. **Performance tracking** at each stage for analysis

Built entirely with **CLI scripts** focused on distributed computing principles for CS532.
