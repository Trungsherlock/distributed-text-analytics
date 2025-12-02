# Stage 5: Query & Retrieval

## Overview

This stage implements and evaluates two retrieval approaches:

1. **Cluster-Aware Retrieval** - Searches only within the most relevant cluster (FAST)
2. **Flat Retrieval** - Searches all documents (BASELINE)

The goal is to demonstrate that cluster-aware retrieval provides significant speedup with acceptable accuracy trade-offs.

---

## Architecture

### Components

```
src/retrieval/
├── query.py           # Query embedding
├── retrieval.py       # Both search methods
├── evaluation.py      # Experiment runner
└── visualization.py   # Performance graphs

Scripts:
└── run_retrieval_experiments.py  # Run full evaluation
```

### How It Works

**Cluster-Aware Retrieval:**
1. Embed query text → vector
2. Compare query to cluster centroids
3. Find most similar cluster
4. Search ONLY within that cluster's documents
5. Return top-k results

**Flat Retrieval (Baseline):**
1. Embed query text → vector
2. Search ALL documents in FAISS
3. Return top-k results

---

## Quick Start

### Run Retrieval Experiments

```bash
# Make sure you have:
# - data/processed/documents.json (from preprocessing)
# - data/processed/clusters.json (from clustering)

# Run experiments
python run_retrieval_experiments.py

# This will:
# - Test 10 default queries
# - Run each query 10 times
# - Generate performance metrics
# - Create visualization plots
# - Save results to data/experiments/
```

---

## Running Experiments

### Basic Experiment

```bash
python run_retrieval_experiments.py
```

**Output:**
- `data/experiments/experiment_results.json` - Detailed results
- `data/experiments/results_summary.csv` - Summary table
- `data/experiments/plots/` - Visualizations:
  - `latency_comparison.png` - Bar chart comparing latencies
  - `speedup_chart.png` - Speedup factors per query
  - `latency_breakdown.png` - Stacked bar showing time components
  - `search_space_reduction.png` - Documents searched comparison

### Custom Queries

```bash
python run_retrieval_experiments.py \
  --queries "neural networks" "data mining" "web search" \
  --k 10 \
  --runs 20
```

### Custom Data Paths

```bash
python run_retrieval_experiments.py \
  --docs data/my_documents.json \
  --clusters data/my_clusters.json \
  --output results/my_experiment
```

---

## Performance Metrics

### What We Measure

**Primary Metrics:**
- **Query Latency** - Total time to process query (ms)
  - Embedding time
  - Cluster selection time (cluster-aware only)
  - FAISS search time
- **Speedup Factor** - `flat_latency / cluster_aware_latency`
- **Search Space Reduction** - Percentage of documents not searched

**Secondary Metrics:**
- **Precision** - Accuracy sanity check (optional)
- **Scalability** - How performance changes with corpus size

### Expected Results

Based on the strategy document, with 1000 documents and 10 clusters:

| Metric | Cluster-Aware | Flat Search | Improvement |
|--------|---------------|-------------|-------------|
| Latency | ~30ms | ~250ms | 8.3x faster |
| Docs Searched | ~100 (10%) | 1000 (100%) | 90% reduction |
| Precision | ~80% | ~100% | Acceptable trade-off |

---

## Understanding the Results

### Results Summary CSV

The `results_summary.csv` contains one row per query:

```csv
query,corpus_size,k,num_runs,embedding_time_ms,cluster_aware_latency_ms,cluster_aware_docs_searched,flat_latency_ms,flat_docs_searched,speedup,latency_diff_ms,search_space_reduction_pct
"machine learning",1000,5,10,15.2,28.5,102,235.7,1000,8.27,207.2,89.8
```

**Key columns:**
- `speedup` - How much faster cluster-aware is
- `search_space_reduction_pct` - % of documents NOT searched
- `latency_diff_ms` - Time saved per query

### Visualizations

**1. Latency Comparison** (`latency_comparison.png`)
- Bar chart showing query latency for each method
- Green bars (cluster-aware) should be shorter than red bars (flat)

**2. Speedup Chart** (`speedup_chart.png`)
- Shows speedup factor for each query
- Values above 1.0 indicate cluster-aware is faster
- Higher is better!

**3. Latency Breakdown** (`latency_breakdown.png`)
- Stacked bar showing time spent in each component
- Demonstrates WHERE the time savings come from

**4. Search Space Reduction** (`search_space_reduction.png`)
- Shows number of documents searched
- Demonstrates WHY cluster-aware is faster

---

## Interpreting Results for Your Report

### What to Include in Your Report

**1. System Design Trade-off (1 paragraph)**
> "We traded 20% accuracy for 5x speed improvement by restricting search to relevant clusters"

**2. Performance Results (1-2 paragraphs + graphs)**
> Include 2-3 key graphs showing:
> - Latency comparison
> - Speedup factors
> - Search space reduction

**3. Scalability Analysis (1 paragraph)**
> "As corpus grows, cluster-aware retrieval maintains constant query time while flat search degrades linearly"

**4. Conclusion (1 paragraph)**
> "For applications where 'good enough' results are acceptable, cluster-aware retrieval significantly reduces computational cost"

### Sample Results Section

```markdown
## Retrieval Performance Evaluation

We compared two retrieval strategies:
- **Cluster-aware**: Search only within most relevant cluster
- **Flat**: Search all documents

### Results on 1000-document corpus:
- Cluster-aware: 28ms average latency
- Flat search: 235ms average latency
- **Speedup: 8.4x**

### Accuracy sanity check (5 test queries):
- Cluster-aware: 3.8/5 relevant docs on average (76%)
- Flat search: 4.6/5 relevant docs on average (92%)

### Conclusion
Cluster-aware retrieval provides significant latency reduction with
acceptable accuracy loss for many applications. The benefit increases
with corpus size.
```

---

## Advanced Usage

### Custom Test Queries

Create a file `test_queries.txt`:
```
machine learning neural networks
data processing pipelines
distributed systems
```

Run with:
```bash
python run_retrieval_experiments.py --queries $(cat test_queries.txt)
```

### Programmatic Usage

```python
from src.retrieval.query import QueryEmbedder
from src.retrieval.retrieval import ClusterAwareRetrieval, FlatRetrieval
from src.retrieval.evaluation import RetrievalEvaluator

# Initialize components
query_embedder = QueryEmbedder()
# ... initialize retrievals ...

# Run single query
query_emb, time = query_embedder.embed_query("test query")
result = cluster_aware.search(query_emb, k=5)

# Run experiments
evaluator = RetrievalEvaluator(
    query_embedder, cluster_aware, flat, documents
)
results = evaluator.run_batch_experiments(["query1", "query2"])
evaluator.save_results("results.json")
```

---

## Troubleshooting

### "FileNotFoundError: documents.json"
- Make sure you have `data/processed/documents.json` from your preprocessing stage
- Make sure you have `data/processed/clusters.json` from your clustering stage
- Check that the paths in `run_retrieval_experiments.py` match your file locations

### Slow performance
- First run includes model loading (15-30 seconds)
- Subsequent queries are much faster
- Use `--runs 3` for quick testing

### Low speedup
- Speedup depends on corpus size and cluster count
- Larger corpus = better speedup
- More clusters = smaller search spaces = faster

---

## Dependencies

Required packages (already in requirements.txt):
- `sentence-transformers` - Query embedding
- `faiss` - Vector search (via vector_store.py)
- `scikit-learn` - Similarity metrics
- `matplotlib` - Visualizations
- `pandas` - Result analysis
- `numpy` - Numerical operations

---

## File Outputs

After running experiments, you'll have:

```
data/
├── experiments/
│   ├── experiment_results.json    # Detailed results
│   ├── results_summary.csv        # Summary table
│   └── plots/
│       ├── latency_comparison.png
│       ├── speedup_chart.png
│       ├── latency_breakdown.png
│       └── search_space_reduction.png
└── processed/
    ├── documents.json             # Exported documents
    └── clusters.json              # Exported clusters
```

---

## Design Decisions

### Why FAISS?
- Fast similarity search at scale
- Supports cluster-based partitioning
- Industry standard for vector search

### Why Sentence Transformers?
- State-of-the-art semantic embeddings
- Same model for documents and queries (consistency)
- Fast inference

### Why Compare to Flat Search?
- Flat search is the baseline everyone understands
- Shows clear benefit of clustering
- Demonstrates system design trade-offs

---

## Next Steps

1. **Run basic experiments** - Use default queries
2. **Generate visualizations** - Get graphs for report
3. **Optional: Test custom queries** - Domain-specific queries
4. **Optional: Accuracy check** - Manual verification on 3-5 queries
5. **Write analysis** - 1-2 pages explaining trade-offs

**Time allocation:**
- Setup & first run: 30 minutes
- Experiments: 1 hour
- Analysis & graphs: 1 hour
- **Total: 2.5 hours**

---

## Key Takeaways

✅ **Fast**: Cluster-aware searches 10% of documents (90% reduction)
✅ **Scalable**: Speedup increases with corpus size
✅ **Practical**: Acceptable accuracy trade-off for many applications
✅ **Measurable**: Clear quantitative metrics demonstrating benefit

The goal is to show you understand **system design trade-offs** and can **measure them quantitatively**.
