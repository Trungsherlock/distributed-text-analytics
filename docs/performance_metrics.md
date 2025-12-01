# Performance Metrics and Distributed Computing Demo

This document explains the new performance tracking and benchmarking features added to the distributed text analytics system.

## Overview

The system now tracks detailed performance metrics during clustering operations and provides benchmarking capabilities to demonstrate distributed computing concepts.

## Key Concepts Demonstrated

### 1. Distributed Computing with Spark
- K-Means is **embarrassingly parallel** - different workers can process different documents simultaneously
- Spark automatically partitions data across workers
- Each worker computes cluster assignments independently

### 2. Speedup Analysis
- **Speedup = T₁ / Tₙ** where T₁ is time with 1 worker, Tₙ with n workers
- **Ideal speedup**: Linear (2x workers = 2x speedup)
- **Real speedup**: Sublinear due to coordination overhead

### 3. Parallel Efficiency
- **Efficiency = (Speedup / Number of Workers) × 100%**
- 100% = perfect scaling
- <50% = significant overhead, diminishing returns

### 4. Amdahl's Law
Demonstrates that speedup is limited by the sequential portions of the algorithm.

## Performance Metrics Collected

### Clustering Metrics
1. **clustering_time_seconds**: Total time for K-Means clustering
2. **memory_usage_mb**: Memory consumed during clustering
3. **num_workers**: Number of Spark workers used
4. **num_partitions**: How data was partitioned across workers
5. **num_iterations**: Iterations needed for convergence
6. **num_documents**: Total documents clustered
7. **feature_dimensions**: Size of TF-IDF feature vectors
8. **documents_per_worker**: Distribution of work

### Pipeline Metrics
1. **tfidf_computation_time_seconds**: Time for TF-IDF vectorization
2. **total_pipeline_time_seconds**: End-to-end processing time
3. **avg_time_per_document_ms**: Average time per document

### Distributed Computing Analysis
1. **parallelism_factor**: Level of parallelism
2. **data_partitions**: Data distribution strategy
3. **documents_per_partition**: Work distribution
4. **convergence_iterations**: Algorithm convergence behavior

## API Endpoints

### 1. Perform Clustering with Metrics
```bash
POST /api/cluster
```

Returns clustering results with performance metrics included.

**Response includes:**
- Cluster assignments with labels
- Performance metrics
- Distributed computing analysis
- Cluster quality scores

### 2. Get Current Metrics
```bash
GET /api/cluster/metrics
```

Retrieves performance metrics from the last clustering operation.

**Example response:**
```json
{
  "performance": {
    "clustering_time_seconds": 2.456,
    "memory_usage_mb": 128.5,
    "num_workers": 4,
    "num_iterations": 12,
    "tfidf_computation_time_seconds": 1.234,
    "total_pipeline_time_seconds": 3.690
  },
  "distributed_analysis": {
    "parallelism_factor": 4,
    "data_partitions": 8,
    "documents_per_partition": 12,
    "convergence_iterations": 12
  },
  "cluster_quality": {
    "silhouette_score": 0.42,
    "num_clusters": 5,
    "total_documents": 100
  }
}
```

### 3. Run Scalability Benchmark ⭐ MOST IMPORTANT
```bash
POST /api/benchmark/scalability
Content-Type: application/json

{
  "worker_configs": [1, 2, 4, 8],
  "n_clusters": 5
}
```

Runs clustering with different worker counts to demonstrate distributed computing speedup.

**Example response:**
```json
{
  "status": "success",
  "benchmark_results": [
    {
      "num_workers": 1,
      "time_seconds": 5.234,
      "speedup": 1.0,
      "efficiency_percent": 100.0,
      "memory_mb": 150.2,
      "iterations": 15
    },
    {
      "num_workers": 2,
      "time_seconds": 2.987,
      "speedup": 1.75,
      "efficiency_percent": 87.5,
      "memory_mb": 145.8,
      "iterations": 15
    },
    {
      "num_workers": 4,
      "time_seconds": 1.876,
      "speedup": 2.79,
      "efficiency_percent": 69.75,
      "memory_mb": 142.3,
      "iterations": 15
    },
    {
      "num_workers": 8,
      "time_seconds": 1.456,
      "speedup": 3.59,
      "efficiency_percent": 44.88,
      "memory_mb": 148.9,
      "iterations": 15
    }
  ],
  "summary": {
    "optimal_worker_count": 4,
    "best_speedup": 3.59,
    "best_speedup_workers": 8,
    "parallel_overhead_detected": true,
    "recommendation": "Use 4 workers for best efficiency with 100 documents"
  },
  "insights": {
    "amdahl_law": "Speedup is limited by the sequential portions of the algorithm",
    "coordination_cost": "More workers increase communication and coordination overhead",
    "sweet_spot": "For 100 documents, 4 workers provide the best efficiency"
  }
}
```

## Usage Examples

### Python/Requests
```python
import requests

# Run clustering with metrics
response = requests.post('http://localhost:5000/api/cluster')
data = response.json()
print(f"Clustering took {data['performance_metrics']['clustering_time_seconds']}s")

# Get metrics
metrics = requests.get('http://localhost:5000/api/cluster/metrics').json()
print(f"Used {metrics['distributed_analysis']['parallelism_factor']} workers")

# Run scalability benchmark
benchmark = requests.post('http://localhost:5000/api/benchmark/scalability', json={
    'worker_configs': [1, 2, 4, 8],
    'n_clusters': 5
}).json()
print(f"Best speedup: {benchmark['summary']['best_speedup']}x")
print(f"Optimal workers: {benchmark['summary']['optimal_worker_count']}")
```

### cURL
```bash
# Trigger clustering
curl -X POST http://localhost:5000/api/cluster

# Get metrics
curl http://localhost:5000/api/cluster/metrics

# Run benchmark
curl -X POST http://localhost:5000/api/benchmark/scalability \
  -H "Content-Type: application/json" \
  -d '{"worker_configs": [1, 2, 4, 8], "n_clusters": 5}'
```

## Interpreting Results

### Speedup Analysis
- **Linear speedup**: Efficiency stays close to 100%
- **Sublinear speedup**: Efficiency decreases with more workers (normal)
- **No speedup**: May indicate coordination overhead exceeds parallelism benefits

### Optimal Configuration
The system automatically identifies:
1. **Best speedup**: Configuration with fastest time
2. **Best efficiency**: Configuration with best efficiency/speed tradeoff
3. **Overhead detection**: When adding more workers hurts performance

### When to Use More Workers
- Large document sets (>1000 documents)
- Complex feature spaces (high-dimensional TF-IDF)
- Multiple clustering iterations

### When Not to Use More Workers
- Small document sets (<100 documents)
- Overhead exceeds benefits (efficiency <30%)
- Limited memory or network bandwidth

## Cluster Labeling

Clusters are automatically labeled with:
- **simple_label**: Top 3 keywords (e.g., "BANKING | FINANCE | CREDIT")
- **top_terms**: Top 5-10 keywords with TF-IDF scores
- **label**: Human-readable description

This makes clusters interpretable without manual inspection.

## Example Output

```json
{
  "clusters": {
    "0": {
      "cluster_id": 0,
      "simple_label": "MEDICAL | PATIENT | TREATMENT",
      "label": "Medical/Clinical - Medical / Patient / Treatment",
      "size": 23,
      "percentage": 23.0,
      "top_terms": [
        ["medical", 0.432],
        ["patient", 0.387],
        ["treatment", 0.312]
      ]
    }
  },
  "performance_metrics": {
    "clustering_time_seconds": 2.456,
    "num_workers": 4,
    "num_iterations": 12
  }
}
```

## System Requirements

- **psutil**: For memory tracking (added to requirements.txt)
- **pyspark**: Distributed computing framework
- **pandas**: For benchmark result formatting
- **flask**: REST API framework

## Next Steps

1. Upload documents via `/api/upload`
2. Trigger clustering via `/api/cluster`
3. View metrics via `/api/cluster/metrics`
4. Run benchmark via `/api/benchmark/scalability`
5. Analyze speedup and efficiency results

## Performance Tips

1. **Start small**: Test with 1-2 workers first
2. **Scale up**: Gradually increase workers to find sweet spot
3. **Monitor memory**: High memory usage may indicate too many workers
4. **Check convergence**: More workers shouldn't change iteration count significantly
5. **Network matters**: Distributed overhead is higher on slower networks

