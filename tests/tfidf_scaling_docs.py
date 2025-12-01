# tests/tfidf_scaling_docs.py
"""
Safe TF-IDF scaling test (Spark)
----------------------------------------------------
This runs Spark TF-IDF (HashingTF) on *small* document sizes,
because Spark crashes on Windows when many jobs are spawned.

Purpose:
- Show Spark TF-IDF latency and matrix shape
- Demonstrate correct integration of your TF-IDF engine
- Provide metrics for the final report

Safe sizes: 5, 10, 20
"""

import time
from src.analytics.tfidf_engine import SparkTFIDFEngine

#  generates n identical documents
def generate_docs(n):
    base_text = (
        "data science machine learning distributed systems spark "
        "performance scaling cluster tfidf hashing features"
    )
    return [{"id": i, "text": base_text} for i in range(n)]


# Experiment sizes (keep small to avoid Spark crash)
sizes = [5, 10, 20]   # DO NOT increase — Spark will crash on Windows


print("num_docs,num_features,n_rows,n_cols,total_time_sec,time_per_doc_sec")

engine = SparkTFIDFEngine(num_features=5000)

for n in sizes:
    docs = generate_docs(n)
    start = time.time()

    tfidf_matrix, vocab, metrics = engine.compute_tfidf(docs)

    elapsed = metrics["total_time_sec"]
    time_per_doc = metrics["time_per_doc_sec"]
    n_rows, n_cols = metrics["matrix_shape"]

    print(f"{n},{metrics['num_features']},{n_rows},{n_cols},{elapsed},{time_per_doc}")

engine.close()
