# tests/tfidf_scaling_sklearn.py

import time
import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# Load corpus
CORPUS_PATH = "data/cleaned/corpus.jsonl"

texts = []

if os.path.exists(CORPUS_PATH):
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "text" in obj and obj["text"].strip():
                    texts.append(obj["text"])
            except:
                continue

if not texts:
    print("# WARNING: No valid corpus found → using dummy data")
    texts = ["this is a sample document about topic modeling"] * 2000


# Experiment sizes
sizes = [10, 20, 50, 100, 200, 500, 1000]


# Run experiments
print("num_docs,vocab_size,n_rows,n_cols,total_time_sec,time_per_doc_sec")

for n in sizes:
    docs = texts[:n]

    start = time.time()

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(docs)

    elapsed = time.time() - start

    time_per_doc = elapsed / n

    print(f"{n},{len(vectorizer.vocabulary_)},{tfidf_matrix.shape[0]},{tfidf_matrix.shape[1]},{elapsed:.4f},{time_per_doc:.6f}")
