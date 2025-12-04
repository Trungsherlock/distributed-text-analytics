# src/analytics/tfidf_engine.py

import time
import numpy as np
from typing import List, Dict
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF
from pyspark.sql.types import StructType, StructField, IntegerType, ArrayType, StringType
import logging
import re


class SparkTFIDFEngine:
    """
    Windows-compatible TF-IDF using PySpark with pre-tokenization.
    - Tokenize in Python (no Python worker for this step)
    - Use only JVM-based Spark operations (HashingTF, IDF)
    - Avoids Python worker crashes on Windows
    """

    def __init__(self, num_features: int = 5000):
        self.spark = (
            SparkSession.builder
            .appName("TFIDFEngine")
            .master("local[*]")
            .config("spark.submit.deployMode", "client")
            .config("spark.driver.memory", "2g")
            .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
	    .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
	    .config("spark.executor.memory", "2g")
            .config("spark.sql.shuffle.partitions", "10")
            .config("spark.default.parallelism", "4")
            .getOrCreate()
        )

        self.num_features = num_features
        self.idf_model = None
        self.log = logging.getLogger("analytics")

        # Common English stop words
        self.stop_words = set([
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'when', 'where', 'who', 'which', 'why', 'how'
        ])

    # --------------------------------------------------------------

    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize and clean text in Python (avoiding Spark Python workers).
        """
        # Convert to lowercase
        text = text.lower()

        # Split on non-alphanumeric characters
        tokens = re.findall(r'\b[a-z]+\b', text)

        # Remove stop words and short tokens
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]

        return tokens

    # --------------------------------------------------------------

    def compute_tfidf(self, documents: List[Dict[str, str]]):
        """
        Compute TF-IDF matrix using PySpark HashingTF + IDF.
        Pre-tokenizes in Python to avoid Python worker issues.
        Returns: tfidf_matrix, vocabulary(None), metrics
        """

        start_time = time.time()

        # Filter out empty documents
        valid_docs = [doc for doc in documents if doc.get("text") and doc["text"].strip()]
        num_docs = len(valid_docs)

        if num_docs == 0:
            raise ValueError("No valid documents found (all documents are empty)")

        if num_docs < len(documents):
            print(f"Filtered out {len(documents) - num_docs} empty documents")
            self.log.warning(f"Filtered out {len(documents) - num_docs} empty documents")

        # Pre-tokenize in Python (avoids Spark Python workers)
        print("   Tokenizing documents...")
        tokenized_docs = []
        for doc in valid_docs:
            tokens = self._tokenize_text(doc["text"])
            if tokens:  # Only include docs with tokens
                tokenized_docs.append((doc["id"], tokens))

        if not tokenized_docs:
            raise ValueError("No documents with valid tokens after preprocessing")

        # Create Spark DataFrame with pre-tokenized data
        schema = StructType([
            StructField("id", IntegerType(), True),
            StructField("tokens", ArrayType(StringType()), True)
        ])

        df = self.spark.createDataFrame(tokenized_docs, schema=schema)

        # HashingTF → JVM-only operation
        print("   Computing TF-IDF features...")
        hashing_tf = HashingTF(
            inputCol="tokens",
            outputCol="raw_features",
            numFeatures=self.num_features
        )
        df = hashing_tf.transform(df)

        # IDF → JVM-only operation
        idf = IDF(inputCol="raw_features", outputCol="features")
        self.idf_model = idf.fit(df)
        df = self.idf_model.transform(df)

        # Collect results
        print("   Converting to matrix...")
        vectors = df.select("features").collect()
        tfidf_matrix = np.array([row["features"].toArray() for row in vectors])

        total_time = time.time() - start_time

        metrics = {
            "num_documents": len(tokenized_docs),
            "num_features": self.num_features,
            "matrix_shape": list(tfidf_matrix.shape),
            "total_time_sec": round(total_time, 4),
            "time_per_doc_sec": round(total_time / max(1, len(tokenized_docs)), 4),
        }

        self.log.info(
            f"[TF-IDF] docs={len(tokenized_docs)} | features={self.num_features} | "
            f"shape={tfidf_matrix.shape} | time={total_time:.4f}s"
        )

        return tfidf_matrix, None, metrics

    # --------------------------------------------------------------

    def close(self):
        if self.spark:
            self.spark.stop()
