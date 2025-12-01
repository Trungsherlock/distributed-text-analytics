# src/analytics/tfidf_engine.py

import time
import numpy as np
from typing import List, Dict

from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF
from pyspark.sql.types import StructType, StructField, StringType, IntegerType


class SparkTFIDFEngine:
    """
    Full Windows-safe TF-IDF:
    - Tokenizer (Spark)
    - Stopwords remover (Spark)
    - HashingTF (Spark JVM)
    - Compute IDF in NumPy (no Python workers)
    """

    def __init__(self, num_features: int = 5000):
        self.num_features = num_features

        self.spark = (
            SparkSession.builder
            .appName("TFIDFEngine")
            .master("local[*]")
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
            .config("spark.python.worker.reuse", "false")
            .getOrCreate()
        )


    def compute_tfidf(self, documents: List[Dict[str, str]]):

        start = time.time()
        N = len(documents)

        schema = StructType([
            StructField("id", IntegerType(), True),
            StructField("text", StringType(), True)
        ])

        df = self.spark.createDataFrame(
            [(d["id"], d["text"]) for d in documents],
            schema=schema
        )

        tokenizer = Tokenizer(inputCol="text", outputCol="words")
        df = tokenizer.transform(df)

        remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
        df = remover.transform(df)

        hashing = HashingTF(
            inputCol="filtered_words",
            outputCol="tf",
            numFeatures=self.num_features
        )
        df = hashing.transform(df)

        # Collect TF vectors (JVM -> Python), no Python worker inside Spark
        tf_vectors = df.select("tf").collect()
        tf = np.array([row["tf"].toArray() for row in tf_vectors])

        # Compute IDF manually in NumPy
        dfreq = np.count_nonzero(tf > 0, axis=0)
        idf = np.log((1 + N) / (1 + dfreq)) + 1

        tfidf = tf * idf

        metrics = {
            "num_documents": N,
            "num_features": self.num_features,
            "matrix_shape": tfidf.shape,
            "total_time_sec": round(time.time() - start, 4),
            "time_per_doc_sec": round((time.time() - start) / (N or 1), 4)
        }

        return tfidf, None, metrics

    def close(self):
        try:
            self.spark.stop()
        except:
            pass
