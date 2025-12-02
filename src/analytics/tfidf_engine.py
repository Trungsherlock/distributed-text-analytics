# src/analytics/tfidf_engine.py

import time
import numpy as np
from typing import List, Dict
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import logging


class SparkTFIDFEngine:
    """
    Stable TF-IDF computation using Spark HashingTF + IDF.
    - No Python worker crashes on Windows
    - Fixed-size vector
    - Fast + scalable
    """

    def __init__(self, num_features: int = 5000):
        self.spark = (
            SparkSession.builder
            .appName("TFIDFEngine")
            .master("local[*]")
            .config("spark.submit.deployMode", "client")
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
            .config("spark.python.worker.reuse", "false")
            .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
            .config("spark.python.worker.faulthandler.enabled", "true")
            .getOrCreate()
        )

        self.num_features = num_features
        self.idf_model = None
        self.log = logging.getLogger("analytics")

    # --------------------------------------------------------------

    def compute_tfidf(self, documents: List[Dict[str, str]]):
        """
        Compute TF-IDF matrix using HashingTF.
        Returns: tfidf_matrix, vocabulary(None), metrics
        """

        start_time = time.time()
        num_docs = len(documents)

        # Spark DataFrame schema
        schema = StructType([
            StructField("id", IntegerType(), True),
            StructField("text", StringType(), True)
        ])

        df = self.spark.createDataFrame(
            [(doc["id"], doc["text"]) for doc in documents],
            schema=schema
        )

        # Tokenize
        tokenizer = Tokenizer(inputCol="text", outputCol="words")
        df = tokenizer.transform(df)

        # Stopword removal
        remover = StopWordsRemover(
            inputCol="words",
            outputCol="filtered_words"
        )
        df = remover.transform(df)

        # HashingTF → fast, stable, JVM-only
        hashing_tf = HashingTF(
            inputCol="filtered_words",
            outputCol="raw_features",
            numFeatures=self.num_features
        )
        df = hashing_tf.transform(df)

        # IDF
        idf = IDF(inputCol="raw_features", outputCol="features")
        self.idf_model = idf.fit(df)
        df = self.idf_model.transform(df)

        # Convert to numpy
        vectors = df.select("features").collect()
        tfidf_matrix = np.array([row["features"].toArray() for row in vectors])

        total_time = time.time() - start_time

        metrics = {
            "num_documents": num_docs,
            "num_features": self.num_features,
            "matrix_shape": tfidf_matrix.shape,
            "total_time_sec": round(total_time, 4),
            "time_per_doc_sec": round(total_time / max(1, num_docs), 4),
        }

        self.log.info(
            f"[TF-IDF] docs={num_docs} | features={self.num_features} | "
            f"shape={tfidf_matrix.shape} | time={total_time:.4f}s"
        )

        return tfidf_matrix, None, metrics

    # --------------------------------------------------------------

    def close(self):
        if self.spark:
            self.spark.stop()
