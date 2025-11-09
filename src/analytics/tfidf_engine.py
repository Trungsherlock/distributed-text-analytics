# src/analytics/tfidf_engine.py

from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import numpy as np
from typing import List, Dict, Tuple

class SparkTFIDFEngine:
    """
    Distributed TF-IDF computation using Spark MLlib
    """
    
    def __init__(self, num_features=10000):
        """
        Initialize Spark session and TF-IDF parameters
        
        Args:
            num_features: Number of features for hashing
        """
        self.spark = SparkSession.builder \
            .appName("DocumentTFIDF") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .getOrCreate()
        
        self.num_features = num_features
        self.hashing_tf = None
        self.idf_model = None
        self.vocabulary = None
    
    def compute_tfidf(self, documents: List[Dict[str, str]]) -> np.ndarray:
        """
        Compute TF-IDF vectors for documents
        
        Args:
            documents: List of dictionaries with 'id' and 'text' keys
            
        Returns:
            TF-IDF matrix as numpy array
        """
        # Create DataFrame
        schema = StructType([
            StructField("id", IntegerType(), True),
            StructField("text", StringType(), True)
        ])
        
        df = self.spark.createDataFrame(
            [(doc['id'], doc['text']) for doc in documents],
            schema=schema
        )
        
        # Tokenization
        tokenizer = Tokenizer(inputCol="text", outputCol="words")
        words_df = tokenizer.transform(df)
        
        # Compute TF
        self.hashing_tf = HashingTF(
            inputCol="words", 
            outputCol="raw_features",
            numFeatures=self.num_features
        )
        tf_df = self.hashing_tf.transform(words_df)
        
        # Compute IDF
        idf = IDF(inputCol="raw_features", outputCol="features")
        self.idf_model = idf.fit(tf_df)
        tfidf_df = self.idf_model.transform(tf_df)
        
        # Convert to numpy array
        tfidf_vectors = tfidf_df.select("features").collect()
        tfidf_matrix = np.array([
            vec["features"].toArray() for vec in tfidf_vectors
        ])
        
        return tfidf_matrix
    
    def get_top_terms_per_document(
        self, 
        tfidf_matrix: np.ndarray, 
        vocab: List[str], 
        top_k: int = 10
    ) -> List[List[Tuple[str, float]]]:
        """
        Extract top k terms for each document based on TF-IDF scores
        """
        top_terms_per_doc = []
        
        for doc_vector in tfidf_matrix:
            # Get indices of top k values
            top_indices = np.argsort(doc_vector)[-top_k:][::-1]
            
            # Get terms and scores
            top_terms = [
                (vocab[idx] if idx < len(vocab) else f"term_{idx}", 
                 doc_vector[idx])
                for idx in top_indices if doc_vector[idx] > 0
            ]
            
            top_terms_per_doc.append(top_terms)
        
        return top_terms_per_doc
    
    def close(self):
        """Close Spark session"""
        if self.spark:
            self.spark.stop()