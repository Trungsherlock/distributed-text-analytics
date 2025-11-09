# src/clustering/kmeans_cluster.py

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
import numpy as np
from typing import List, Dict, Tuple

class SparkKMeansClustering:
    """
    Distributed K-Means clustering using Spark MLlib
    """
    
    def __init__(self, n_clusters: int = 5, max_iter: int = 100, seed: int = 42):
        """
        Initialize K-Means clustering
        
        Args:
            n_clusters: Number of clusters
            max_iter: Maximum iterations
            seed: Random seed
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.seed = seed
        
        self.spark = SparkSession.builder \
            .appName("KMeansClustering") \
            .config("spark.driver.memory", "4g") \
            .getOrCreate()
        
        self.model = None
        self.predictions = None
    
    def fit_predict(self, tfidf_matrix: np.ndarray) -> np.ndarray:
        """
        Fit K-Means model and predict clusters
        
        Args:
            tfidf_matrix: TF-IDF feature matrix
            
        Returns:
            Cluster assignments for each document
        """
        # Convert numpy array to Spark DataFrame
        def array_to_vector(array):
            return Vectors.dense(array.tolist())
        
        array_to_vector_udf = udf(array_to_vector, VectorUDT())
        
        # Create DataFrame with features
        data = [(Vectors.dense(row.tolist()),) for row in tfidf_matrix]
        df = self.spark.createDataFrame(data, ["features"])
        
        # Train K-Means model
        kmeans = KMeans(
            k=self.n_clusters,
            maxIter=self.max_iter,
            seed=self.seed,
            featuresCol="features",
            predictionCol="cluster"
        )
        
        self.model = kmeans.fit(df)
        self.predictions = self.model.transform(df)
        
        # Extract cluster assignments
        clusters = self.predictions.select("cluster").collect()
        cluster_assignments = np.array([row["cluster"] for row in clusters])
        
        return cluster_assignments
    
    def get_cluster_centers(self) -> np.ndarray:
        """
        Get cluster centroids
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        centers = self.model.clusterCenters()
        return np.array([center.toArray() for center in centers])
    
    def evaluate_clustering(self) -> float:
        """
        Evaluate clustering using Silhouette score
        """
        if self.predictions is None:
            raise ValueError("No predictions available")
        
        evaluator = ClusteringEvaluator(
            predictionCol="cluster",
            featuresCol="features",
            metricName="silhouette"
        )
        
        silhouette = evaluator.evaluate(self.predictions)
        return silhouette
    
    def get_cluster_statistics(
        self, 
        cluster_assignments: np.ndarray
    ) -> Dict[int, Dict]:
        """
        Get statistics for each cluster
        """
        unique_clusters = np.unique(cluster_assignments)
        stats = {}
        
        for cluster_id in unique_clusters:
            cluster_docs = np.where(cluster_assignments == cluster_id)[0]
            stats[int(cluster_id)] = {
                'size': len(cluster_docs),
                'document_indices': cluster_docs.tolist(),
                'percentage': len(cluster_docs) / len(cluster_assignments) * 100
            }
        
        return stats
    
    def close(self):
        """Close Spark session"""
        if self.spark:
            self.spark.stop()