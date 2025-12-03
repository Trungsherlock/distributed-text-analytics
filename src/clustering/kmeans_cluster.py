# src/clustering/kmeans_cluster.py

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
import numpy as np
import time
import psutil
from typing import List, Dict, Tuple

class SparkKMeansClustering:
    """
    Distributed K-Means clustering using Spark MLlib
    """
    
    def __init__(self, n_clusters: int = 5, max_iter: int = 100, seed: int = 42, num_workers: int = None):
        """
        Initialize K-Means clustering
        
        Args:
            n_clusters: Number of clusters
            max_iter: Maximum iterations
            seed: Random seed
            num_workers: Number of Spark workers (None = auto-detect)
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.seed = seed
        
        # Configure Spark with specific worker count for performance testing
        builder = SparkSession.builder \
            .appName("KMeansClustering") \
            .config("spark.driver.memory", "1g") \
            .config("spark.executor.memory", "1g") \
            .config("spark.rpc.message.maxSize", "128") \
            .config("spark.driver.maxResultSize", "512m") \
            .config("spark.kryoserializer.buffer.max", "512m")
        
        if num_workers:
            builder = builder \
                .config("spark.executor.instances", str(num_workers)) \
                .config("spark.default.parallelism", str(num_workers * 2))
        
        self.spark = builder.getOrCreate()

        # Reduce logging noise
        self.spark.sparkContext.setLogLevel("ERROR")

        self.num_workers = num_workers or self.spark.sparkContext.defaultParallelism
        
        self.model = None
        self.predictions = None
        
        # Performance tracking attributes
        self.performance_metrics = {}
        self.iteration_count = 0
    
    def fit_predict(self, tfidf_matrix: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Fit K-Means model and predict clusters with performance metrics
        
        Args:
            tfidf_matrix: TF-IDF feature matrix
            
        Returns:
            Tuple of (cluster assignments, performance metrics dict)
        """
        # Start performance tracking
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Convert numpy array to Spark DataFrame
        def array_to_vector(array):
            return Vectors.dense(array.tolist())
        
        array_to_vector_udf = udf(array_to_vector, VectorUDT())
        
        # Create DataFrame with features and partition immediately
        # Convert to list of tuples first, then create RDD with explicit partitioning
        num_docs = len(tfidf_matrix)
        target_partitions = max(50, num_docs // 100)  # At least 50 partitions or 100 docs per partition

        # Create RDD with explicit partitioning to avoid large tasks
        data = [(Vectors.dense(row.tolist()),) for row in tfidf_matrix]
        rdd = self.spark.sparkContext.parallelize(data, target_partitions)
        df = self.spark.createDataFrame(rdd, ["features"])

        # Track partitioning
        num_partitions = df.rdd.getNumPartitions()
        
        # Train K-Means model
        kmeans = KMeans(
            k=self.n_clusters,
            maxIter=self.max_iter,
            seed=self.seed,
            featuresCol="features",
            predictionCol="cluster"
        )
        
        self.model = kmeans.fit(df)
        
        # Get actual iterations from model
        self.iteration_count = self.model.summary.numIter
        
        self.predictions = self.model.transform(df)
        
        # Extract cluster assignments
        clusters = self.predictions.select("cluster").collect()
        cluster_assignments = np.array([row["cluster"] for row in clusters])
        
        # End performance tracking
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Collect performance metrics
        self.performance_metrics = {
            'clustering_time_seconds': round(end_time - start_time, 3),
            'memory_usage_mb': round(end_memory - start_memory, 2),
            'num_workers': self.num_workers,
            'num_partitions': num_partitions,
            'num_iterations': self.iteration_count,
            'num_documents': len(tfidf_matrix),
            'feature_dimensions': tfidf_matrix.shape[1],
            'documents_per_worker': len(tfidf_matrix) // max(self.num_workers, 1),
        }
        
        return cluster_assignments, self.performance_metrics
    
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