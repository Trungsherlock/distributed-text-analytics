from pyspark.sql import SparkSession
from pyspark.ml.feature import (
    RegexTokenizer,
    StopWordsRemover,
    NGram,
    CountVectorizer,
    IDF
)
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.functions import concat, col
import numpy as np
from typing import List, Dict, Tuple

class SparkTFIDFEngine:
    """
    Distributed TF-IDF computation using Spark MLlib,
    with support for unigrams + bigrams, stop-word removal,
    dropping very common and very rare terms.
    """
    def __init__(self,
                 vocab_size: int = 10000,
                 ngram_range: Tuple[int, int] = (1, 2),
                 min_df: float = 2,
                 max_df: float = 0.8,
                 stop_words_lang: str = 'english'):
        """
        Args:
          vocab_size: maximum size of vocabulary (for CountVectorizer)
          ngram_range: tuple (min_n, max_n) for n-grams (e.g. (1,2))
          min_df: minimum document frequency threshold (int or fraction)
          max_df: maximum document frequency threshold (fraction)
          stop_words_lang: language for default stop words list (Spark’s StopWordsRemover)
        """
        self.spark = SparkSession.builder \
            .appName("DocumentTFIDF") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .getOrCreate()
        
        self.vocab_size = vocab_size
        self.min_n, self.max_n = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.stop_words_lang = stop_words_lang
        
        self.cv_model = None
        self.idf_model = None
        self.vocabulary = None
    
    def compute_tfidf(self, documents: List[Dict[str, str]]):
        """
        Compute TF-IDF vectors for documents.

        Args:
          documents: List of dicts each with 'id' and 'text'

        Returns:
          TF-IDF matrix as numpy array, plus vocabulary list
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
        
        # Tokenize (unigrams)
        tokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W+", toLowercase=True)
        words_df = tokenizer.transform(df)
        
        # Remove stop words
        remover = StopWordsRemover(
            inputCol="words",
            outputCol="filtered_words",
            stopWords=StopWordsRemover.loadDefaultStopWords(self.stop_words_lang),
            caseSensitive=False
        )
        filtered_df = remover.transform(words_df)
        
        # Build n-grams (if max_n > 1)
        # We’ll generate bigrams if max_n =2; you could loop for higher.
        ngram_cols = []
        current_col = "filtered_words"
        all_grams_col = current_col
        for n in range(2, self.max_n + 1):
            ngram = NGram(n=n, inputCol=current_col, outputCol=f"{n}gram_words")
            filtered_df = ngram.transform(filtered_df)
            # Then we union the unigram list and ngram list into a single column
            # For simplicity: create a new column combining
            filtered_df = filtered_df.withColumn(
                "all_grams",
                concat(col("filtered_words"), col("2gram_words"))
            )
            current_col = "all_grams"
            all_grams_col = "all_grams"
        
        if self.max_n == 1:
            input_for_cv = "filtered_words"
        else:
            input_for_cv = all_grams_col
        
        # CountVectorizer to get term frequencies + drop rare/common terms
        from pyspark.ml.feature import CountVectorizer
        cv = CountVectorizer(
            inputCol=input_for_cv,
            outputCol="raw_features",
            vocabSize=self.vocab_size,
            minDF=self.min_df,
            maxDF=self.max_df
        )
        self.cv_model = cv.fit(filtered_df)
        vectorized_df = self.cv_model.transform(filtered_df)
        
        # Save vocabulary
        self.vocabulary = self.cv_model.vocabulary
        
        # Compute IDF
        idf = IDF(inputCol="raw_features", outputCol="features", minDocFreq=0)
        self.idf_model = idf.fit(vectorized_df)
        tfidf_df = self.idf_model.transform(vectorized_df)
        
        # Convert to numpy array
        tfidf_vectors = tfidf_df.select("features").collect()
        tfidf_matrix = np.array([vec["features"].toArray() for vec in tfidf_vectors])
        
        return tfidf_matrix, self.vocabulary
    
    def get_top_terms_per_document(
        self,
        tfidf_matrix: np.ndarray,
        vocab: List[str],
        top_k: int = 10
    ) -> List[List[Tuple[str, float]]]:
        top_terms_per_doc = []
        
        for doc_vector in tfidf_matrix:
            top_indices = np.argsort(doc_vector)[-top_k:][::-1]
            top_terms = [
                (vocab[idx], doc_vector[idx])
                for idx in top_indices if doc_vector[idx] > 0
            ]
            top_terms_per_doc.append(top_terms)
        return top_terms_per_doc
    
    def close(self):
        if self.spark:
            self.spark.stop()
