# config/settings.py

"""
Central configuration for the project
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import torch

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# Spark Configuration
SPARK_APP_NAME = "DocumentAnalytics"
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[*]")
SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "4g")
SPARK_EXECUTOR_MEMORY = os.getenv("SPARK_EXECUTOR_MEMORY", "4g")

# Processing Configuration
MAX_DOCUMENTS_PER_BATCH = 100
PROCESSING_TIMEOUT_SECONDS = 300
TARGET_PROCESSING_TIME_PER_DOC = 2.0  # seconds

# Clustering Configuration
MIN_DOCUMENTS_FOR_CLUSTERING = 5
MAX_CLUSTERS = 20
SILHOUETTE_SCORE_THRESHOLD = 0.5

# Embedding Configuration
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", 
    "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_BATCH_SIZE = 32
USE_GPU = torch.cuda.is_available()

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 5000))
API_DEBUG = os.getenv("API_DEBUG", "true").lower() == "true"

# Vector Store Configuration
VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "chroma")  # or "faiss"
VECTOR_STORE_PERSIST = True

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = BASE_DIR / "logs" / "app.log"