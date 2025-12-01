# src/api/routes.py

from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
import threading
import queue
from typing import List, Dict
from pathlib import Path
import sys

# -------------------------------------------------------------------
# Fix import paths
# -------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

TEMPLATE_DIR = os.path.join(BASE_DIR, "ui", "templates")

# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------
from src.ingestion.parser import DocumentParser
from src.ingestion.preprocessor import TextPreprocessor
from src.analytics.ngram_extractor import NgramExtractor
from src.analytics.tfidf_engine import SparkTFIDFEngine
from src.clustering.kmeans_cluster import SparkKMeansClustering
from src.clustering.cluster_metadata import ClusterMetadataGenerator

# -------------------------------------------------------------------
# Flask App
# -------------------------------------------------------------------
app = Flask(__name__, template_folder=TEMPLATE_DIR)

app.config["UPLOAD_FOLDER"] = "data/raw"
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# -------------------------------------------------------------------
# Components
# -------------------------------------------------------------------
parser = DocumentParser()
preprocessor = TextPreprocessor()
ngram_extractor = NgramExtractor()

tfidf_engine = None
clustering_engine = None
metadata_gen = ClusterMetadataGenerator()

document_store: List[Dict] = []
cluster_data = {}
processing_queue = queue.Queue()

# Spark lock (Spark is NOT thread-safe)
spark_lock = threading.Lock()

# -------------------------------------------------------------------
# UI Routes
# -------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/scripts/<path:filename>")
def ui_scripts(filename):
    return send_from_directory(os.path.join(BASE_DIR, "ui", "scripts"), filename)


@app.route("/styles/<path:filename>")
def ui_styles(filename):
    return send_from_directory(os.path.join(BASE_DIR, "ui", "styles"), filename)

# -------------------------------------------------------------------
# Upload Endpoint
# -------------------------------------------------------------------
@app.route("/api/upload", methods=["POST"])
def upload_documents():
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files = request.files.getlist("files")
    uploaded, errors = [], []

    for file in files:
        if not file or not file.filename:
            continue

        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in DocumentParser.SUPPORTED_FORMATS:
            errors.append(f"{file.filename}: Unsupported format")
            continue

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        processing_queue.put(filepath)
        uploaded.append(filename)

    if uploaded:
        threading.Thread(target=process_documents_batch, daemon=True).start()

    return jsonify({
        "uploaded": uploaded,
        "errors": errors,
        "message": f"Queued {len(uploaded)} files for processing"
    })

# -------------------------------------------------------------------
# Background Document Processing
# -------------------------------------------------------------------
def process_documents_batch():
    global tfidf_engine, clustering_engine

    while not processing_queue.empty():
        filepath = processing_queue.get()
        result = parser.parse_document(filepath)

        if not result.get("success"):
            continue

        processed_text = preprocessor.preprocess(result["text"])
        top_ngrams = ngram_extractor.get_top_ngrams(processed_text)

        doc_id = len(document_store)

        original_text = result.get("text", "")
        document_store.append({
            "id": doc_id,
            "file_name": result["file_name"],
            "format": result["format"],
            "text": processed_text,
            "original_text": original_text,
            "preview_text": original_text[:500],
            "ngrams": top_ngrams,
            "metadata": result["metadata"],
            "word_count": result["word_count"],
        })

    # reset engines after new docs
    tfidf_engine = None
    clustering_engine = None

# -------------------------------------------------------------------
# Trigger Clustering
# -------------------------------------------------------------------
@app.route("/api/cluster", methods=["POST"])
def trigger_clustering():
    if len(document_store) < 2:
        return jsonify({"error": "Need at least 2 documents"}), 400

    try:
        with spark_lock:
            result = perform_clustering()

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------------------------
# Clustering Pipeline
# -------------------------------------------------------------------
def perform_clustering(num_workers=None):
    global tfidf_engine, clustering_engine, cluster_data
    import time

    total_start = time.time()

    if tfidf_engine is None:
        tfidf_engine = SparkTFIDFEngine()

    if clustering_engine is None:
        clustering_engine = SparkKMeansClustering(
            n_clusters=min(10, len(document_store)),
            num_workers=num_workers
        )

    docs_for_tfidf = [{"id": d["id"], "text": d["text"]} for d in document_store]

    # TF-IDF
    tfidf_start = time.time()
    tfidf_matrix, vocabulary, tfidf_metrics = tfidf_engine.compute_tfidf(docs_for_tfidf)
    tfidf_time = time.time() - tfidf_start

    # clustering
    labels = clustering_engine.fit_predict(tfidf_matrix)
    cluster_stats = clustering_engine.get_cluster_statistics(labels)

    cluster_data = {
        "clusters": {},
        "silhouette_score": clustering_engine.evaluate_clustering(),
        "num_clusters": clustering_engine.n_clusters,
        "total_documents": len(document_store),
        "tfidf_metrics": tfidf_metrics,
    }

    # attach metadata
    for cid, stats in cluster_stats.items():
        metadata = metadata_gen.generate_cluster_metadata(
            cid,
            stats["document_indices"],
            document_store,
            tfidf_matrix,
            vocabulary,
        )
        cluster_data["clusters"][cid] = metadata

    return cluster_data

# -------------------------------------------------------------------
# Get Cluster Summary
# -------------------------------------------------------------------
@app.route("/api/clusters", methods=["GET"])
def get_clusters():
    if not cluster_data:
        return jsonify({"message": "No clustering yet"}), 404
    return jsonify(cluster_data)

# -------------------------------------------------------------------
# Cluster Details Endpoint
# -------------------------------------------------------------------
@app.route("/api/cluster/<int:cluster_id>", methods=["GET"])
def get_cluster_details(cluster_id):
    if cluster_id not in cluster_data.get("clusters", {}):
        return jsonify({"error": "Cluster not found"}), 404

    info = cluster_data["clusters"][cluster_id]
    docs = []

    for idx in info["document_indices"]:
        doc = document_store[idx]
        preview_src = doc.get("preview_text") or doc.get("original_text", "")

        docs.append({
            "id": idx,
            "file_name": doc["file_name"],
            "format": doc["format"],
            "preview": preview_src[:200] + "...",
        })

    return jsonify({**info, "documents": docs})

# -------------------------------------------------------------------
# Document Details
# -------------------------------------------------------------------
@app.route("/api/documents/<int:doc_id>", methods=["GET"])
def get_document(doc_id):
    if doc_id < 0 or doc_id >= len(document_store):
        return jsonify({"error": "Document not found"}), 404

    doc = document_store[doc_id]

    return jsonify({
        "id": doc["id"],
        "file_name": doc["file_name"],
        "format": doc["format"],
        "word_count": doc["word_count"],
        "preview": doc.get("preview_text") or doc.get("original_text", "")[:500],
        "clean_text": doc.get("text", ""),
        "original_text": doc.get("original_text", ""),
        "metadata": doc.get("metadata", {}),
        "top_unigrams": doc["ngrams"].get(1, [])[:10],
        "top_bigrams": doc["ngrams"].get(2, [])[:10],
    })

# -------------------------------------------------------------------
# System Statistics
# -------------------------------------------------------------------
@app.route("/api/stats", methods=["GET"])
def get_statistics():
    accuracy_report = parser.get_accuracy_report()

    stats = {
        "total_documents": len(document_store),
        "clusters_created": len(cluster_data.get("clusters", {})),
        "parsing_accuracy": accuracy_report["accuracy"],
        "avg_processing_time": accuracy_report["avg_time_per_doc"],
        "silhouette_score": cluster_data.get("silhouette_score", 0),
        "formats": {},
    }

    for doc in document_store:
        fmt = doc["format"]
        stats["formats"][fmt] = stats["formats"].get(fmt, 0) + 1

    return jsonify(stats)

# -------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
