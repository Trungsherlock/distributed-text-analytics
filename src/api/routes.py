# src/api/routes.py

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import threading
import queue
from typing import List, Dict


from src.ingestion.parser import DocumentParser
from src.ingestion.preprocessor import TextPreprocessor
from src.analytics.ngram_extractor import NgramExtractor
from src.analytics.tfidf_engine import SparkTFIDFEngine
from src.clustering.kmeans_cluster import SparkKMeansClustering
from src.clustering.cluster_metadata import ClusterMetadataGenerator

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEMPLATE_DIR = os.path.join(BASE_DIR, "ui", "templates")

# Flask App Configuration
app = Flask(__name__,template_folder=TEMPLATE_DIR)
app.config['UPLOAD_FOLDER'] = "data/raw"
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# Create directories
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)


parser = DocumentParser()
preprocessor = TextPreprocessor()
ngram_extractor = NgramExtractor()

tfidf_engine = None
kmeans = None
metadata_gen = ClusterMetadataGenerator()

document_store: List[Dict] = []
cluster_data = {}
processing_queue = queue.Queue()

# Spark is NOT thread-safe → protect with lock
spark_lock = threading.Lock()



@app.route("/")
def index():
    return render_template("index.html")



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

    # Start background ingestion thread
    if uploaded:
        threading.Thread(target=process_documents_batch, daemon=True).start()

    return jsonify({
        "uploaded": uploaded,
        "errors": errors,
        "message": f"Queued {len(uploaded)} files for processing"
    })


def process_documents_batch():
    """
    Parse + preprocess + extract n-grams in background.
    Clustering is NOT done here (Spark cannot run from threads).
    """
    global document_store, tfidf_engine, kmeans

    new_docs = False

    while not processing_queue.empty():
        filepath = processing_queue.get()

        result = parser.parse_document(filepath)
        if not result["success"]:
            continue

        processed_text = preprocessor.preprocess(result["text"])
        top_ngrams = ngram_extractor.get_top_ngrams(processed_text)

        doc_id = len(document_store)
        document_store.append({
            "id": doc_id,
            "file_name": result["file_name"],
            "format": result["format"],
            "text": processed_text,
            "original_text": result["text"][:500],
            "ngrams": top_ngrams,
            "metadata": result["metadata"],
            "word_count": result["word_count"]
        })

        new_docs = True

    # Reset engines so next clustering uses updated documents
    if new_docs:
        tfidf_engine = None
        kmeans = None



@app.route("/api/cluster", methods=["POST"])
def trigger_clustering():
    if len(document_store) < 2:
        return jsonify({"error": "Need at least 2 documents"}), 400

    try:
        with spark_lock:  # prevent Spark multi-thread crash
            result = perform_clustering()

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500



def perform_clustering():
    global tfidf_engine, kmeans, cluster_data

    if tfidf_engine is None:
        tfidf_engine = SparkTFIDFEngine()

    if kmeans is None:
        kmeans = SparkKMeansClustering(
            n_clusters=min(5, len(document_store))
        )

    docs_for_tfidf = [
        {"id": d["id"], "text": d["text"]} for d in document_store
    ]

    # ---- TF-IDF ----
    tfidf_matrix, vocab, tfidf_metrics = tfidf_engine.compute_tfidf(docs_for_tfidf)

    # ---- Clustering ----
    labels = kmeans.fit_predict(tfidf_matrix)

    stats = kmeans.get_cluster_statistics(labels)

    cluster_data = {
        "clusters": {},
        "silhouette_score": kmeans.evaluate_clustering(),
        "num_clusters": kmeans.n_clusters,
        "total_documents": len(document_store),
        "tfidf_metrics": tfidf_metrics
    }

    # ---- Metadata ----
    for cid, s in stats.items():
        meta = metadata_gen.generate_cluster_metadata(
            cid,
            s["document_indices"],
            document_store,
            tfidf_matrix,
            vocab  # may be None - metadata_gen must handle this
        )
        cluster_data["clusters"][cid] = meta

    return cluster_data


@app.route("/api/clusters", methods=["GET"])
def get_clusters():
    if not cluster_data:
        return jsonify({"message": "No clustering yet"}), 404

    return jsonify(cluster_data)


@app.route("/api/cluster/<int:cluster_id>", methods=["GET"])
def get_cluster_details(cluster_id):
    if cluster_id not in cluster_data.get("clusters", {}):
        return jsonify({"error": "Cluster not found"}), 404

    info = cluster_data["clusters"][cluster_id]

    docs = [
        {
            "id": idx,
            "file_name": document_store[idx]["file_name"],
            "format": document_store[idx]["format"],
            "preview": document_store[idx]["original_text"][:200] + "..."
        }
        for idx in info["document_indices"]
    ]

    return jsonify({**info, "documents": docs})


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
        "preview": doc["original_text"],
        "top_unigrams": doc["ngrams"].get(1, [])[:10],
        "top_bigrams": doc["ngrams"].get(2, [])[:10]
    })


@app.route("/api/stats", methods=["GET"])
def get_statistics():
    accuracy = parser.get_accuracy_report()

    stats = {
        "total_documents": len(document_store),
        "clusters_created": len(cluster_data.get("clusters", {})),
        "parsing_accuracy": accuracy["accuracy"],
        "avg_processing_time": accuracy["avg_time_per_doc"],
        "silhouette_score": cluster_data.get("silhouette_score", 0),
        "formats": {}
    }

    for doc in document_store:
        fmt = doc["format"]
        stats["formats"][fmt] = stats["formats"].get(fmt, 0) + 1

    return jsonify(stats)



if __name__ == "__main__":
    app.run(debug=True, port=5000)
