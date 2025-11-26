# src/api/routes.py

from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import os
import json
from typing import Dict, List
import threading
import queue

# add project root to sys.path for easier imports
import os, sys

# add .../src to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

# Import our modules
from ingestion.parser import DocumentParser
from ingestion.preprocessor import TextPreprocessor
from analytics.ngram_extractor import NgramExtractor
from analytics.tfidf_engine import SparkTFIDFEngine
from clustering.kmeans_cluster import SparkKMeansClustering
from clustering.cluster_metadata import ClusterMetadataGenerator

#TODO: Try different clustering algorithms
import hdbscan



# fix base directory for templates for Flask app
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'ui', 'templates')
)
app.config['UPLOAD_FOLDER'] = 'data/raw'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Initialize components
parser = DocumentParser()
preprocessor = TextPreprocessor()
ngram_extractor = NgramExtractor()
tfidf_engine = None  # Initialize when needed
clustering_engine = None  # Initialize when needed
metadata_gen = ClusterMetadataGenerator()

# Storage for processed data
document_store = []
cluster_data = {}
processing_queue = queue.Queue()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_documents():
    """
    Endpoint for document upload
    Handles multiple files, validates format, starts processing
    """
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    print(f"DEBUG: Requess: {request}")
    files = request.files.getlist('files')
    uploaded_files = []
    errors = []

    #TODO: Check uploaded files for duplicates
    for file in files:
        if file and file.filename:
            # Validate file extension
            extension = os.path.splitext(file.filename)[1].lower()
            if extension not in DocumentParser.SUPPORTED_FORMATS:
                errors.append(f"{file.filename}: Unsupported format")
                continue
            
            # Save file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Add to processing queue
            processing_queue.put(filepath)
            uploaded_files.append(filename)
    print(f"DEBUG: Processing queue: {list(processing_queue.queue)}")
    # Start background processing
    if uploaded_files:
        threading.Thread(target=process_documents_batch, daemon=True).start()
    
    return jsonify({
        'uploaded': uploaded_files,
        'errors': errors,
        'message': f'Successfully uploaded {len(uploaded_files)} files for processing'
    })

def process_documents_batch():
    """
    Background task to process uploaded documents
    """
    global tfidf_engine, clustering_engine
    
    batch_documents = []
    
    # Process all queued documents
    while not processing_queue.empty():
        filepath = processing_queue.get()
        
        # Parse document
        result = parser.parse_document(filepath)

        # print(f"DEBUG: Checking parsed text: {result}")
        if result['success']:
            # Preprocess text
            processed_text = preprocessor.preprocess(result['text'])
            
            # Extract n-grams
            top_ngrams = ngram_extractor.get_top_ngrams(processed_text)
            
            # Store document
            doc_id = len(document_store)
            document_data = {
                'id': doc_id,
                'file_name': result['file_name'],
                'format': result['format'],
                'text': processed_text,
                'original_text': result['text'][:500],  # Store snippet
                'ngrams': top_ngrams,
                'metadata': result['metadata'],
                'word_count': result['word_count']
            }
            
            document_store.append(document_data)
            batch_documents.append(document_data)
    print(f"DEBUG: document length: {len(document_store)}")
    # Perform clustering if we have enough documents
    if len(document_store) >= 5:
        perform_clustering()

@app.route('/api/cluster', methods=['POST'])
def trigger_clustering():
    """
    Manually trigger clustering on current documents
    """

    if len(document_store) < 2:
        return jsonify({'error': 'Need at least 2 documents for clustering'}), 400
    
    try:
        result = perform_clustering()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def perform_clustering():
    """
    Perform TF-IDF and K-Means clustering
    """
    global tfidf_engine, clustering_engine, cluster_data
    
    # Initialize engines if needed
    if tfidf_engine is None:
        tfidf_engine = SparkTFIDFEngine()

    #TODO: Replace K-Means with some algorithm with unknow number of clusters, can deal with both little and large documents
    if clustering_engine is None:
        # clustering_engine = hdbscan.HDBSCAN( min_cluster_size=2, min_samples=2)
        clustering_engine = SparkKMeansClustering(n_clusters=min(10, len(document_store)))
    
    #TODO: TF-IDF should include n-grams
    # Prepare documents for TF-IDF
    docs_for_tfidf = [
        {'id': doc['id'], 'text': doc['text']} 
        for doc in document_store
    ]
    
    # Compute TF-IDF
    tfidf_matrix, vocabulary = tfidf_engine.compute_tfidf(docs_for_tfidf)

    # Check if TF-IDF matrix is valid
    if len(tfidf_matrix.shape) != 2:
        raise ValueError(f"Unexpected TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"DEBUG: TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"DEBUG: Vocabulary: {vocabulary}")
    
    # Perform clustering using HDBSCAN
    cluster_assignments = clustering_engine.fit_predict(tfidf_matrix)
    
    # Generate cluster metadata
    cluster_stats = clustering_engine.get_cluster_statistics(cluster_assignments)
    
    cluster_data = {
        'clusters': {},
        'silhouette_score': clustering_engine.evaluate_clustering(),
        'total_documents': len(document_store),
        'num_clusters': clustering_engine.n_clusters
    }
    
    for cluster_id, stats in cluster_stats.items():
        metadata = metadata_gen.generate_cluster_metadata(
            cluster_id,
            stats['document_indices'],
            document_store,
            tfidf_matrix,
            vocabulary
        )
        cluster_data['clusters'][cluster_id] = metadata
    
    print(f"DEBUG: Cluster data: {cluster_data}")
    return cluster_data

@app.route('/api/clusters', methods=['GET'])
def get_clusters():
    """
    Get current cluster information
    """
    if not cluster_data:
        return jsonify({'message': 'No clustering performed yet'}), 404
    
    return jsonify(cluster_data)

@app.route('/api/cluster/<int:cluster_id>', methods=['GET'])
def get_cluster_details(cluster_id):
    """
    Get detailed information about a specific cluster
    """
    if not cluster_data or cluster_id not in cluster_data['clusters']:
        return jsonify({'error': 'Cluster not found'}), 404
    
    cluster_info = cluster_data['clusters'][cluster_id]
    
    # Add document details
    cluster_documents = [
        {
            'id': idx,
            'file_name': document_store[idx]['file_name'],
            'format': document_store[idx]['format'],
            'preview': document_store[idx]['original_text'][:200] + '...'
        }
        for idx in cluster_info['document_indices']
    ]
    
    response = {
        **cluster_info,
        'documents': cluster_documents
    }
    
    return jsonify(response)

@app.route('/api/documents/<int:doc_id>', methods=['GET'])
def get_document(doc_id):
    """
    Get document details
    """
    if doc_id < 0 or doc_id >= len(document_store):
        return jsonify({'error': 'Document not found'}), 404
    
    doc = document_store[doc_id]
    return jsonify({
        'id': doc['id'],
        'file_name': doc['file_name'],
        'format': doc['format'],
        'word_count': doc['word_count'],
        'preview': doc['original_text'],
        'top_unigrams': doc['ngrams'].get(1, [])[:10] if 1 in doc['ngrams'] else [],
        'top_bigrams': doc['ngrams'].get(2, [])[:10] if 2 in doc['ngrams'] else []
    })

@app.route('/api/stats', methods=['GET'])
def get_statistics():
    """
    Get system statistics and performance metrics
    """
    accuracy_report = parser.get_accuracy_report()
    
    stats = {
        'total_documents': len(document_store),
        'clusters_created': len(cluster_data.get('clusters', {})),
        'parsing_accuracy': accuracy_report['accuracy'],
        'avg_processing_time': accuracy_report['avg_time_per_doc'],
        'silhouette_score': cluster_data.get('silhouette_score', 0),
        'formats': {}
    }
    
    # Count document formats
    for doc in document_store:
        fmt = doc['format']
        stats['formats'][fmt] = stats['formats'].get(fmt, 0) + 1
    
    return jsonify(stats)

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    app.run(debug=True, port=5000)