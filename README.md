# COMPSCI 532 Final Project 

## Document Ingestion and Preprocessing Pipeline  

This module implements the **data ingestion and text preprocessing pipeline** for our COMPSCI 532 final project.  
It supports automated parsing and normalization of **PDF, DOCX, and TXT documents**, generating standardized JSONL output for downstream components (similarity analysis, clustering, REST API, and UI integration).

---

Project Overview  

**Objective:**  
Efficiently ingest and parse up to 500 documents across multiple formats, achieving high text extraction accuracy and processing efficiency (< 2 seconds per document).

**Deliverable Output:**  
A unified, cleaned corpus (`corpus.jsonl`) ready for machine learning and document similarity tasks.

Features  

- Multi-format text ingestion (`PDF`, `DOCX`, `TXT`)  
- Automated preprocessing using **NLTK**  
  - Tokenization, stopword removal, and lemmatization  
- Structured JSONL output for downstream pipeline use  
- Modular, maintainable code with clear separation of concerns  
- Scalable for large document batches  

---

This project was developed by the team using standard Python tools and libraries.
AI assistance was limited to productivity support and code explanation.