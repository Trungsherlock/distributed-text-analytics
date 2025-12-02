#!/usr/bin/env python3
# run_parser.py

"""
Stage 1: Document Ingestion

Parses PDF, DOCX, and TXT files from a directory and extracts text content.
Outputs cleaned documents in JSON format for downstream processing.
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ingestion.parser import DocumentParser
from ingestion.clean_text import clean_text


def main():
    print("=" * 60)
    print("STAGE 1: DOCUMENT INGESTION")
    print("=" * 60)

    parser = DocumentParser()

    input_dir = 'data/raw'
    output_dir = 'data/processed'
    jsonl_file = 'data/cleaned/corpus.jsonl'

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Parse documents
    print(f"\nParsing documents from {input_dir}...")
    results = parser.parse_directory(
        directory=input_dir,
        recursive=True,
        verbose=True
    )

    if not results:
        print("\n❌ No documents found or parsed successfully")
        sys.exit(1)

    # Get accuracy report
    report = parser.get_accuracy_report()
    print(f"\n{'='*60}")
    print(f"Parsing Summary")
    print(f"{'='*60}")
    print(f"Total processed: {report['total_processed']} documents")
    print(f"Successful: {report['successful']}")
    print(f"Failed: {report['failed']}")
    print(f"Accuracy: {report['accuracy']:.1f}%")
    print(f"Average time: {report['avg_time_per_doc']:.2f}s per document")

    # Create structured documents with cleaned text
    print(f"\nCleaning and structuring documents...")
    documents = []
    for i, result in enumerate(results):
        if result['success']:
            # Apply simple text cleaning
            cleaned_text = clean_text(result['text'], simple=True)

            documents.append({
                'id': i,
                'file_name': result['file_name'],
                'format': result['format'],
                'text': cleaned_text,
                'metadata': result.get('metadata', {}),
                'word_count': result['word_count']
            })

    # Save as JSONL (backward compatibility)
    os.makedirs(os.path.dirname(jsonl_file), exist_ok=True)
    parser.save_to_jsonl(results, jsonl_file, include_metadata=False, verbose=False)
    print(f"   ✓ Saved JSONL to {jsonl_file}")

    # Save structured JSON for pipeline
    documents_file = os.path.join(output_dir, 'documents.json')
    with open(documents_file, 'w') as f:
        json.dump(documents, f, indent=2)
    print(f"   ✓ Saved structured documents to {documents_file}")

    print("\n" + "=" * 60)
    print("STAGE 1 COMPLETE")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  - {jsonl_file} (JSONL format)")
    print(f"  - {documents_file} (JSON format)")
    print(f"\nNext step:")
    print(f"  Run feature extraction:")
    print(f"    python run_feature_extraction.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
