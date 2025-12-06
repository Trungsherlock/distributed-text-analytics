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
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ingestion.parser import DocumentParser
from ingestion.clean_text import clean_text


def plot_cumulative_latency(results: list, output_path: str = 'data/experiments/plots/parsing_cumulative_latency.png'):
    """
    Visualize cumulative latency for document processing.
    
    Shows how total processing time accumulates as more documents are processed.
    E.g., 100 documents processed in 5s, 500 documents in 20s, etc.
    
    Args:
        results: List of parsing results with 'extraction_time' field
        output_path: Path to save the plot
    """
    if not results:
        print("No results to visualize")
        return
    
    # Calculate cumulative times
    extraction_times = [r['extraction_time'] for r in results]
    cumulative_times = np.cumsum(extraction_times)
    doc_counts = np.arange(1, len(results) + 1)
    
    # Create the plot with a modern style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Main line plot
    ax.plot(doc_counts, cumulative_times, 
            color='#2563eb', linewidth=2.5, 
            label='Cumulative Processing Time')
    
    # Fill area under curve
    ax.fill_between(doc_counts, cumulative_times, 
                    alpha=0.2, color='#2563eb')
    
    # Add milestone markers at key points
    total_docs = len(results)
    milestones = []
    
    # Determine milestone intervals based on total documents
    if total_docs <= 50:
        interval = 10
    elif total_docs <= 200:
        interval = 25
    elif total_docs <= 500:
        interval = 50
    elif total_docs <= 1000:
        interval = 100
    else:
        interval = 250
    
    # Generate milestone points
    for i in range(interval, total_docs, interval):
        milestones.append(i)
    
    # Always include the last document
    if total_docs not in milestones:
        milestones.append(total_docs)
    
    # Plot milestone points
    milestone_times = [cumulative_times[m - 1] for m in milestones]
    ax.scatter(milestones, milestone_times, 
               color='#dc2626', s=80, zorder=5,
               label='Milestones', edgecolors='white', linewidths=1.5)
    
    # Annotate milestones
    for i, (doc_count, cum_time) in enumerate(zip(milestones, milestone_times)):
        # Alternate label positions to avoid overlap
        y_offset = 15 if i % 2 == 0 else -25
        
        ax.annotate(
            f'{doc_count} docs\n{cum_time:.2f}s',
            xy=(doc_count, cum_time),
            xytext=(0, y_offset),
            textcoords='offset points',
            ha='center',
            fontsize=9,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fef3c7', edgecolor='#f59e0b', alpha=0.9)
        )
    
    # Calculate and display statistics
    total_time = cumulative_times[-1]
    avg_time_per_doc = total_time / total_docs
    
    # Add statistics box
    stats_text = (
        f'Total Documents: {total_docs}\n'
        f'Total Time: {total_time:.2f}s\n'
        f'Avg Time/Doc: {avg_time_per_doc*1000:.1f}ms\n'
        f'Throughput: {total_docs/total_time:.1f} docs/s'
    )
    
    props = dict(boxstyle='round,pad=0.5', facecolor='#ecfdf5', edgecolor='#10b981', alpha=0.95)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, fontfamily='monospace')
    
    # Styling
    ax.set_xlabel('Number of Documents Processed', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Processing Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('📊 Document Parsing: Cumulative Latency Over Time', 
                 fontsize=14, fontweight='bold', pad=15)
    
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(0, total_docs + total_docs * 0.05)
    ax.set_ylim(0, total_time * 1.15)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✓ Cumulative latency plot saved to {output_path}")
    
    # Print summary table
    print("\n" + "=" * 50)
    print("CUMULATIVE LATENCY SUMMARY")
    print("=" * 50)
    print(f"{'Documents':<15} {'Cumulative Time':<20} {'Avg Time/Doc':<15}")
    print("-" * 50)
    
    for milestone in milestones:
        cum_time = cumulative_times[milestone - 1]
        avg = cum_time / milestone
        print(f"{milestone:<15} {cum_time:.3f}s{'':<13} {avg*1000:.2f}ms")
    
    print("=" * 50)


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

    # Generate cumulative latency visualization
    print(f"\nGenerating cumulative latency visualization...")
    plot_path = 'data/experiments/plots/parsing_cumulative_latency.png'
    plot_cumulative_latency(results, plot_path)

    print("\n" + "=" * 60)
    print("STAGE 1 COMPLETE")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  - {jsonl_file} (JSONL format)")
    print(f"  - {documents_file} (JSON format)")
    print(f"  - {plot_path} (Latency visualization)")
    print(f"\nNext step:")
    print(f"  Run feature extraction:")
    print(f"    python run_feature_extraction.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
