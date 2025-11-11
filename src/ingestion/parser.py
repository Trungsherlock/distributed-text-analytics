import os
import json
from typing import Dict, List, Any
from pathlib import Path
# import pdfminer.six
from pdfminer.high_level import extract_text
from docx import Document
import chardet
import time

class DocumentParser:
    """
    Unified document parser for multiple formats
    Target: < 2 seconds per document, 95% extraction accuracy
    """
    
    SUPPORTED_FORMATS = {'.pdf', '.docx', '.txt', '.json'}
    
    def __init__(self):
        self.extraction_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'avg_processing_time': 0
        }
    
    def parse_document(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a document and extract text with metadata
        
        Args:
            file_path: Path to the document
            
        Returns:
            Dictionary containing text, metadata, and processing info
        """
        start_time = time.time()
        
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = path.suffix.lower()
        if extension not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {extension}")
        
        result = {
            'file_path': str(path),
            'file_name': path.name,
            'file_size': path.stat().st_size,
            'format': extension,
            'text': '',
            'metadata': {},
            'extraction_time': 0,
            'success': False,
            'error': None
        }
        
        try:
            # Format-specific parsing
            if extension == '.pdf':
                result['text'] = self._parse_pdf(file_path)
            elif extension == '.docx':
                result['text'] = self._parse_docx(file_path)
            elif extension == '.txt':
                result['text'] = self._parse_txt(file_path)
            elif extension == '.json':
                result['text'], result['metadata'] = self._parse_json(file_path)
            
            result['success'] = True
            result['char_count'] = len(result['text'])
            result['word_count'] = len(result['text'].split())
            
        except Exception as e:
            result['error'] = str(e)
            self.extraction_stats['failed'] += 1
        
        # Update statistics
        processing_time = time.time() - start_time
        result['extraction_time'] = processing_time
        self._update_stats(result)
        
        return result
    
    def _parse_pdf(self, file_path: str) -> str:
        """Extract text from PDF using pdfminer.six"""
        text = extract_text(file_path)
        return text.strip()
    
    def _parse_docx(self, file_path: str) -> str:
        """Extract text from DOCX"""
        doc = Document(file_path)
        paragraphs = [para.text for para in doc.paragraphs]
        return '\n'.join(paragraphs).strip()
    
    def _parse_txt(self, file_path: str) -> str:
        """Extract text from TXT with encoding detection"""
        # Detect encoding
        with open(file_path, 'rb') as file:
            raw = file.read()
            encoding_info = chardet.detect(raw)
            encoding = encoding_info['encoding'] or 'utf-8'
        
        # Read with detected encoding
        with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
            return file.read().strip()
    
    def _parse_json(self, file_path: str) -> tuple:
        """Extract text and metadata from JSON"""
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Extract text from various possible fields
        text_fields = ['text', 'content', 'body', 'description']
        text = ''
        for field in text_fields:
            if field in data:
                text = str(data[field])
                break
        
        # Remaining fields as metadata
        metadata = {k: v for k, v in data.items() if k not in text_fields}
        return text, metadata
    
    def _update_stats(self, result: Dict):
        """Update extraction statistics"""
        self.extraction_stats['total_processed'] += 1
        if result['success']:
            self.extraction_stats['successful'] += 1
        
        # Update average processing time
        n = self.extraction_stats['total_processed']
        avg = self.extraction_stats['avg_processing_time']
        self.extraction_stats['avg_processing_time'] = (
            (avg * (n - 1) + result['extraction_time']) / n
        )
    
    def get_accuracy_report(self) -> Dict:
        """Calculate and return accuracy metrics"""
        total = self.extraction_stats['total_processed']
        if total == 0:
            return {'accuracy': 0, 'stats': self.extraction_stats}
        
        accuracy = (self.extraction_stats['successful'] / total) * 100
        return {
            'accuracy': accuracy,
            'avg_time_per_doc': self.extraction_stats['avg_processing_time'],
            'total_processed': total,
            'successful': self.extraction_stats['successful'],
            'failed': self.extraction_stats['failed']
        }