import os
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from pdfminer.high_level import extract_text
from docx import Document
import chardet
import time
import glob

class DocumentParser:

    SUPPORTED_FORMATS = {'.pdf', '.docx', '.txt', '.json'}

    def __init__(self):
        self.extraction_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'avg_processing_time': 0
        }

    def parse_document(self, file_path: str) -> Dict[str, Any]:
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

        processing_time = time.time() - start_time
        result['extraction_time'] = processing_time
        self._update_stats(result)

        return result

    def _parse_pdf(self, file_path: str) -> str:
        text = extract_text(file_path)
        return text.strip()

    def _parse_docx(self, file_path: str) -> str:
        doc = Document(file_path)
        paragraphs = [para.text for para in doc.paragraphs]
        return '\n'.join(paragraphs).strip()

    def _parse_txt(self, file_path: str) -> str:
        with open(file_path, 'rb') as file:
            raw = file.read()
            encoding_info = chardet.detect(raw)
            encoding = encoding_info['encoding'] or 'utf-8'

        with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
            return file.read().strip()

    def _parse_json(self, file_path: str) -> tuple:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        text_fields = ['text', 'content', 'body', 'description']
        text = ''
        for field in text_fields:
            if field in data:
                text = str(data[field])
                break

        metadata = {k: v for k, v in data.items() if k not in text_fields}
        return text, metadata

    def _update_stats(self, result: Dict):
        self.extraction_stats['total_processed'] += 1
        if result['success']:
            self.extraction_stats['successful'] += 1

        n = self.extraction_stats['total_processed']
        avg = self.extraction_stats['avg_processing_time']
        self.extraction_stats['avg_processing_time'] = (
            (avg * (n - 1) + result['extraction_time']) / n
        )

    def get_accuracy_report(self) -> Dict:
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

    def parse_batch(self, file_paths: List[str], verbose: bool = True) -> List[Dict[str, Any]]:
        results = []

        if verbose:
            print(f"[INFO] Processing {len(file_paths)} documents...\n")

        for i, file_path in enumerate(file_paths, 1):
            result = self.parse_document(file_path)
            results.append(result)

            if verbose:
                status = "✓" if result['success'] else "✗"
                print(f"[{i}/{len(file_paths)}] {status} {result['file_name']} - "
                      f"{result['extraction_time']:.2f}s")

        if verbose:
            report = self.get_accuracy_report()
            print(f"\n{'='*50}")
            print(f"Batch Processing Complete")
            print(f"{'='*50}")
            print(f"Total: {report['total_processed']}")
            print(f"Successful: {report['successful']}")
            print(f"Failed: {report['failed']}")
            print(f"Accuracy: {report['accuracy']:.1f}%")
            print(f"Average time: {report['avg_time_per_doc']:.2f}s per document")

        return results

    def parse_directory(self, directory: str, patterns: Optional[List[str]] = None,
                       recursive: bool = True, verbose: bool = True) -> List[Dict[str, Any]]:
        if patterns is None:
            patterns = ['*.pdf', '*.docx', '*.txt', '*.json']

        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        file_paths = []
        for pattern in patterns:
            if recursive:
                search_pattern = f"**/{pattern}"
            else:
                search_pattern = pattern

            matched_files = glob.glob(str(dir_path / search_pattern), recursive=recursive)
            file_paths.extend(matched_files)

        file_paths = sorted(set(file_paths))

        if not file_paths:
            if verbose:
                print(f"[WARNING] No files found in {directory}")
            return []

        return self.parse_batch(file_paths, verbose=verbose)

    def save_to_jsonl(self, results: List[Dict[str, Any]], output_file: str,
                     include_metadata: bool = True, verbose: bool = True):
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                if not result['success']:
                    continue

                if include_metadata:
                    record = result
                else:
                    record = {
                        'doc_id': result['file_name'],
                        'text': result['text'],
                        'word_count': result['word_count'],
                        'processing_time': result['extraction_time']
                    }

                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        if verbose:
            print(f"[INFO] Saved {len([r for r in results if r['success']])} documents to {output_file}")

    def process_and_save(self, directory: str, output_file: str,
                        patterns: Optional[List[str]] = None,
                        recursive: bool = True,
                        include_metadata: bool = True,
                        verbose: bool = True):
        results = self.parse_directory(directory, patterns, recursive, verbose)
        if results:
            self.save_to_jsonl(results, output_file, include_metadata, verbose)

        return results
