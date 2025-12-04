from pathlib import Path

import pytest

from src.ingestion.parser import DocumentParser


TEST_DOC = Path("test_documents/doc_0.txt")


def test_parse_document_missing_file_raises():
    parser = DocumentParser()
    with pytest.raises(FileNotFoundError):
        parser.parse_document("data/raw/resumes/does_not_exist.pdf")


def test_parse_document_reads_txt_successfully():
    parser = DocumentParser()
    result = parser.parse_document(str(TEST_DOC))

    assert result["success"] is True
    assert result["format"] == ".txt"
    assert result["text"]
    assert result["word_count"] > 0
    assert result["file_name"] == TEST_DOC.name
