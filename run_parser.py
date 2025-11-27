from src.ingestion.parser import DocumentParser
import os

def main():
    parser = DocumentParser()

    input_dir = 'data/raw'
    output_file = 'data/cleaned/corpus.jsonl'

    os.makedirs(input_dir, exist_ok=True)

    results = parser.process_and_save(
        directory=input_dir,
        output_file=output_file,
        recursive=True,
        include_metadata=False,
        verbose=True
    )

    if results:
        report = parser.get_accuracy_report()
        print(f"\nProcessed: {report['total_processed']} documents")
        print(f"Success: {report['successful']}")
        print(f"Failed: {report['failed']}")

if __name__ == "__main__":
    main()
