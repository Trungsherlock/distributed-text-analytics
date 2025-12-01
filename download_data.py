import kagglehub
import shutil
from pathlib import Path

def download_kaggle_dataset(dataset_name, dataset_path=None, file_types=None):
    if file_types is None:
        file_types = ['*.pdf', '*.docx', '*.txt', '*.json']

    if dataset_path is None:
        dataset_path = dataset_name.split('/')[-1]

    path = kagglehub.dataset_download(dataset_name)
    print(f"Downloaded to: {path}")

    source_dir = Path(path)
    target_dir = Path(f'data/raw/{dataset_path}')
    target_dir.mkdir(parents=True, exist_ok=True)

    all_files = []
    for pattern in file_types:
        files = list(source_dir.rglob(pattern))
        all_files.extend(files)

    print(f"Found {len(all_files)} files")

    for file in all_files:
        target_file = target_dir / file.name
        shutil.copy2(file, target_file)
        print(f"Copied: {file.name}")

    print(f"\nAll files copied to: {target_dir}")
    return target_dir

if __name__ == "__main__":
    DATASET_NAME = "snehaanbhawal/resume-dataset"
    DATASET_PATH = "resumes"
    download_kaggle_dataset(DATASET_NAME, DATASET_PATH)
