# Data Directory

This directory contains all data files for the Document Analytics project.

## Directory Structure

- `raw/` - Original downloaded datasets (not in Git)
- `processed/` - Processed and parsed documents
- `embeddings/` - Vector embeddings and indices
- `temp/` - Temporary processing files

## Obtaining Data

The datasets used in this project are available from Kaggle:

1. [Company Documents Dataset](https://www.kaggle.com/datasets/ayoubcherguelaine/company-documents-dataset) (486 MB)
2. [Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) (365 MB)
3. [Dataset of PDF Files](https://www.kaggle.com/datasets/manisha717/dataset-of-pdf-files) (sample 500 docs)
4. [Regulations PDF](https://www.kaggle.com/datasets/terryeppler/regulations-pdf) (sample 100 docs)

Download these datasets and place them in the `downloads/` folder, then run:
```bash
python scripts/prepare_data.py
```