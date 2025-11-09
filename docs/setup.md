# Initial Setup Commands

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create directory structure
python scripts/prepare_data.py

# 4. Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# 5. Place your downloaded Kaggle datasets in 'downloads' folder
# Then run the data preparation script again
python scripts/prepare_data.py

# 6. Run the application
python src/api/routes.py

# 7. Access the UI
# Open browser to http://localhost:5000

# 8. Run tests
python test_milestone.py
```