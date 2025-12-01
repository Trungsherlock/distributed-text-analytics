# Dockerfile

FROM python:3.11.14-slim

# Install Java for Spark
RUN apt-get update && \
    apt-get install -y default-jdk && \
    apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Copy application code
COPY src/ ./src/
COPY data/ ./data/

# Set environment variables
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV SPARK_HOME=/usr/local/lib/python3.11/site-packages/pyspark
ENV PYTHONPATH=/app

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "src/api/routes.py"]