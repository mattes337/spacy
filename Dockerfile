FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/temp /app/logs /root/.cache/whisper /root/.cache/spacy

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy models (default and German for multi-language support)
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download de_core_news_sm

# Copy application code
COPY . .

# Set environment variables with defaults
ENV PYTHONUNBUFFERED=1
ENV WHISPER_MODEL=base
ENV SPACY_MODEL=en_core_web_sm
ENV MAX_FILE_SIZE=100MB
ENV TEMP_DIR=/app/temp
ENV LOG_LEVEL=INFO
ENV HOST=0.0.0.0
ENV PORT=8000

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "app.py"]
