# Audio/Video Text Extractor

A FastAPI-based service for extracting and structuring text from audio and video files using Whisper, spaCy, and other ML libraries. Designed for integration with vector databases and agentic coding systems.

## 🚀 Current Status

✅ **Working Components:**
- FastAPI application structure
- File upload endpoints for audio and video
- Mock processing pipeline for testing
- Vector database client for data preparation
- Docker configuration
- Comprehensive error handling

⚠️ **Known Issues:**
- ML dependencies (spaCy, Whisper) have installation issues on Python 3.13/Windows
- Requires Python 3.9-3.11 for full functionality

## 📁 Project Structure

```
├── app.py                 # Main FastAPI application
├── test_app.py           # Mock version for testing
├── vector_db_client.py   # Vector database integration client
├── test_vector_client.py # Test script for vector client
├── requirements.txt      # Python dependencies
├── Dockerfile           # Docker configuration
├── IMPLEMENTATION.md    # Detailed implementation guide
├── TASKS.md            # TODOs and missing features
└── README.md           # This file
```

## 🛠️ Installation

### Option 1: Using Python 3.9-3.11 (Recommended)

```bash
# Create virtual environment with Python 3.9-3.11
python3.9 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Option 2: Using Docker

```bash
# Build the Docker image
docker build -t audio-video-indexer .

# Run the container
docker run -p 8000:8000 audio-video-indexer
```

### Option 3: Testing Mode (Current Environment)

```bash
# Install minimal dependencies for testing
pip install fastapi uvicorn python-multipart

# Run the mock version
python test_app.py
```

## 🚀 Usage

### Starting the Server

```bash
# Production version (requires ML dependencies)
python app.py

# Test version (mock processing)
python test_app.py
```

The server will start on `http://localhost:8000`

### API Endpoints

#### Process Audio File
```bash
curl -X POST "http://localhost:8000/process-audio" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_audio.wav"
```

#### Process Video File
```bash
curl -X POST "http://localhost:8000/process-video" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_video.mp4"
```

#### Health Check
```bash
curl http://localhost:8000/health
```

### Response Format

```json
{
  "raw_text": "Transcribed text from audio/video",
  "sentences": ["Sentence 1", "Sentence 2"],
  "entities": [
    {
      "text": "Entity text",
      "label": "PERSON",
      "description": "People, including fictional",
      "start": 10,
      "end": 20
    }
  ],
  "keywords": [
    {
      "text": "keyword",
      "lemma": "keyword",
      "pos": "NOUN",
      "is_stop": false
    }
  ],
  "noun_phrases": ["noun phrase 1", "noun phrase 2"],
  "summary_stats": {
    "total_tokens": 50,
    "sentences_count": 2,
    "entities_count": 1,
    "unique_entities": 1
  },
  "source_type": "audio",
  "filename": "your_audio.wav"
}
```

## 🧪 Testing

### Test Vector Database Client
```bash
python test_vector_client.py
```

### Test API Endpoints
```bash
# Start test server
python test_app.py

# In another terminal, test endpoints
curl http://localhost:8000/health
curl -X POST "http://localhost:8000/process-audio" -F "file=@test_audio.wav"
```

## 🔧 Configuration

### Supported File Formats

**Audio:** `.wav`, `.mp3`, `.m4a`, `.flac`
**Video:** `.mp4`, `.avi`, `.mov`, `.mkv`

### Environment Variables (Future)
- `WHISPER_MODEL`: Whisper model size (base, small, medium, large)
- `SPACY_MODEL`: spaCy model name
- `MAX_FILE_SIZE`: Maximum upload file size
- `TEMP_DIR`: Temporary file storage directory

## 🐳 Docker Deployment

```bash
# Build and run
docker build -t audio-video-indexer .
docker run -p 8000:8000 audio-video-indexer

# With environment variables
docker run -p 8000:8000 -e WHISPER_MODEL=small audio-video-indexer
```

## 🔗 Vector Database Integration

The `VectorDBClient` prepares structured data for vector database indexing:

```python
from vector_db_client import VectorDBClient

client = VectorDBClient()
documents = client.prepare_for_indexing(structured_data)

# Documents are ready for indexing in your vector database
# Each document has: id, text, metadata
```

## 📋 Next Steps

See `TASKS.md` for detailed TODOs and missing features.

**High Priority:**
1. Fix dependency compatibility issues
2. Implement real model loading
3. Add comprehensive error handling
4. Optimize performance for large files

## 🤝 Contributing

1. Check `TASKS.md` for open tasks
2. Test with Python 3.9-3.11 environment
3. Ensure all tests pass before submitting changes
4. Update documentation as needed

## 📄 License

[Add your license here]

## 🆘 Troubleshooting

### Common Issues

1. **Dependency installation fails**
   - Use Python 3.9-3.11 instead of 3.13
   - Consider using conda for ML packages

2. **File permission errors on Windows**
   - Fixed in current version with proper file cleanup

3. **Models not loading**
   - Ensure spaCy model is downloaded: `python -m spacy download en_core_web_sm`
   - Check internet connection for Whisper model download

4. **Large file processing fails**
   - Implement chunking for large files (see TASKS.md)
   - Increase timeout settings
