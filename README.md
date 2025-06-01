# Audio/Video Transcription Service

A focused FastAPI-based microservice that transcribes audio and video files and returns structured JSON data. This service is designed to be called by AI agents and other automated systems that need reliable audio/video transcription capabilities.

## 🎯 Service Scope

**This service has ONE focused responsibility:**
- **Input**: Audio files (.wav, .mp3, .m4a, .flac) or Video files (.mp4, .avi, .mov, .mkv)
- **Output**: Structured JSON with transcribed text, entities, keywords, and metadata
- **Usage**: Called by AI agents and automated systems via REST API

## 🚀 Current Status

✅ **Working Components:**
- FastAPI application with transcription endpoints
- File upload handling for audio and video formats
- Mock processing pipeline for testing environments
- Structured JSON output with comprehensive text analysis
- Docker configuration for containerized deployment
- Comprehensive error handling and file cleanup

⚠️ **Known Issues:**
- ML dependencies (spaCy, Whisper) have installation issues on Python 3.13/Windows
- Requires Python 3.10-3.11 for full functionality (updated for compatibility)

## 📁 Project Structure

```
├── app.py                    # Main FastAPI transcription service
├── vector_db_client.py      # Data preparation utilities
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker configuration
├── IMPLEMENTATION.md       # Detailed implementation guide
├── TASKS.md               # Development tasks and roadmap
├── README.md              # This file
└── __test__/              # Test files and mock implementations
    ├── test_app.py        # Mock version for testing
    └── test_vector_client.py # Test script for data preparation
```

## 🛠️ Installation

### Option 1: Using Python 3.10-3.11 (Recommended)

```bash
# Create virtual environment with Python 3.10-3.11
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Option 2: Using Docker

```bash
# Build the Docker image
docker build -t audio-video-transcription .

# Run the container
docker run -p 8000:8000 audio-video-transcription
```

### Option 3: Testing Mode (Current Environment)

```bash
# Install minimal dependencies for testing
pip install fastapi uvicorn python-multipart

# Run the mock version
python __test__/test_app.py
```

## 🚀 Usage

### Starting the Transcription Service

```bash
# Production version (requires ML dependencies)
python app.py

# Test version (mock processing)
python __test__/test_app.py
```

The service will start on `http://localhost:8000` and be ready to receive transcription requests from agents.

### API Endpoints for Agent Integration

#### Transcribe Audio File
```bash
curl -X POST "http://localhost:8000/process-audio" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_audio.wav"
```

#### Transcribe Video File (extracts audio first)
```bash
curl -X POST "http://localhost:8000/process-video" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_video.mp4"
```

#### Service Health Check
```bash
curl http://localhost:8000/health
```

### Structured JSON Response

The service returns comprehensive structured data that agents can easily process:

```json
{
  "raw_text": "Complete transcribed text from audio/video",
  "sentences": ["Sentence 1", "Sentence 2", "..."],
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

### Test Data Preparation Utilities
```bash
python __test__/test_vector_client.py
```

### Test Transcription API
```bash
# Start test server
python __test__/test_app.py

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
# Build and run the transcription service
docker build -t audio-video-transcription .
docker run -p 8000:8000 audio-video-transcription

# With environment variables for model configuration
docker run -p 8000:8000 -e WHISPER_MODEL=small audio-video-transcription
```

## 🔗 Agent Integration

The service is designed for seamless integration with AI agents:

```python
import requests

# Agent calls the transcription service
response = requests.post(
    "http://localhost:8000/process-audio",
    files={"file": open("audio.wav", "rb")}
)

structured_data = response.json()
# Agent can now process the structured transcription data
```

## 📋 Development Roadmap

See `TASKS.md` for detailed development tasks and priorities.

**High Priority for Transcription Service:**
1. ✅ Fixed dependency compatibility issues (Python 3.10-3.11)
2. Implement robust model loading and caching
3. Optimize transcription accuracy and performance
4. Add comprehensive error handling for agent integration

## 🤝 Contributing

1. Check `TASKS.md` for transcription service development tasks
2. Test with Python 3.10-3.11 environment for ML dependencies
3. Ensure transcription accuracy and API reliability
4. Update documentation to reflect service scope

## 📄 License

[Add your license here]

## 🆘 Troubleshooting

### Common Transcription Service Issues

1. **ML dependency installation fails**
   - Use Python 3.9-3.11 instead of 3.13
   - Consider using conda for ML packages
   - Use Docker for consistent environment

2. **File permission errors on Windows**
   - Fixed in current version with proper file cleanup
   - Ensure temp directory has write permissions

3. **Transcription models not loading**
   - Ensure spaCy model is downloaded: `python -m spacy download en_core_web_sm`
   - Check internet connection for Whisper model download
   - Verify model files are accessible

4. **Large audio/video file processing fails**
   - Implement chunking for large files (see TASKS.md)
   - Increase timeout settings for agent requests
   - Monitor memory usage during transcription
