### Implementation Details: Audio/Video Transcription Service

This document outlines the implementation details for a focused audio and video transcription microservice, designed to be called by AI agents and automated systems. The service uses Python, FastAPI, spaCy, and Whisper to transcribe audio/video files and return structured JSON data for agent consumption.

#### 1. Project Structure

The transcription service consists of the following core files:

-   `app.py`: Main FastAPI transcription service with audio/video processing endpoints
-   `requirements.txt`: Python dependencies for transcription and NLP processing
-   `Dockerfile`: Container configuration for deployment in agent systems
-   `vector_db_client.py`: Data preparation utilities for structured output
-   `__test__/`: Test files and mock implementations for development
    -   `test_app.py`: Mock transcription service for testing agent integration
    -   `test_vector_client.py`: Test script for data preparation utilities

#### 2. Dockerfile

The `Dockerfile` sets up the containerized transcription service for agent deployment:

```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "app.py"]
```

-   Starts from a slim Python 3.9 image for ML compatibility
-   Installs `ffmpeg` and `libsndfile1` for audio/video transcription processing
-   Sets the working directory to `/app` for the transcription service
-   Installs Python dependencies required for transcription (spaCy, Whisper, etc.)
-   Downloads the `en_core_web_sm` spaCy model for text analysis
-   Copies the transcription service code
-   Exposes port 8000 for agent API access
-   Runs the transcription service using `python app.py`

#### 3. requirements.txt

This file lists the Python dependencies required for the transcription service:

```txt
spacy==3.7.2
speechrecognition==3.10.0
pydub==0.25.1
moviepy==1.0.3
whisper-openai==20231117
torch==2.1.0
torchaudio==2.1.0
fastapi==0.104.1
uvicorn==0.24.0
numpy==1.24.3
```

-   `spacy`: Natural language processing for text analysis and entity extraction
-   `speechrecognition`: Speech-to-text conversion (backup to Whisper)
-   `pydub`: Audio format handling and manipulation
-   `moviepy`: Video processing and audio extraction
-   `whisper-openai`: Primary transcription engine for high accuracy
-   `torch` and `torchaudio`: PyTorch dependencies required by Whisper
-   `fastapi`: REST API framework for agent integration
-   `uvicorn`: ASGI server for running the transcription service
-   `numpy`: Numerical operations for audio processing

#### 4. app.py

This file contains the core transcription service logic for agent integration:

import os
import tempfile
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
import spacy
import speech_recognition as sr
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
import whisper
import uvicorn

app = FastAPI(title="Audio/Video Text Extractor")

# Load models
nlp = spacy.load("en_core_web_sm")
whisper_model = whisper.load_model("base")

class TextProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def extract_audio_from_video(self, video_path: str) -> str:
        """Extract audio from video file"""
        try:
            video = VideoFileClip(video_path)
            audio_path = video_path.replace('.mp4', '.wav').replace('.avi', '.wav')
            video.audio.write_audiofile(audio_path, verbose=False, logger=None)
            video.close()
            return audio_path
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Video processing error: {str(e)}")

    def transcribe_with_whisper(self, audio_path: str) -> str:
        """Transcribe audio using Whisper"""
        try:
            result = whisper_model.transcribe(audio_path)
            return result["text"]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Transcription error: {str(e)}")

    def structure_text_with_spacy(self, text: str) -> Dict[str, Any]:
        """Process text with spaCy for structured extraction"""
        doc = nlp(text)

        # Extract structured information
        structured_data = {
            "raw_text": text,
            "sentences": [sent.text.strip() for sent in doc.sents],
            "entities": [
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "description": spacy.explain(ent.label_),
                    "start": ent.start_char,
                    "end": ent.end_char
                }
                for ent in doc.ents
            ],
            "keywords": [
                {
                    "text": token.text,
                    "lemma": token.lemma_,
                    "pos": token.pos_,
                    "is_stop": token.is_stop
                }
                for token in doc
                if not token.is_stop and not token.is_punct and len(token.text) > 2
            ],
            "noun_phrases": [chunk.text for chunk in doc.noun_chunks],
            "summary_stats": {
                "total_tokens": len(doc),
                "sentences_count": len(list(doc.sents)),
                "entities_count": len(doc.ents),
                "unique_entities": len(set([ent.label_ for ent in doc.ents]))
            }
        }

        return structured_data

processor = TextProcessor()

@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):
    """Process audio file and extract structured text"""
    if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file.flush()

        try:
            # Transcribe audio
            transcript = processor.transcribe_with_whisper(tmp_file.name)

            # Structure with spaCy
            structured_data = processor.structure_text_with_spacy(transcript)
            structured_data["source_type"] = "audio"
            structured_data["filename"] = file.filename

            return structured_data

        finally:
            os.unlink(tmp_file.name)

@app.post("/process-video")
async def process_video(file: UploadFile = File(...)):
    """Process video file and extract structured text from audio"""
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Unsupported video format")

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file.flush()

        audio_path = None
        try:
            # Extract audio from video
            audio_path = processor.extract_audio_from_video(tmp_file.name)

            # Transcribe audio
            transcript = processor.transcribe_with_whisper(audio_path)

            # Structure with spaCy
            structured_data = processor.structure_text_with_spacy(transcript)
            structured_data["source_type"] = "video"
            structured_data["filename"] = file.filename

            return structured_data

        finally:
            os.unlink(tmp_file.name)
            if audio_path and os.path.exists(audio_path):
                os.unlink(audio_path)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": True}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

Key transcription service components:

-   **Imports**: Core libraries for transcription, NLP, and API functionality
-   **FastAPI App**: REST API service configured for agent integration
-   **Model Loading**: spaCy's `en_core_web_sm` and Whisper's `base` model for transcription
-   **TextProcessor Class**: Core transcription logic
    -   `extract_audio_from_video`: Extracts audio tracks from video files using MoviePy
    -   `transcribe_with_whisper`: High-accuracy transcription using OpenAI Whisper
    -   `structure_text_with_spacy`: NLP analysis for structured JSON output
-   **Agent API Endpoints**:
    -   `/process-audio`: Transcribes audio files and returns structured JSON
    -   `/process-video`: Extracts audio from video, transcribes, and returns structured JSON
    -   `/health`: Service health check for agent dependency monitoring
-   **Service Execution**: Runs the transcription service using Uvicorn ASGI server

#### 5. vector\_db\_client.py

This utility prepares the structured transcription data for further processing by agent systems:

import json
from typing import List, Dict, Any
import numpy as np

class VectorDBClient:
    """Example client for vector database integration"""

    def prepare_for_indexing(self, structured_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare structured data for vector database indexing"""
        documents = []

        # Index full text
        documents.append({
            "id": f"{structured_data['filename']}_full",
            "text": structured_data["raw_text"],
            "metadata": {
                "source": structured_data["filename"],
                "type": structured_data["source_type"],
                "content_type": "full_transcript"
            }
        })

        # Index sentences separately
        for i, sentence in enumerate(structured_data["sentences"]):
            documents.append({
                "id": f"{structured_data['filename']}_sent_{i}",
                "text": sentence,
                "metadata": {
                    "source": structured_data["filename"],
                    "type": structured_data["source_type"],
                    "content_type": "sentence",
                    "sentence_index": i
                }
            })

        # Index entities
        for entity in structured_data["entities"]:
            documents.append({
                "id": f"{structured_data['filename']}_entity_{entity['start']}",
                "text": entity["text"],
                "metadata": {
                    "source": structured_data["filename"],
                    "type": structured_data["source_type"],
                    "content_type": "entity",
                    "entity_label": entity["label"],
                    "entity_description": entity["description"]
                }
            })

        return documents

-   The `VectorDBClient` class provides data preparation utilities for agent systems
-   The `prepare_for_indexing` method structures transcription data into documents
-   Creates separate documents for full transcripts, sentences, and entities with metadata
-   Enables agents to process transcription data at different granularity levels

#### 6. Agent Integration Usage

1.  **Deploy the transcription service:**

    ```bash
    docker build -t audio-video-transcription .
    docker run -p 8000:8000 audio-video-transcription
    ```

2.  **Agent API calls:**

    ```bash
    # Transcribe audio file
    curl -X POST "http://localhost:8000/process-audio" \
         -H "accept: application/json" \
         -H "Content-Type: multipart/form-data" \
         -F "file=@audio_file.wav"

    # Transcribe video file (extracts audio first)
    curl -X POST "http://localhost:8000/process-video" \
         -H "accept: application/json" \
         -H "Content-Type: multipart/form-data" \
         -F "file=@video_file.mp4"

    # Check service health
    curl http://localhost:8000/health
    ```

#### 7. Considerations for Agent Integration

-   **Reliable API Responses**: Ensure consistent JSON structure and error handling for agent consumption
-   **Scalability**: Design for multiple concurrent agent requests with async processing and queuing
-   **Model Management**: Implement robust model loading and caching for consistent transcription quality
-   **Agent Timeout Handling**: Handle long transcription processes with appropriate timeouts and status updates
-   **Monitoring**: Add structured logging and metrics for agent system monitoring and debugging
-   **Security**: Implement secure file handling and validation for agent-uploaded content
-   **Configuration**: Use environment variables for model selection and service configuration
-   **Service Health**: Provide detailed health checks for agent dependency management