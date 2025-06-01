### Implementation Details: Audio/Video Text Extractor

This document outlines the implementation details for an audio and video text extraction service, designed to be used with an agentic coder. The service uses Python, FastAPI, spaCy, SpeechRecognition, pydub, MoviePy, and Whisper to extract text from audio and video files, structure it, and prepare it for indexing in a vector database.

#### 1. Project Structure

The project consists of the following files:

-   `Dockerfile`: Defines the Docker image for the service.
-   `requirements.txt`: Lists the Python dependencies.
-   `app.py`: Contains the FastAPI application code.
-   `vector_db_client.py` (optional): Example client for integrating with a vector database.

#### 2. Dockerfile

The `Dockerfile` sets up the environment for the application:

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

-   It starts from a slim Python 3.9 image.
-   Installs `ffmpeg` and `libsndfile1` for audio/video processing.
-   Sets the working directory to `/app`.
-   Copies and installs Python dependencies from `requirements.txt`.
-   Downloads the `en_core_web_sm` spaCy model.
-   Copies the application code.
-   Exposes port 8000.
-   Runs the `app.py` script using `python`.

#### 3. requirements.txt

This file lists the Python dependencies:

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

-   `spacy`: For natural language processing.
-   `speechrecognition`: For speech-to-text conversion (though Whisper is the primary).
-   `pydub`: For audio manipulation.
-   `moviepy`: For video processing.
-   `whisper-openai`: For audio transcription.
-   `torch` and `torchaudio`: Required by Whisper.
-   `fastapi`: For creating the API.
-   `uvicorn`: ASGI server for running the FastAPI application.
-   `numpy`: For numerical operations.

#### 4. app.py

This file contains the FastAPI application logic:

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

Key components:

-   **Imports**: Necessary libraries are imported.
-   **FastAPI App**: An instance of FastAPI is created.
-   **Model Loading**: spaCy's `en_core_web_sm` model and Whisper's `base` model are loaded.
-   **TextProcessor Class**:
    -   `extract_audio_from_video`: Extracts audio from a video file using MoviePy.
    -   `transcribe_with_whisper`: Transcribes audio using the Whisper model.
    -   `structure_text_with_spacy`: Processes text with spaCy to extract sentences, entities, keywords, and noun phrases.
-   **API Endpoints**:
    -   `/process-audio`: Accepts audio files, transcribes them, and structures the text.
    -   `/process-video`: Accepts video files, extracts audio, transcribes it, and structures the text.
    -   `/health`: A health check endpoint.
-   **Main Execution**: Runs the FastAPI application using Uvicorn.

#### 5. vector\_db\_client.py (Optional)

This file provides an example of how to prepare the structured data for indexing in a vector database:

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

-   The `VectorDBClient` class has a `prepare_for_indexing` method that takes the structured data and creates a list of documents suitable for indexing.
-   It indexes the full text, individual sentences, and entities, each with associated metadata.

#### 6. Usage

1.  **Build the Docker image:**

    ```bash
    docker build -t audio-video-indexer .
    ```

2.  **Run the container:**

    ```bash
    docker run -p 8000:8000 audio-video-indexer
    ```

3.  **Use the API:**

    ```bash
    # Process audio file
    curl -X POST "http://localhost:8000/process-audio" \
         -H "accept: application/json" \
         -H "Content-Type: multipart/form-data" \
         -F "file=@your_audio.wav"

    # Process video file
    curl -X POST "http://localhost:8000/process-video" \
         -H "accept: application/json" \
         -H "Content-Type: multipart/form-data" \
         -F "file=@your_video.mp4"
    ```

#### 7. Considerations for Agentic Coder

-   **Error Handling**: Ensure robust error handling, especially around file processing and API requests.
-   **Scalability**: Consider how to scale the service for large volumes of audio and video files.  This might involve asynchronous processing or distributed task queues.
-   **Model Management**:  Allow for easy updates to the spaCy and Whisper models.
-   **Vector Database Integration**: Implement the actual integration with a vector database, using the `vector_db_client.py` as a starting point.  Consider using environment variables for database credentials.
-   **Monitoring**: Add monitoring and logging to track the service's performance and identify issues.
-   **Security**: Implement appropriate security measures, especially when handling user-uploaded files.
-   **Configuration**: Use environment variables for configurable parameters like model paths, API keys, and database settings.