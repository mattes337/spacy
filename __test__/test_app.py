import os
import tempfile
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
import uvicorn

app = FastAPI(title="Audio/Video Transcription Service - Test Version")

class MockTextProcessor:
    """Mock transcription processor for testing agent integration without ML dependencies"""
    
    def __init__(self):
        print("MockTextProcessor initialized")

    def extract_audio_from_video(self, video_path: str) -> str:
        """Mock audio extraction"""
        audio_path = video_path.replace('.mp4', '.wav').replace('.avi', '.wav')
        # Create a dummy audio file for testing
        with open(audio_path, 'w') as f:
            f.write("mock audio data")
        return audio_path

    def transcribe_with_whisper(self, audio_path: str, language: str = None) -> Dict[str, Any]:
        """Mock transcription with language detection"""
        transcript = "This is a mock transcription of the audio file."

        # Mock language detection
        if language is None:
            detected_language = "en"
            language_info = {
                "detected_language": "en",
                "language_confidence": 0.95,
                "language_probabilities": {"en": 0.95, "es": 0.03, "fr": 0.02},
                "language_source": "auto_detected"
            }
        else:
            language_info = {
                "detected_language": language,
                "language_confidence": 1.0,
                "language_probabilities": {language: 1.0},
                "language_source": "manually_specified"
            }

        return {
            "transcript": transcript,
            "language_info": language_info,
            "whisper_result": {
                "language": language_info["detected_language"],
                "segments": []
            }
        }

    def structure_text_with_spacy(self, text: str) -> Dict[str, Any]:
        """Mock text structuring"""
        words = text.split()
        structured_data = {
            "raw_text": text,
            "sentences": [text],  # Mock: treat entire text as one sentence
            "entities": [
                {
                    "text": "mock",
                    "label": "MISC",
                    "description": "Miscellaneous entities",
                    "start": 0,
                    "end": 4
                }
            ],
            "keywords": [
                {
                    "text": word,
                    "lemma": word.lower(),
                    "pos": "NOUN",
                    "is_stop": False
                }
                for word in words if len(word) > 2
            ],
            "noun_phrases": ["mock transcription", "audio file"],
            "summary_stats": {
                "total_tokens": len(words),
                "sentences_count": 1,
                "entities_count": 1,
                "unique_entities": 1
            }
        }
        return structured_data

processor = MockTextProcessor()

@app.post("/process-audio")
async def process_audio(
    file: UploadFile = File(...),
    language: str = Query(None, description="Language code for transcription (e.g., 'en', 'es', 'fr'). If not specified, language will be auto-detected.")
):
    """Process audio file and extract structured text with language detection"""
    if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    # Create temporary file
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
    try:
        content = await file.read()
        tmp_file.write(content)
        tmp_file.flush()
        tmp_file.close()  # Close file before processing

        # Mock transcribe audio with language detection
        transcription_result = processor.transcribe_with_whisper(tmp_file.name, language=language)
        transcript = transcription_result["transcript"]
        language_info = transcription_result["language_info"]

        # Mock structure with spaCy
        structured_data = processor.structure_text_with_spacy(transcript)
        structured_data["source_type"] = "audio"
        structured_data["filename"] = file.filename
        structured_data["language_detection"] = language_info
        structured_data["whisper_segments"] = transcription_result["whisper_result"]["segments"]

        return structured_data

    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_file.name)
        except (OSError, PermissionError):
            pass  # Ignore cleanup errors

@app.post("/process-video")
async def process_video(
    file: UploadFile = File(...),
    language: str = Query(None, description="Language code for transcription (e.g., 'en', 'es', 'fr'). If not specified, language will be auto-detected.")
):
    """Process video file and extract structured text from audio with language detection"""
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Unsupported video format")

    # Create temporary file
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
    audio_path = None
    try:
        content = await file.read()
        tmp_file.write(content)
        tmp_file.flush()
        tmp_file.close()  # Close file before processing

        # Mock extract audio from video
        audio_path = processor.extract_audio_from_video(tmp_file.name)

        # Mock transcribe audio with language detection
        transcription_result = processor.transcribe_with_whisper(audio_path, language=language)
        transcript = transcription_result["transcript"]
        language_info = transcription_result["language_info"]

        # Mock structure with spaCy
        structured_data = processor.structure_text_with_spacy(transcript)
        structured_data["source_type"] = "video"
        structured_data["filename"] = file.filename
        structured_data["language_detection"] = language_info
        structured_data["whisper_segments"] = transcription_result["whisper_result"]["segments"]

        return structured_data

    finally:
        # Clean up temporary files
        try:
            os.unlink(tmp_file.name)
        except (OSError, PermissionError):
            pass
        if audio_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except (OSError, PermissionError):
                pass

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": False, "mode": "mock"}

@app.get("/")
async def root():
    return {"message": "Audio/Video Transcription Service API", "version": "test", "mode": "mock", "endpoints": ["/process-audio", "/process-video", "/health"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
