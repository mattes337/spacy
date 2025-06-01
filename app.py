import os
import tempfile
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import spacy
import speech_recognition as sr
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
import whisper
import uvicorn

from config import config

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOG_FILE) if config.LOG_FILE else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

# Print configuration if debug mode
if config.DEBUG:
    config.print_config()

app = FastAPI(
    title="Audio/Video Text Extractor",
    description="Transcription service for audio and video files with NLP analysis",
    version="1.0.0",
    debug=config.DEBUG
)

# Add CORS middleware if origins are configured
if config.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Load models with configuration
logger.info(f"Loading spaCy model: {config.SPACY_MODEL}")
try:
    nlp = spacy.load(config.SPACY_MODEL)
    logger.info("spaCy model loaded successfully")
except OSError as e:
    logger.error(f"Failed to load spaCy model {config.SPACY_MODEL}: {e}")
    raise

logger.info(f"Loading Whisper model: {config.WHISPER_MODEL}")
try:
    whisper_model = whisper.load_model(config.WHISPER_MODEL)
    logger.info("Whisper model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Whisper model {config.WHISPER_MODEL}: {e}")
    raise

class TextProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.max_file_size = config.get_max_file_size_bytes()
        logger.info(f"TextProcessor initialized with max file size: {config.MAX_FILE_SIZE}")

    def extract_audio_from_video(self, video_path: str) -> str:
        """Extract audio from video file"""
        try:
            logger.info(f"Extracting audio from video: {video_path}")
            video = VideoFileClip(video_path)

            # Create audio file in temp directory
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            audio_path = os.path.join(config.TEMP_DIR, f"{base_name}_audio.wav")

            video.audio.write_audiofile(audio_path, verbose=False, logger=None)
            video.close()
            logger.info(f"Audio extracted successfully: {audio_path}")
            return audio_path
        except Exception as e:
            logger.error(f"Video processing error: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Video processing error: {str(e)}")

    def transcribe_with_whisper(self, audio_path: str, language: str = None) -> Dict[str, Any]:
        """Transcribe audio using Whisper with language detection"""
        try:
            logger.info(f"Transcribing audio with Whisper model: {config.WHISPER_MODEL}")

            # Load and preprocess audio for language detection
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio, n_mels=whisper_model.dims.n_mels).to(whisper_model.device)

            # Detect language if not specified
            language_info = {}
            if language is None:
                logger.info("Detecting language...")
                _, probs = whisper_model.detect_language(mel)
                detected_language = max(probs, key=probs.get)
                language_confidence = probs[detected_language]

                language_info = {
                    "detected_language": detected_language,
                    "language_confidence": float(language_confidence),
                    "language_probabilities": {lang: float(prob) for lang, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]},
                    "language_source": "auto_detected"
                }

                # Use detected language for transcription
                transcribe_language = detected_language
                logger.info(f"Detected language: {detected_language} (confidence: {language_confidence:.3f})")
            else:
                language_info = {
                    "detected_language": language,
                    "language_confidence": 1.0,
                    "language_probabilities": {language: 1.0},
                    "language_source": "manually_specified"
                }
                transcribe_language = language
                logger.info(f"Using manually specified language: {language}")

            # Transcribe with detected or specified language
            result = whisper_model.transcribe(audio_path, language=transcribe_language)
            transcript = result["text"].strip()

            logger.info(f"Transcription completed, length: {len(transcript)} characters")

            return {
                "transcript": transcript,
                "language_info": language_info,
                "whisper_result": {
                    "language": result.get("language", transcribe_language),
                    "segments": result.get("segments", [])
                }
            }

        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Transcription error: {str(e)}")

    def validate_file_size(self, file_size: int) -> None:
        """Validate file size against configured maximum"""
        if file_size > self.max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"File size ({file_size} bytes) exceeds maximum allowed size ({config.MAX_FILE_SIZE})"
            )

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
async def process_audio(
    file: UploadFile = File(...),
    language: str = Query(None, description="Language code for transcription (e.g., 'en', 'es', 'fr'). If not specified, language will be auto-detected.")
):
    """Process audio file and extract structured text with language detection"""
    logger.info(f"Processing audio file: {file.filename}")

    # Validate file format
    supported_formats = config.get_supported_audio_formats()
    if not any(file.filename.lower().endswith(fmt) for fmt in supported_formats):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format. Supported formats: {', '.join(supported_formats)}"
        )

    # Read file content and validate size
    content = await file.read()
    processor.validate_file_size(len(content))

    # Create temporary file in configured temp directory
    tmp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=os.path.splitext(file.filename)[1],
        dir=config.TEMP_DIR
    )
    try:
        tmp_file.write(content)
        tmp_file.flush()
        tmp_file.close()  # Close file before processing

        # Transcribe audio with language detection
        transcription_result = processor.transcribe_with_whisper(tmp_file.name, language=language)
        transcript = transcription_result["transcript"]
        language_info = transcription_result["language_info"]

        # Structure with spaCy
        structured_data = processor.structure_text_with_spacy(transcript)
        structured_data["source_type"] = "audio"
        structured_data["filename"] = file.filename
        structured_data["file_size"] = len(content)
        structured_data["language_detection"] = language_info
        structured_data["whisper_segments"] = transcription_result["whisper_result"]["segments"]
        structured_data["models_used"] = {
            "whisper": config.WHISPER_MODEL,
            "spacy": config.SPACY_MODEL
        }

        logger.info(f"Audio processing completed for: {file.filename}")
        return structured_data

    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_file.name)
        except (OSError, PermissionError):
            logger.warning(f"Failed to cleanup temporary file: {tmp_file.name}")

@app.post("/process-video")
async def process_video(
    file: UploadFile = File(...),
    language: str = Query(None, description="Language code for transcription (e.g., 'en', 'es', 'fr'). If not specified, language will be auto-detected.")
):
    """Process video file and extract structured text from audio with language detection"""
    logger.info(f"Processing video file: {file.filename}")

    # Validate file format
    supported_formats = config.get_supported_video_formats()
    if not any(file.filename.lower().endswith(fmt) for fmt in supported_formats):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format. Supported formats: {', '.join(supported_formats)}"
        )

    # Read file content and validate size
    content = await file.read()
    processor.validate_file_size(len(content))

    # Create temporary file in configured temp directory
    tmp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=os.path.splitext(file.filename)[1],
        dir=config.TEMP_DIR
    )
    audio_path = None
    try:
        tmp_file.write(content)
        tmp_file.flush()
        tmp_file.close()  # Close file before processing

        # Extract audio from video
        audio_path = processor.extract_audio_from_video(tmp_file.name)

        # Transcribe audio with language detection
        transcription_result = processor.transcribe_with_whisper(audio_path, language=language)
        transcript = transcription_result["transcript"]
        language_info = transcription_result["language_info"]

        # Structure with spaCy
        structured_data = processor.structure_text_with_spacy(transcript)
        structured_data["source_type"] = "video"
        structured_data["filename"] = file.filename
        structured_data["file_size"] = len(content)
        structured_data["language_detection"] = language_info
        structured_data["whisper_segments"] = transcription_result["whisper_result"]["segments"]
        structured_data["models_used"] = {
            "whisper": config.WHISPER_MODEL,
            "spacy": config.SPACY_MODEL
        }

        logger.info(f"Video processing completed for: {file.filename}")
        return structured_data

    finally:
        # Clean up temporary files
        try:
            os.unlink(tmp_file.name)
        except (OSError, PermissionError):
            logger.warning(f"Failed to cleanup temporary video file: {tmp_file.name}")
        if audio_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except (OSError, PermissionError):
                logger.warning(f"Failed to cleanup temporary audio file: {audio_path}")

@app.get("/health")
async def health_check():
    """Health check endpoint with configuration information"""
    return {
        "status": "healthy",
        "models_loaded": True,
        "configuration": {
            "whisper_model": config.WHISPER_MODEL,
            "spacy_model": config.SPACY_MODEL,
            "max_file_size": config.MAX_FILE_SIZE,
            "supported_audio_formats": config.get_supported_audio_formats(),
            "supported_video_formats": config.get_supported_video_formats(),
            "test_mode": config.TEST_MODE
        }
    }

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "message": "Audio/Video Transcription Service API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "/health",
            "/config",
            "/process-audio",
            "/process-video"
        ],
        "documentation": "/docs"
    }

@app.get("/config")
async def get_config():
    """Get current service configuration"""
    return {
        "whisper_model": config.WHISPER_MODEL,
        "spacy_model": config.SPACY_MODEL,
        "max_file_size": config.MAX_FILE_SIZE,
        "temp_dir": config.TEMP_DIR,
        "debug": config.DEBUG,
        "log_level": config.LOG_LEVEL,
        "supported_formats": {
            "audio": config.get_supported_audio_formats(),
            "video": config.get_supported_video_formats()
        }
    }

if __name__ == "__main__":
    logger.info(f"Starting transcription service on {config.HOST}:{config.PORT}")
    if config.RELOAD:
        # For development with reload
        uvicorn.run(
            "app:app",  # Use import string for reload
            host=config.HOST,
            port=config.PORT,
            reload=True,
            log_level=config.LOG_LEVEL.lower()
        )
    else:
        # For production - use import string to enable workers
        if config.MAX_WORKERS > 1:
            uvicorn.run(
                "app:app",  # Use import string for workers
                host=config.HOST,
                port=config.PORT,
                log_level=config.LOG_LEVEL.lower(),
                workers=config.MAX_WORKERS
            )
        else:
            # Single worker mode
            uvicorn.run(
                app,
                host=config.HOST,
                port=config.PORT,
                log_level=config.LOG_LEVEL.lower()
            )
