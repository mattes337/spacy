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
