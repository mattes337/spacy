"""Background task processor for handling transcription jobs asynchronously."""

import asyncio
import logging
import tempfile
import os
import httpx
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from models import DatabaseManager, JobStatus, ProcessingJob
from config import config

logger = logging.getLogger(__name__)


class WebhookClient:
    """Client for sending webhook notifications."""
    
    def __init__(self):
        self.timeout = 30.0
        self.max_retries = 3
        self.retry_delay = 5.0
    
    async def send_webhook(self, webhook_url: str, job_data: Dict[str, Any]) -> bool:
        """Send webhook notification with retry logic."""
        if not webhook_url:
            return True
        
        payload = {
            "job_id": job_data.get("job_id"),
            "status": job_data.get("status"),
            "completed_at": datetime.utcnow().isoformat(),
            "result": job_data.get("result"),
            "error_message": job_data.get("error_message")
        }
        
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        webhook_url,
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if response.status_code < 400:
                        logger.info(f"Webhook sent successfully to {webhook_url} for job {job_data.get('job_id')}")
                        return True
                    else:
                        logger.warning(f"Webhook failed with status {response.status_code}: {response.text}")
                        
            except Exception as e:
                logger.error(f"Webhook attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        logger.error(f"All webhook attempts failed for {webhook_url}")
        return False


class BackgroundProcessor:
    """Background processor for handling transcription jobs."""
    
    def __init__(self, text_processor, db_manager: DatabaseManager):
        self.text_processor = text_processor
        self.db_manager = db_manager
        self.webhook_client = WebhookClient()
        self.executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)
        self.processing_jobs = set()
        
    async def process_job_async(self, job_id: str, file_content: bytes, file_extension: str):
        """Process a transcription job asynchronously."""
        logger.info(f"Starting background processing for job {job_id}")
        
        # Add to processing set
        self.processing_jobs.add(job_id)
        
        try:
            # Update status to processing
            self.db_manager.update_job_status(
                job_id, 
                JobStatus.PROCESSING, 
                progress=5, 
                current_step="Initializing"
            )
            
            # Get job details
            job = self.db_manager.get_job(job_id)
            if not job:
                raise Exception("Job not found")
            
            # Create temporary file
            tmp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=file_extension,
                dir=config.TEMP_DIR
            )
            
            try:
                # Write file content
                tmp_file.write(file_content)
                tmp_file.flush()
                tmp_file.close()
                
                # Update progress
                self.db_manager.update_job_status(
                    job_id, 
                    JobStatus.PROCESSING, 
                    progress=10, 
                    current_step="File prepared"
                )
                
                # Process based on file type
                if job.file_type == "video":
                    result = await self._process_video_job(job_id, tmp_file.name, job.language)
                else:
                    result = await self._process_audio_job(job_id, tmp_file.name, job.language)
                
                # Add job metadata to result
                result["job_id"] = job_id
                result["processing_completed_at"] = datetime.utcnow().isoformat()
                
                # Update job with result
                self.db_manager.update_job_result(job_id, result)
                
                logger.info(f"Job {job_id} completed successfully")
                
                # Send webhook if configured
                if job.webhook_url:
                    await self.webhook_client.send_webhook(
                        job.webhook_url,
                        {
                            "job_id": job_id,
                            "status": JobStatus.COMPLETED.value,
                            "result": result
                        }
                    )
                
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
                    
        except Exception as e:
            error_message = str(e)
            logger.error(f"Job {job_id} failed: {error_message}")
            
            # Update job status to failed
            self.db_manager.update_job_status(
                job_id, 
                JobStatus.FAILED, 
                progress=0, 
                current_step="Failed",
                error_message=error_message
            )
            
            # Send webhook for failure
            job = self.db_manager.get_job(job_id)
            if job and job.webhook_url:
                await self.webhook_client.send_webhook(
                    job.webhook_url,
                    {
                        "job_id": job_id,
                        "status": JobStatus.FAILED.value,
                        "error_message": error_message
                    }
                )
        
        finally:
            # Remove from processing set
            self.processing_jobs.discard(job_id)
    
    async def _process_audio_job(self, job_id: str, file_path: str, language: Optional[str]) -> Dict[str, Any]:
        """Process audio file in background thread."""
        self.db_manager.update_job_status(
            job_id, 
            JobStatus.PROCESSING, 
            progress=20, 
            current_step="Transcribing audio"
        )
        
        # Run transcription in thread pool
        loop = asyncio.get_event_loop()
        transcription_result = await loop.run_in_executor(
            self.executor,
            self.text_processor.transcribe_with_whisper,
            file_path,
            language
        )
        
        self.db_manager.update_job_status(
            job_id, 
            JobStatus.PROCESSING, 
            progress=70, 
            current_step="Processing with spaCy"
        )
        
        # Structure with spaCy
        structured_data = await loop.run_in_executor(
            self.executor,
            self.text_processor.structure_text_with_spacy,
            transcription_result["transcript"],
            transcription_result["whisper_result"]["segments"]
        )
        
        # Add metadata
        structured_data["source_type"] = "audio"
        structured_data["language_detection"] = transcription_result["language_info"]
        structured_data["whisper_segments"] = transcription_result["whisper_result"]["segments"]
        structured_data["models_used"] = {
            "whisper": config.WHISPER_MODEL,
            "spacy": config.SPACY_MODEL
        }
        
        self.db_manager.update_job_status(
            job_id, 
            JobStatus.PROCESSING, 
            progress=95, 
            current_step="Finalizing"
        )
        
        return structured_data
    
    async def _process_video_job(self, job_id: str, file_path: str, language: Optional[str]) -> Dict[str, Any]:
        """Process video file in background thread."""
        self.db_manager.update_job_status(
            job_id, 
            JobStatus.PROCESSING, 
            progress=15, 
            current_step="Extracting audio from video"
        )
        
        # Extract audio in thread pool
        loop = asyncio.get_event_loop()
        audio_path = await loop.run_in_executor(
            self.executor,
            self.text_processor.extract_audio_from_video,
            file_path
        )
        
        try:
            self.db_manager.update_job_status(
                job_id, 
                JobStatus.PROCESSING, 
                progress=30, 
                current_step="Transcribing audio"
            )
            
            # Transcribe audio
            transcription_result = await loop.run_in_executor(
                self.executor,
                self.text_processor.transcribe_with_whisper,
                audio_path,
                language
            )
            
            self.db_manager.update_job_status(
                job_id, 
                JobStatus.PROCESSING, 
                progress=75, 
                current_step="Processing with spaCy"
            )
            
            # Structure with spaCy
            structured_data = await loop.run_in_executor(
                self.executor,
                self.text_processor.structure_text_with_spacy,
                transcription_result["transcript"],
                transcription_result["whisper_result"]["segments"]
            )
            
            # Add metadata
            structured_data["source_type"] = "video"
            structured_data["language_detection"] = transcription_result["language_info"]
            structured_data["whisper_segments"] = transcription_result["whisper_result"]["segments"]
            structured_data["models_used"] = {
                "whisper": config.WHISPER_MODEL,
                "spacy": config.SPACY_MODEL
            }
            
            self.db_manager.update_job_status(
                job_id, 
                JobStatus.PROCESSING, 
                progress=95, 
                current_step="Finalizing"
            )
            
            return structured_data
            
        finally:
            # Clean up extracted audio file
            if audio_path and os.path.exists(audio_path):
                os.unlink(audio_path)
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status."""
        return {
            "active_jobs": len(self.processing_jobs),
            "max_workers": config.MAX_WORKERS,
            "processing_job_ids": list(self.processing_jobs)
        }
