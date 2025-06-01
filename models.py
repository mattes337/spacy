"""Database models for job tracking and asynchronous processing."""

import uuid
import json
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
from sqlalchemy import Column, String, DateTime, Text, Integer, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.sqlite import JSON

Base = declarative_base()


class JobStatus(str, Enum):
    """Job status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingJob(Base):
    """Model for tracking transcription jobs."""
    __tablename__ = "processing_jobs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    file_type = Column(String, nullable=False)  # 'audio' or 'video'
    language = Column(String, nullable=True)
    webhook_url = Column(String, nullable=True)
    
    status = Column(String, nullable=False, default=JobStatus.PENDING.value)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Store the result as JSON
    result = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Progress tracking
    progress_percentage = Column(Integer, default=0)
    current_step = Column(String, nullable=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for API responses."""
        return {
            "job_id": self.id,
            "filename": self.filename,
            "file_size": self.file_size,
            "file_type": self.file_type,
            "language": self.language,
            "webhook_url": self.webhook_url,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress_percentage": self.progress_percentage,
            "current_step": self.current_step,
            "error_message": self.error_message
        }
    
    def to_status_dict(self) -> Dict[str, Any]:
        """Convert job to status dictionary (without result data)."""
        status_dict = self.to_dict()
        # Don't include the full result in status responses
        status_dict["has_result"] = self.result is not None
        return status_dict


class DatabaseManager:
    """Database manager for handling job persistence with SQLite."""

    def __init__(self, database_path: str = "./transcription_jobs.db"):
        database_url = f"sqlite:///{database_path}"
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.create_tables()
    
    def create_tables(self):
        """Create database tables."""
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self):
        """Get database session."""
        return self.SessionLocal()
    
    def create_job(self, filename: str, file_size: int, file_type: str, 
                   language: Optional[str] = None, webhook_url: Optional[str] = None) -> ProcessingJob:
        """Create a new processing job."""
        with self.get_session() as session:
            job = ProcessingJob(
                filename=filename,
                file_size=file_size,
                file_type=file_type,
                language=language,
                webhook_url=webhook_url
            )
            session.add(job)
            session.commit()
            session.refresh(job)
            return job
    
    def get_job(self, job_id: str) -> Optional[ProcessingJob]:
        """Get job by ID."""
        with self.get_session() as session:
            return session.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
    
    def update_job_status(self, job_id: str, status: JobStatus, 
                         progress: Optional[int] = None, 
                         current_step: Optional[str] = None,
                         error_message: Optional[str] = None):
        """Update job status and progress."""
        with self.get_session() as session:
            job = session.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
            if job:
                job.status = status.value
                if progress is not None:
                    job.progress_percentage = progress
                if current_step is not None:
                    job.current_step = current_step
                if error_message is not None:
                    job.error_message = error_message
                
                if status == JobStatus.PROCESSING and job.started_at is None:
                    job.started_at = datetime.utcnow()
                elif status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                    job.completed_at = datetime.utcnow()
                
                session.commit()
    
    def update_job_result(self, job_id: str, result: Dict[str, Any]):
        """Update job result."""
        with self.get_session() as session:
            job = session.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
            if job:
                job.result = result
                job.status = JobStatus.COMPLETED.value
                job.completed_at = datetime.utcnow()
                job.progress_percentage = 100
                session.commit()
    
    def get_jobs_by_status(self, status: JobStatus) -> list[ProcessingJob]:
        """Get all jobs with specific status."""
        with self.get_session() as session:
            return session.query(ProcessingJob).filter(ProcessingJob.status == status.value).all()
    
    def cleanup_old_jobs(self, days: int = 7):
        """Clean up jobs older than specified days."""
        from datetime import timedelta
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        with self.get_session() as session:
            old_jobs = session.query(ProcessingJob).filter(
                ProcessingJob.created_at < cutoff_date,
                ProcessingJob.status.in_([JobStatus.COMPLETED.value, JobStatus.FAILED.value])
            ).all()
            
            for job in old_jobs:
                session.delete(job)
            
            session.commit()
            return len(old_jobs)
