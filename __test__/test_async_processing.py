"""Test asynchronous processing functionality."""

import asyncio
import json
import tempfile
import os
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import pytest

# Import the app and components
from app import app, db_manager, background_processor
from models import JobStatus


class TestAsyncProcessing:
    """Test class for asynchronous processing functionality."""
    
    def setup_method(self):
        """Setup test client."""
        self.client = TestClient(app)
    
    def test_process_endpoint_returns_job_id(self):
        """Test that /process endpoint returns a job ID immediately."""
        # Create a small test audio file
        test_content = b"fake audio content"
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(test_content)
            tmp_file.flush()
            
            try:
                with open(tmp_file.name, "rb") as f:
                    response = self.client.post(
                        "/process",
                        files={"file": ("test.wav", f, "audio/wav")},
                        params={"language": "en"}
                    )
                
                assert response.status_code == 200
                data = response.json()
                
                # Check response structure
                assert "job_id" in data
                assert "status" in data
                assert data["status"] == "pending"
                assert "message" in data
                assert "filename" in data
                assert data["filename"] == "test.wav"
                assert "file_size" in data
                assert "file_type" in data
                assert data["file_type"] == "audio"
                assert "created_at" in data
                
                # Verify job was created in database
                job = db_manager.get_job(data["job_id"])
                assert job is not None
                assert job.filename == "test.wav"
                assert job.file_type == "audio"
                assert job.status == JobStatus.PENDING.value
                
            finally:
                os.unlink(tmp_file.name)
    
    def test_process_endpoint_with_webhook(self):
        """Test /process endpoint with webhook URL."""
        test_content = b"fake audio content"
        webhook_url = "https://example.com/webhook"
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_file.write(test_content)
            tmp_file.flush()
            
            try:
                with open(tmp_file.name, "rb") as f:
                    response = self.client.post(
                        "/process",
                        files={"file": ("test.mp3", f, "audio/mp3")},
                        params={
                            "language": "es",
                            "webhook_url": webhook_url
                        }
                    )
                
                assert response.status_code == 200
                data = response.json()
                
                # Verify job was created with webhook URL
                job = db_manager.get_job(data["job_id"])
                assert job is not None
                assert job.webhook_url == webhook_url
                assert job.language == "es"
                
            finally:
                os.unlink(tmp_file.name)
    
    def test_status_endpoint(self):
        """Test /status/{job_id} endpoint."""
        # Create a test job
        job = db_manager.create_job(
            filename="test.wav",
            file_size=1024,
            file_type="audio"
        )
        
        response = self.client.get(f"/status/{job.id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["job_id"] == job.id
        assert data["filename"] == "test.wav"
        assert data["file_size"] == 1024
        assert data["file_type"] == "audio"
        assert data["status"] == "pending"
        assert "created_at" in data
        assert "has_result" in data
        assert data["has_result"] is False
    
    def test_status_endpoint_not_found(self):
        """Test /status/{job_id} endpoint with non-existent job."""
        response = self.client.get("/status/non-existent-job-id")
        assert response.status_code == 404
        assert "Job not found" in response.json()["detail"]
    
    def test_result_endpoint_pending_job(self):
        """Test /result/{job_id} endpoint with pending job."""
        job = db_manager.create_job(
            filename="test.wav",
            file_size=1024,
            file_type="audio"
        )
        
        response = self.client.get(f"/result/{job.id}")
        assert response.status_code == 202
        
        data = response.json()
        assert data["job_id"] == job.id
        assert data["status"] == "pending"
        assert "message" in data
    
    def test_result_endpoint_processing_job(self):
        """Test /result/{job_id} endpoint with processing job."""
        job = db_manager.create_job(
            filename="test.wav",
            file_size=1024,
            file_type="audio"
        )
        
        # Update job to processing status
        db_manager.update_job_status(
            job.id,
            JobStatus.PROCESSING,
            progress=50,
            current_step="Transcribing audio"
        )
        
        response = self.client.get(f"/result/{job.id}")
        assert response.status_code == 202
        
        data = response.json()
        assert data["job_id"] == job.id
        assert data["status"] == "processing"
        assert data["progress_percentage"] == 50
        assert data["current_step"] == "Transcribing audio"
    
    def test_result_endpoint_failed_job(self):
        """Test /result/{job_id} endpoint with failed job."""
        job = db_manager.create_job(
            filename="test.wav",
            file_size=1024,
            file_type="audio"
        )
        
        # Update job to failed status
        db_manager.update_job_status(
            job.id,
            JobStatus.FAILED,
            error_message="Test error message"
        )
        
        response = self.client.get(f"/result/{job.id}")
        assert response.status_code == 500
        
        data = response.json()
        assert data["job_id"] == job.id
        assert data["status"] == "failed"
        assert data["error_message"] == "Test error message"
    
    def test_result_endpoint_completed_job(self):
        """Test /result/{job_id} endpoint with completed job."""
        job = db_manager.create_job(
            filename="test.wav",
            file_size=1024,
            file_type="audio"
        )
        
        # Create mock result
        mock_result = {
            "transcript": "Hello world",
            "language_detection": {"language": "en", "confidence": 0.95},
            "topics": [{"summary": "", "seconds": 0.0, "sentences": ["Hello world"]}]
        }
        
        # Update job with result
        db_manager.update_job_result(job.id, mock_result)
        
        response = self.client.get(f"/result/{job.id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["job_id"] == job.id
        assert data["status"] == "completed"
        assert "completed_at" in data
        assert "result" in data
        assert data["result"]["transcript"] == "Hello world"
    
    def test_result_endpoint_not_found(self):
        """Test /result/{job_id} endpoint with non-existent job."""
        response = self.client.get("/result/non-existent-job-id")
        assert response.status_code == 404
        assert "Job not found" in response.json()["detail"]
    
    def test_processing_status_endpoint(self):
        """Test /jobs/processing endpoint."""
        response = self.client.get("/jobs/processing")
        assert response.status_code == 200
        
        data = response.json()
        assert "active_jobs" in data
        assert "max_workers" in data
        assert "processing_job_ids" in data
        assert isinstance(data["active_jobs"], int)
        assert isinstance(data["processing_job_ids"], list)
    
    def test_unsupported_file_format(self):
        """Test /process endpoint with unsupported file format."""
        test_content = b"fake content"
        
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_file:
            tmp_file.write(test_content)
            tmp_file.flush()
            
            try:
                with open(tmp_file.name, "rb") as f:
                    response = self.client.post(
                        "/process",
                        files={"file": ("test.txt", f, "text/plain")}
                    )
                
                assert response.status_code == 400
                assert "Unsupported file format" in response.json()["detail"]
                
            finally:
                os.unlink(tmp_file.name)
    
    def test_health_endpoint_includes_async_info(self):
        """Test that health endpoint includes async processing information."""
        response = self.client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "background_processor" in data
        assert "database_connected" in data
        assert data["database_connected"] is True
        
        # Check background processor info
        bg_info = data["background_processor"]
        assert "active_jobs" in bg_info
        assert "max_workers" in bg_info
        assert "processing_job_ids" in bg_info
        
        # Check webhook settings in configuration
        config_data = data["configuration"]
        assert "webhook_settings" in config_data
        webhook_settings = config_data["webhook_settings"]
        assert "timeout" in webhook_settings
        assert "max_retries" in webhook_settings
        assert "retry_delay" in webhook_settings


if __name__ == "__main__":
    # Run basic tests
    test_instance = TestAsyncProcessing()
    test_instance.setup_method()
    
    print("Testing async processing endpoints...")
    
    try:
        test_instance.test_processing_status_endpoint()
        print("✓ Processing status endpoint test passed")
        
        test_instance.test_health_endpoint_includes_async_info()
        print("✓ Health endpoint async info test passed")
        
        print("✓ All basic async tests passed!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        raise
