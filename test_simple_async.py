#!/usr/bin/env python3
"""Simple test for the streamlined async implementation."""

import asyncio
import tempfile
import os
from models import DatabaseManager, JobStatus


def test_database():
    """Test basic database functionality."""
    print("Testing database functionality...")
    
    # Create database manager
    db = DatabaseManager("test_jobs.db")
    
    # Create a job
    job = db.create_job(
        filename="test.wav",
        file_size=1024,
        file_type="audio",
        language="en",
        webhook_url="https://example.com/webhook"
    )
    
    print(f"✓ Created job: {job.id}")
    assert job.filename == "test.wav"
    assert job.file_type == "audio"
    assert job.status == JobStatus.PENDING.value
    
    # Retrieve job
    retrieved_job = db.get_job(job.id)
    assert retrieved_job is not None
    assert retrieved_job.id == job.id
    print(f"✓ Retrieved job: {retrieved_job.filename}")
    
    # Update job status
    db.update_job_status(
        job.id,
        JobStatus.PROCESSING,
        progress=50,
        current_step="Testing"
    )
    
    updated_job = db.get_job(job.id)
    assert updated_job.status == JobStatus.PROCESSING.value
    assert updated_job.progress_percentage == 50
    print("✓ Job status update successful")
    
    # Update job result
    test_result = {
        "transcript": "Hello world",
        "language_detection": {"language": "en", "confidence": 0.95}
    }
    
    db.update_job_result(job.id, test_result)
    
    completed_job = db.get_job(job.id)
    assert completed_job.status == JobStatus.COMPLETED.value
    assert completed_job.result["transcript"] == "Hello world"
    print("✓ Job result update successful")
    
    # Clean up test database
    os.unlink("test_jobs.db")
    print("✓ Database test completed")


async def test_webhook_client():
    """Test webhook client functionality."""
    print("\nTesting webhook client...")
    
    from background_processor import WebhookClient
    
    webhook_client = WebhookClient()
    
    # Test with invalid URL (should fail gracefully)
    test_data = {
        "job_id": "test-job-123",
        "status": "completed",
        "result": {"transcript": "test"}
    }
    
    # This should fail but not raise an exception
    result = await webhook_client.send_webhook(
        "http://invalid-url-that-does-not-exist.com/webhook",
        test_data
    )
    
    assert result is False  # Should fail gracefully
    print("✓ Webhook client handles invalid URLs gracefully")
    
    # Test with no webhook URL
    result = await webhook_client.send_webhook("", test_data)
    assert result is True  # Should return True for empty URL
    print("✓ Webhook client handles empty URLs correctly")
    
    print("✓ Webhook client test completed")


def test_config():
    """Test configuration."""
    print("\nTesting configuration...")
    
    from config import config
    
    # Test basic config values
    assert hasattr(config, 'DATABASE_PATH')
    assert hasattr(config, 'MAX_CONCURRENT_JOBS')
    assert hasattr(config, 'WEBHOOK_TIMEOUT')
    
    print(f"✓ Database path: {config.DATABASE_PATH}")
    print(f"✓ Max concurrent jobs: {config.MAX_CONCURRENT_JOBS}")
    print(f"✓ Webhook timeout: {config.WEBHOOK_TIMEOUT}")
    print("✓ Configuration test completed")


async def test_background_processor():
    """Test background processor initialization."""
    print("\nTesting background processor...")
    
    from background_processor import BackgroundProcessor
    
    # Create mock processor
    class MockTextProcessor:
        def validate_file_size(self, size):
            if size > 100 * 1024 * 1024:
                raise Exception("File too large")
    
    db = DatabaseManager("test_bg.db")
    mock_processor = MockTextProcessor()
    
    bg_processor = BackgroundProcessor(mock_processor, db)
    
    # Test processing status
    status = bg_processor.get_processing_status()
    assert "active_jobs" in status
    assert "max_workers" in status
    assert "processing_job_ids" in status
    print("✓ Background processor status retrieval successful")
    
    # Clean up
    os.unlink("test_bg.db")
    print("✓ Background processor test completed")


async def main():
    """Run all tests."""
    print("=== Testing Streamlined Async Implementation ===\n")
    
    try:
        # Test database functionality
        test_database()
        
        # Test configuration
        test_config()
        
        # Test webhook client
        await test_webhook_client()
        
        # Test background processor
        await test_background_processor()
        
        print("\n=== All Tests Passed! ===")
        print("✓ Database operations working")
        print("✓ Configuration working")
        print("✓ Webhook client working")
        print("✓ Background processor working")
        print("\nThe streamlined async processing system is ready!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
