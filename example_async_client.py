"""Simple example client for async transcription API."""

import asyncio
import aiohttp
import time
from typing import Optional


class SimpleAsyncClient:
    """Client for interacting with the asynchronous transcription service."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
    
    async def upload_file(self, file_path: str, language: Optional[str] = None, 
                         webhook_url: Optional[str] = None) -> dict:
        """Upload a file for transcription and get job ID."""
        url = f"{self.base_url}/process"
        
        params = {}
        if language:
            params["language"] = language
        if webhook_url:
            params["webhook_url"] = webhook_url
        
        async with aiohttp.ClientSession() as session:
            with open(file_path, "rb") as f:
                data = aiohttp.FormData()
                data.add_field("file", f, filename=file_path.split("/")[-1])
                
                async with session.post(url, data=data, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        raise Exception(f"Upload failed: {response.status} - {error_text}")
    
    async def get_job_status(self, job_id: str) -> dict:
        """Get the status of a processing job."""
        url = f"{self.base_url}/status/{job_id}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 404:
                    raise Exception("Job not found")
                else:
                    error_text = await response.text()
                    raise Exception(f"Status check failed: {response.status} - {error_text}")
    
    async def get_job_result(self, job_id: str) -> dict:
        """Get the result of a processing job."""
        url = f"{self.base_url}/result/{job_id}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                result = await response.json()
                return {
                    "status_code": response.status,
                    "data": result
                }
    
    async def wait_for_completion(self, job_id: str, poll_interval: int = 5, 
                                 timeout: int = 3600) -> dict:
        """Wait for a job to complete and return the result."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                result = await self.get_job_result(job_id)
                
                if result["status_code"] == 200:
                    # Job completed successfully
                    return result["data"]
                elif result["status_code"] == 500:
                    # Job failed
                    raise Exception(f"Job failed: {result['data'].get('error_message', 'Unknown error')}")
                elif result["status_code"] == 202:
                    # Job still processing
                    status_data = result["data"]
                    print(f"Job {job_id} status: {status_data.get('status', 'unknown')}")
                    if "progress_percentage" in status_data:
                        print(f"Progress: {status_data['progress_percentage']}% - {status_data.get('current_step', '')}")
                    
                    await asyncio.sleep(poll_interval)
                else:
                    raise Exception(f"Unexpected response: {result['status_code']}")
                    
            except Exception as e:
                print(f"Error checking job status: {e}")
                await asyncio.sleep(poll_interval)
        
        raise Exception(f"Job {job_id} did not complete within {timeout} seconds")
    
    async def get_service_health(self) -> dict:
        """Get service health information."""
        url = f"{self.base_url}/health"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Health check failed: {response.status} - {error_text}")
    
    async def get_processing_status(self) -> dict:
        """Get current processing status."""
        url = f"{self.base_url}/jobs/processing"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Processing status check failed: {response.status} - {error_text}")


async def simple_example():
    """Simple example of async transcription."""
    client = SimpleAsyncClient()

    print("=== Simple Async Transcription Example ===")

    # Check service health
    try:
        health = await client.get_service_health()
        print("✓ Service is healthy")
    except Exception as e:
        print(f"✗ Service health check failed: {e}")
        return

    # Create a test file for demonstration
    test_content = b"fake audio content for testing"

    try:
        # Upload test content
        print("\n1. Uploading test file...")
        data = aiohttp.FormData()
        data.add_field("file", test_content, filename="test.wav", content_type="audio/wav")

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{client.base_url}/process", data=data) as response:
                if response.status == 200:
                    upload_result = await response.json()
                    job_id = upload_result["job_id"]
                    print(f"✓ Job created: {job_id}")
                else:
                    print(f"✗ Upload failed: {response.status}")
                    return

        # Check job status
        print("\n2. Checking job status...")
        status = await client.get_job_status(job_id)
        print(f"✓ Job status: {status['status']}")

        # Wait for completion (simplified)
        print("\n3. Waiting for completion...")
        for i in range(30):  # Wait up to 30 seconds
            result = await client.get_job_result(job_id)
            if result["status_code"] == 200:
                print("✓ Job completed!")
                break
            elif result["status_code"] == 202:
                print(f"  Still processing... ({i+1}/30)")
                await asyncio.sleep(1)
            else:
                print(f"✗ Job failed: {result['data']}")
                break

    except Exception as e:
        print(f"✗ Error: {e}")


def example_webhook_handler():
    """Example webhook handler (Flask/FastAPI endpoint)."""
    webhook_example = '''
# Example webhook handler (Flask)
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/transcription-complete', methods=['POST'])
def handle_transcription_complete():
    data = request.json
    
    job_id = data.get('job_id')
    status = data.get('status')
    
    if status == 'completed':
        result = data.get('result', {})
        transcript = result.get('transcript', '')
        language = result.get('language_detection', {}).get('language', 'unknown')
        
        print(f"Job {job_id} completed!")
        print(f"Language: {language}")
        print(f"Transcript: {transcript[:100]}...")
        
        # Process the result as needed
        # e.g., save to database, send notification, etc.
        
    elif status == 'failed':
        error_message = data.get('error_message', 'Unknown error')
        print(f"Job {job_id} failed: {error_message}")
    
    return jsonify({"status": "received"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
'''
    
    print("\n=== Example Webhook Handler ===")
    print(webhook_example)


if __name__ == "__main__":
    print("Running simple async transcription example...")

    # Run the example
    asyncio.run(simple_example())

    # Show webhook example
    example_webhook_handler()
