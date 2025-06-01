# Asynchronous Processing API Guide

The transcription service now supports asynchronous processing for handling large audio and video files without timeouts. This guide explains how to use the new async API endpoints.

## Overview

The async processing system provides:

- **Immediate response** with job ID when uploading files
- **Background processing** to handle large files (multiple GBs)
- **Job status tracking** with progress updates
- **Webhook notifications** when processing completes
- **Result retrieval** when jobs are finished

## API Endpoints

### 1. Submit File for Processing

**POST** `/process`

Submit an audio or video file for asynchronous transcription.

**Parameters:**
- `file` (form-data): Audio or video file to transcribe
- `language` (query, optional): Language code (e.g., 'en', 'es', 'fr')
- `webhook_url` (query, optional): URL to call when processing completes

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "File uploaded successfully. Processing started.",
  "filename": "large_video.mp4",
  "file_size": 2147483648,
  "file_type": "video",
  "created_at": "2024-01-15T10:30:00.000Z"
}
```

### 2. Check Job Status

**GET** `/status/{job_id}`

Get the current status of a processing job.

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "large_video.mp4",
  "file_size": 2147483648,
  "file_type": "video",
  "language": "en",
  "webhook_url": "https://example.com/webhook",
  "status": "processing",
  "created_at": "2024-01-15T10:30:00.000Z",
  "started_at": "2024-01-15T10:30:05.000Z",
  "completed_at": null,
  "progress_percentage": 45,
  "current_step": "Transcribing audio",
  "error_message": null,
  "has_result": false
}
```

**Status Values:**
- `pending`: Job is queued for processing
- `processing`: Job is currently being processed
- `completed`: Job finished successfully
- `failed`: Job failed with an error

### 3. Get Job Result

**GET** `/result/{job_id}`

Retrieve the result of a completed job.

**Response Codes:**
- `200`: Job completed successfully (returns full result)
- `202`: Job still pending or processing
- `404`: Job not found
- `500`: Job failed

**Successful Response (200):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "completed_at": "2024-01-15T10:35:30.000Z",
  "result": {
    "transcript": "Hello, this is the transcribed text...",
    "language_detection": {
      "language": "en",
      "confidence": 0.95
    },
    "topics": [
      {
        "summary": "",
        "seconds": 0.0,
        "sentences": ["Hello, this is the transcribed text..."]
      }
    ],
    "source_type": "video",
    "filename": "large_video.mp4",
    "file_size": 2147483648,
    "whisper_segments": [...],
    "models_used": {
      "whisper": "base",
      "spacy": "en_core_web_sm"
    },
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "processing_completed_at": "2024-01-15T10:35:30.000Z"
  }
}
```

**Processing Response (202):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "progress_percentage": 45,
  "current_step": "Transcribing audio",
  "message": "Job is currently being processed"
}
```

### 4. Get Processing Status

**GET** `/jobs/processing`

Get information about currently active processing jobs.

**Response:**
```json
{
  "active_jobs": 2,
  "max_workers": 4,
  "processing_job_ids": [
    "550e8400-e29b-41d4-a716-446655440000",
    "660f9511-f3ac-52e5-b827-557766551111"
  ]
}
```

## Webhook Integration

When you provide a `webhook_url` parameter, the service will send a POST request to that URL when processing completes.

### Webhook Payload

**Success:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "completed_at": "2024-01-15T10:35:30.000Z",
  "result": {
    "transcript": "...",
    "language_detection": {...},
    "topics": [...],
    ...
  }
}
```

**Failure:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "failed",
  "completed_at": "2024-01-15T10:35:30.000Z",
  "error_message": "File format not supported"
}
```

### Webhook Configuration

The service will retry failed webhook calls with the following settings:
- **Timeout**: 30 seconds (configurable via `WEBHOOK_TIMEOUT`)
- **Max Retries**: 3 attempts (configurable via `WEBHOOK_MAX_RETRIES`)
- **Retry Delay**: 5 seconds between attempts (configurable via `WEBHOOK_RETRY_DELAY`)

## Usage Examples

### Python Client Example

```python
import asyncio
import aiohttp

async def transcribe_large_file():
    # Upload file
    async with aiohttp.ClientSession() as session:
        with open("large_video.mp4", "rb") as f:
            data = aiohttp.FormData()
            data.add_field("file", f, filename="large_video.mp4")
            
            async with session.post(
                "http://localhost:8000/process",
                data=data,
                params={
                    "language": "en",
                    "webhook_url": "https://myapp.com/webhook"
                }
            ) as response:
                upload_result = await response.json()
                job_id = upload_result["job_id"]
    
    # Poll for completion
    while True:
        async with session.get(f"http://localhost:8000/result/{job_id}") as response:
            if response.status == 200:
                result = await response.json()
                print("Transcription completed!")
                print(f"Transcript: {result['result']['transcript'][:100]}...")
                break
            elif response.status == 202:
                status_data = await response.json()
                print(f"Progress: {status_data.get('progress_percentage', 0)}%")
                await asyncio.sleep(10)
            else:
                print("Job failed or not found")
                break

# Run the example
asyncio.run(transcribe_large_file())
```

### cURL Examples

**Submit file:**
```bash
curl -X POST "http://localhost:8000/process" \
  -F "file=@large_video.mp4" \
  -G -d "language=en" \
  -d "webhook_url=https://myapp.com/webhook"
```

**Check status:**
```bash
curl "http://localhost:8000/status/550e8400-e29b-41d4-a716-446655440000"
```

**Get result:**
```bash
curl "http://localhost:8000/result/550e8400-e29b-41d4-a716-446655440000"
```

## Configuration

### Environment Variables

- `MAX_CONCURRENT_JOBS`: Maximum number of concurrent processing jobs (default: 4)
- `DATABASE_PATH`: SQLite database file path (default: ./transcription_jobs.db)
- `WEBHOOK_TIMEOUT`: Webhook request timeout in seconds (default: 30)
- `WEBHOOK_MAX_RETRIES`: Maximum webhook retry attempts (default: 3)
- `WEBHOOK_RETRY_DELAY`: Delay between webhook retries in seconds (default: 5)

### Database

The service uses SQLite for job tracking, which is simple and requires no additional setup.

## Migration from Synchronous API

The original synchronous endpoints (`/process-audio`, `/process-video`) are still available for backward compatibility, but the new `/process` endpoint is recommended for all new integrations.

**Key differences:**
- Synchronous: Returns full result immediately, may timeout on large files
- Asynchronous: Returns job ID immediately, retrieve result separately

## Error Handling

- **File too large**: Returns 400 error immediately
- **Unsupported format**: Returns 400 error immediately  
- **Processing failure**: Job status becomes "failed", error details in `/result/{job_id}`
- **Webhook failure**: Retried automatically, but job still completes successfully

## Monitoring

Use the `/health` endpoint to monitor service status:

```bash
curl "http://localhost:8000/health"
```

The health response includes background processor status and active job counts.
