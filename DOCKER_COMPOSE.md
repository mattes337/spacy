# Docker Compose Setup for Audio/Video Transcription Service

This document provides detailed instructions for running the transcription service using Docker Compose.

## 📋 Prerequisites

- Docker and Docker Compose installed
- At least 4GB of available RAM
- 2GB of free disk space for models and dependencies

## 🚀 Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd spacy

# Start the service
docker-compose up -d

# Check service health
curl http://localhost:8000/health

# View logs
docker-compose logs -f transcription-service

# Stop the service
docker-compose down
```

## 📁 Available Configurations

### 1. Basic Setup (`docker-compose.yml`)

**Use case:** Simple deployment, development, testing

**Features:**
- Single transcription service container
- Health checks with curl
- Persistent volumes for model cache and temp files
- Automatic restart on failure

**Usage:**
```bash
docker-compose up -d
```

### 2. Development Setup (`docker-compose.dev.yml`)

**Use case:** Active development with code changes

**Features:**
- Source code mounted as volume for live editing
- Auto-reload on code changes
- Test service on port 8001
- Development-friendly configuration

**Usage:**
```bash
docker-compose -f docker-compose.dev.yml up -d

# Access main service: http://localhost:8000
# Access test service: http://localhost:8001
```

### 3. Production Setup (`docker-compose.prod.yml`)

**Use case:** Production deployment with load balancing

**Features:**
- Nginx reverse proxy with rate limiting
- Redis for caching and session management
- Resource limits and monitoring
- SSL/HTTPS ready configuration
- Persistent volumes for data

**Usage:**
```bash
docker-compose -f docker-compose.prod.yml up -d
```

## 🔧 Configuration

### Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
# Edit .env with your preferred settings
```

Key variables:
- `WHISPER_MODEL`: Model size (base, small, medium, large)
- `SPACY_MODEL`: spaCy model name
- `MAX_FILE_SIZE`: Maximum upload size
- `DEBUG`: Enable debug mode

### Volumes

The service uses persistent volumes:
- `model-cache`: Stores downloaded ML models
- `temp-files`: Temporary file processing

## 🧪 Testing

### Health Check
```bash
curl http://localhost:8000/health
# Expected: {"status":"healthy","models_loaded":true}
```

### Audio Processing Test
```bash
curl -X POST "http://localhost:8000/process-audio" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_audio.wav"
```

### Video Processing Test
```bash
curl -X POST "http://localhost:8000/process-video" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_video.mp4"
```

## 📊 Monitoring

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f transcription-service

# Last 100 lines
docker-compose logs --tail=100 transcription-service
```

### Check Status
```bash
docker-compose ps
```

### Resource Usage
```bash
docker stats
```

## 🔧 Troubleshooting

### Service Won't Start
```bash
# Check logs
docker-compose logs transcription-service

# Rebuild image
docker-compose build --no-cache

# Check configuration
docker-compose config
```

### Health Check Failing
```bash
# Wait for model download (first run takes 2-3 minutes)
docker-compose logs -f transcription-service

# Check if service is responding
curl -v http://localhost:8000/health
```

### Out of Memory
```bash
# Check resource usage
docker stats

# Increase Docker memory limit in Docker Desktop
# Or use smaller Whisper model in .env:
# WHISPER_MODEL=tiny
```

### Port Already in Use
```bash
# Check what's using port 8000
netstat -tulpn | grep 8000

# Use different port
docker-compose up -d --scale transcription-service=0
docker run -p 8001:8000 spacy-transcription-service
```

## 🔄 Updates and Maintenance

### Update Service
```bash
# Pull latest changes
git pull

# Rebuild and restart
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Clean Up
```bash
# Stop and remove containers
docker-compose down

# Remove volumes (WARNING: deletes cached models)
docker-compose down -v

# Remove images
docker-compose down --rmi all
```

### Backup Models
```bash
# Create backup of model cache
docker run --rm -v spacy_model-cache:/data -v $(pwd):/backup alpine tar czf /backup/model-cache-backup.tar.gz -C /data .

# Restore backup
docker run --rm -v spacy_model-cache:/data -v $(pwd):/backup alpine tar xzf /backup/model-cache-backup.tar.gz -C /data
```

## 🚀 Production Deployment

For production deployment with `docker-compose.prod.yml`:

1. **Configure SSL certificates:**
   ```bash
   mkdir ssl
   # Add your cert.pem and key.pem files to ssl/
   ```

2. **Set environment variables:**
   ```bash
   cp .env.example .env
   # Configure production settings
   ```

3. **Start services:**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

4. **Monitor services:**
   ```bash
   docker-compose -f docker-compose.prod.yml logs -f
   ```

## 📞 Support

For issues and questions:
1. Check the logs: `docker-compose logs -f`
2. Verify configuration: `docker-compose config`
3. Check resource usage: `docker stats`
4. Review the main README.md for additional troubleshooting
