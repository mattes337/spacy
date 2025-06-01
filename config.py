"""Configuration module for the transcription service."""

import os
from typing import List


class Config:
    """Configuration class that reads from environment variables."""
    
    # Model Configuration
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "base")
    SPACY_MODEL: str = os.getenv("SPACY_MODEL", "en_core_web_sm")
    
    # File Processing Configuration
    MAX_FILE_SIZE: str = os.getenv("MAX_FILE_SIZE", "100MB")
    TEMP_DIR: str = os.getenv("TEMP_DIR", "/tmp")
    
    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    RELOAD: bool = os.getenv("RELOAD", "false").lower() == "true"
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "/app/logs/transcription.log")
    
    # Security Configuration
    ALLOWED_HOSTS: List[str] = os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
    CORS_ORIGINS: List[str] = os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else []
    
    # Resource Configuration
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "4"))
    WORKER_TIMEOUT: int = int(os.getenv("WORKER_TIMEOUT", "300"))
    
    # Cache Configuration
    MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "/root/.cache")
    WHISPER_CACHE_DIR: str = os.getenv("WHISPER_CACHE_DIR", "/root/.cache/whisper")
    SPACY_CACHE_DIR: str = os.getenv("SPACY_CACHE_DIR", "/root/.cache/spacy")
    
    # Redis Configuration (for production)
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # Test Mode
    TEST_MODE: bool = os.getenv("TEST_MODE", "false").lower() == "true"

    # Topic Segmentation Configuration
    TOPIC_SIMILARITY_THRESHOLD: float = float(os.getenv("TOPIC_SIMILARITY_THRESHOLD", "0.75"))
    MIN_TOPIC_SENTENCES: int = int(os.getenv("MIN_TOPIC_SENTENCES", "2"))
    ENABLE_TOPIC_SEGMENTATION: bool = os.getenv("ENABLE_TOPIC_SEGMENTATION", "true").lower() == "true"
    
    @classmethod
    def get_max_file_size_bytes(cls) -> int:
        """Convert MAX_FILE_SIZE string to bytes."""
        size_str = cls.MAX_FILE_SIZE.upper()
        if size_str.endswith("MB"):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith("GB"):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        elif size_str.endswith("KB"):
            return int(size_str[:-2]) * 1024
        else:
            # Assume bytes if no unit specified
            return int(size_str)
    
    @classmethod
    def setup_cache_dirs(cls):
        """Create cache directories if they don't exist."""
        import os
        os.makedirs(cls.MODEL_CACHE_DIR, exist_ok=True)
        os.makedirs(cls.WHISPER_CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SPACY_CACHE_DIR, exist_ok=True)
        os.makedirs(cls.TEMP_DIR, exist_ok=True)
        
        # Create logs directory
        log_dir = os.path.dirname(cls.LOG_FILE)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
    @classmethod
    def get_supported_audio_formats(cls) -> List[str]:
        """Get list of supported audio file formats."""
        return ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac']
    
    @classmethod
    def get_supported_video_formats(cls) -> List[str]:
        """Get list of supported video file formats."""
        return ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
    
    @classmethod
    def print_config(cls):
        """Print current configuration (for debugging)."""
        print("=== Transcription Service Configuration ===")
        print(f"WHISPER_MODEL: {cls.WHISPER_MODEL}")
        print(f"SPACY_MODEL: {cls.SPACY_MODEL}")
        print(f"MAX_FILE_SIZE: {cls.MAX_FILE_SIZE}")
        print(f"TEMP_DIR: {cls.TEMP_DIR}")
        print(f"HOST: {cls.HOST}")
        print(f"PORT: {cls.PORT}")
        print(f"DEBUG: {cls.DEBUG}")
        print(f"TEST_MODE: {cls.TEST_MODE}")
        print(f"LOG_LEVEL: {cls.LOG_LEVEL}")
        print(f"ENABLE_TOPIC_SEGMENTATION: {cls.ENABLE_TOPIC_SEGMENTATION}")
        print(f"TOPIC_SIMILARITY_THRESHOLD: {cls.TOPIC_SIMILARITY_THRESHOLD}")
        print(f"MIN_TOPIC_SENTENCES: {cls.MIN_TOPIC_SENTENCES}")
        print("=" * 45)


# Initialize configuration
config = Config()

# Setup cache directories on import
config.setup_cache_dirs()
