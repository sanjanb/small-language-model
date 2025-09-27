"""Utility functions for the document extraction system."""
import logging
import os
from pathlib import Path
from typing import List, Optional, Union
import hashlib
import json
from datetime import datetime


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger("document_extractor")
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_file_hash(file_path: Union[str, Path]) -> str:
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def ensure_directory(directory: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_supported_image_extensions() -> List[str]:
    """Get list of supported image file extensions."""
    return ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp']


def is_supported_image(file_path: Union[str, Path]) -> bool:
    """Check if file has a supported image extension."""
    return Path(file_path).suffix.lower() in get_supported_image_extensions()


def find_images_in_directory(directory: Union[str, Path], 
                            recursive: bool = False) -> List[Path]:
    """Find all supported image files in a directory."""
    directory = Path(directory)
    image_files = []
    
    extensions = get_supported_image_extensions()
    
    if recursive:
        for ext in extensions:
            image_files.extend(directory.rglob(f"*{ext}"))
            image_files.extend(directory.rglob(f"*{ext.upper()}"))
    else:
        for ext in extensions:
            image_files.extend(directory.glob(f"*{ext}"))
            image_files.extend(directory.glob(f"*{ext.upper()}"))
    
    return sorted(image_files)


def save_json(data: dict, file_path: Union[str, Path], indent: int = 2) -> None:
    """Save dictionary as JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(file_path: Union[str, Path]) -> dict:
    """Load JSON file as dictionary."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_timestamp() -> str:
    """Create ISO format timestamp string."""
    return datetime.now().isoformat()


def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing/replacing invalid characters."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename


def calculate_processing_stats(results: List[dict]) -> dict:
    """Calculate processing statistics from results."""
    if not results:
        return {
            "total_files": 0,
            "successful": 0,
            "failed": 0,
            "success_rate": 0.0,
            "average_processing_time": 0.0,
            "total_processing_time": 0.0
        }
    
    successful = sum(1 for r in results if r.get('success', False))
    total = len(results)
    processing_times = [r.get('processing_time', 0) for r in results]
    
    return {
        "total_files": total,
        "successful": successful,
        "failed": total - successful,
        "success_rate": successful / total if total > 0 else 0,
        "average_processing_time": sum(processing_times) / len(processing_times) if processing_times else 0,
        "total_processing_time": sum(processing_times)
    }


class ProgressTracker:
    """Simple progress tracking utility."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.logger = logging.getLogger(__name__)
    
    def update(self, increment: int = 1) -> None:
        """Update progress."""
        self.current = min(self.current + increment, self.total)
        percentage = (self.current / self.total) * 100 if self.total > 0 else 0
        
        self.logger.info(f"{self.description}: {self.current}/{self.total} ({percentage:.1f}%)")
    
    def finish(self) -> None:
        """Mark as finished."""
        self.current = self.total
        self.logger.info(f"{self.description}: Completed ({self.total} items)")


def validate_config_file(config_path: Union[str, Path]) -> bool:
    """Validate that a config file exists and is readable."""
    try:
        path = Path(config_path)
        if not path.exists():
            return False
        
        if not path.is_file():
            return False
        
        # Try to read as JSON or YAML
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            return len(content.strip()) > 0
            
    except Exception:
        return False


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    size_index = 0
    
    while size_bytes >= 1024 and size_index < len(size_names) - 1:
        size_bytes /= 1024.0
        size_index += 1
    
    return f"{size_bytes:.1f}{size_names[size_index]}"


def get_file_info(file_path: Union[str, Path]) -> dict:
    """Get comprehensive file information."""
    path = Path(file_path)
    
    if not path.exists():
        return {"error": "File does not exist"}
    
    stat = path.stat()
    
    return {
        "name": path.name,
        "path": str(path.absolute()),
        "size": stat.st_size,
        "size_formatted": format_file_size(stat.st_size),
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "extension": path.suffix.lower(),
        "is_supported": is_supported_image(path)
    }