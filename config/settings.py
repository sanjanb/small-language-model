"""
Configuration settings for the document text extraction system.
"""

import os
from pathlib import Path


class Config:
    """Global configuration settings."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    RESULTS_DIR = PROJECT_ROOT / "results"
    
    # Data paths
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    
    # Model settings
    DEFAULT_MODEL_NAME = "distilbert-base-uncased"
    DEFAULT_MODEL_PATH = MODELS_DIR / "document_ner_model"
    
    # Training settings
    DEFAULT_BATCH_SIZE = 16
    DEFAULT_LEARNING_RATE = 2e-5
    DEFAULT_NUM_EPOCHS = 3
    DEFAULT_MAX_LENGTH = 512
    
    # OCR settings
    TESSERACT_PATH = os.getenv('TESSERACT_PATH', None)
    
    # API settings
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    
    # Entity labels
    ENTITY_LABELS = [
        'O', 'B-NAME', 'I-NAME', 'B-DATE', 'I-DATE', 
        'B-INVOICE_NO', 'I-INVOICE_NO', 'B-AMOUNT', 'I-AMOUNT',
        'B-ADDRESS', 'I-ADDRESS', 'B-PHONE', 'I-PHONE',
        'B-EMAIL', 'I-EMAIL'
    ]
    
    # Supported file formats
    SUPPORTED_FORMATS = ['.pdf', '.docx', '.png', '.jpg', '.jpeg', '.tiff', '.bmp']
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories."""
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.MODELS_DIR,
            cls.RESULTS_DIR,
            cls.RESULTS_DIR / "plots",
            cls.RESULTS_DIR / "metrics"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)