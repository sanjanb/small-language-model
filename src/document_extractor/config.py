"""Configuration management for document extraction system."""
import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class OCRConfig(BaseModel):
    """OCR configuration settings."""
    engine: str = Field(default="tesseract", description="OCR engine to use")
    tesseract_config: str = Field(default="--oem 3 --psm 6", description="Tesseract config")
    language: str = Field(default="eng", description="OCR language")
    confidence_threshold: float = Field(default=0.6, description="Minimum confidence for OCR results")


class PreprocessingConfig(BaseModel):
    """Image preprocessing configuration."""
    resize_factor: float = Field(default=2.0, description="Factor to resize image for better OCR")
    apply_denoising: bool = Field(default=True, description="Apply denoising filter")
    enhance_contrast: bool = Field(default=True, description="Enhance image contrast")
    threshold_method: str = Field(default="adaptive", description="Thresholding method")


class NERConfig(BaseModel):
    """NER configuration settings."""
    model_name: str = Field(default="en_core_web_sm", description="spaCy model name")
    custom_entities: list = Field(
        default_factory=lambda: ["INVOICE_NUMBER", "AMOUNT", "DATE", "VENDOR", "CUSTOMER"],
        description="Custom entity types to extract"
    )
    confidence_threshold: float = Field(default=0.8, description="Minimum confidence for NER predictions")


class OutputConfig(BaseModel):
    """Output configuration settings."""
    format: str = Field(default="json", description="Output format (json, csv, xml)")
    structured: bool = Field(default=True, description="Structure output by document type")
    include_confidence: bool = Field(default=True, description="Include confidence scores")


class Config(BaseModel):
    """Main configuration class."""
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    ner: NERConfig = Field(default_factory=NERConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        with open(config_path, 'w') as f:
            yaml.dump(self.dict(), f, default_flow_style=False)


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or return default."""
    if config_path and Path(config_path).exists():
        return Config.from_yaml(config_path)
    return get_default_config()