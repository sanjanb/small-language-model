"""OCR utilities for text extraction from documents."""
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import pytesseract
import easyocr
from dataclasses import dataclass

from ..config import OCRConfig

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Data class for OCR results."""
    text: str
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None
    word_confidences: Optional[List[float]] = None


class OCREngine:
    """Base class for OCR engines."""
    
    def __init__(self, config: OCRConfig):
        self.config = config
    
    def extract_text(self, image: np.ndarray) -> OCRResult:
        """Extract text from image."""
        raise NotImplementedError


class TesseractOCR(OCREngine):
    """Tesseract OCR implementation."""
    
    def __init__(self, config: OCRConfig):
        super().__init__(config)
        self._verify_tesseract()
    
    def _verify_tesseract(self):
        """Verify Tesseract installation."""
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            logger.error(f"Tesseract not found: {e}")
            raise RuntimeError("Tesseract not installed or not found in PATH")
    
    def extract_text(self, image: np.ndarray) -> OCRResult:
        """Extract text using Tesseract."""
        try:
            # Convert numpy array to PIL Image if needed
            if isinstance(image, np.ndarray):
                image_pil = Image.fromarray(image)
            else:
                image_pil = image
            
            # Extract text with confidence scores
            data = pytesseract.image_to_data(
                image_pil,
                config=self.config.tesseract_config,
                lang=self.config.language,
                output_type=pytesseract.Output.DICT
            )
            
            # Filter by confidence threshold
            confident_words = []
            word_confidences = []
            
            for i, conf in enumerate(data['conf']):
                if int(conf) >= self.config.confidence_threshold * 100:
                    word = data['text'][i].strip()
                    if word:  # Only add non-empty words
                        confident_words.append(word)
                        word_confidences.append(int(conf) / 100.0)
            
            text = ' '.join(confident_words)
            avg_confidence = np.mean(word_confidences) if word_confidences else 0.0
            
            return OCRResult(
                text=text,
                confidence=avg_confidence,
                word_confidences=word_confidences
            )
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return OCRResult(text="", confidence=0.0)


class EasyOCREngine(OCREngine):
    """EasyOCR implementation."""
    
    def __init__(self, config: OCRConfig):
        super().__init__(config)
        try:
            self.reader = easyocr.Reader([self.config.language])
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            raise RuntimeError(f"EasyOCR initialization failed: {e}")
    
    def extract_text(self, image: np.ndarray) -> OCRResult:
        """Extract text using EasyOCR."""
        try:
            results = self.reader.readtext(image)
            
            # Filter by confidence threshold
            confident_results = [
                (text, conf) for _, text, conf in results 
                if conf >= self.config.confidence_threshold
            ]
            
            if not confident_results:
                return OCRResult(text="", confidence=0.0)
            
            texts, confidences = zip(*confident_results)
            combined_text = ' '.join(texts)
            avg_confidence = np.mean(confidences)
            
            return OCRResult(
                text=combined_text,
                confidence=avg_confidence,
                word_confidences=list(confidences)
            )
            
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            return OCRResult(text="", confidence=0.0)


class OCRProcessor:
    """Main OCR processor that coordinates different engines."""
    
    def __init__(self, config: OCRConfig):
        self.config = config
        self.engine = self._create_engine()
    
    def _create_engine(self) -> OCREngine:
        """Create OCR engine based on configuration."""
        if self.config.engine.lower() == "tesseract":
            return TesseractOCR(self.config)
        elif self.config.engine.lower() == "easyocr":
            return EasyOCREngine(self.config)
        else:
            raise ValueError(f"Unsupported OCR engine: {self.config.engine}")
    
    def process_image(self, image_path: str) -> OCRResult:
        """Process image file and extract text."""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract text
            result = self.engine.extract_text(image_rgb)
            
            logger.info(f"Extracted text from {image_path} with confidence {result.confidence:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            return OCRResult(text="", confidence=0.0)
    
    def process_image_array(self, image: np.ndarray) -> OCRResult:
        """Process image array and extract text."""
        try:
            result = self.engine.extract_text(image)
            logger.info(f"Extracted text from image array with confidence {result.confidence:.2f}")
            return result
        except Exception as e:
            logger.error(f"Failed to process image array: {e}")
            return OCRResult(text="", confidence=0.0)