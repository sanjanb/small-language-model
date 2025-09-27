"""Image preprocessing utilities for better OCR results."""
import logging
from typing import Tuple, Optional
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from dataclasses import dataclass

from ..config import PreprocessingConfig

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingResult:
    """Result of image preprocessing."""
    processed_image: np.ndarray
    original_shape: Tuple[int, int]
    processing_steps: list


class ImagePreprocessor:
    """Image preprocessing for better OCR results."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
    
    def preprocess(self, image: np.ndarray) -> PreprocessingResult:
        """Apply preprocessing pipeline to image."""
        original_shape = image.shape[:2]
        processed_image = image.copy()
        steps = []
        
        try:
            # Step 1: Resize for better OCR
            if self.config.resize_factor != 1.0:
                processed_image = self._resize_image(processed_image, self.config.resize_factor)
                steps.append(f"resized by factor {self.config.resize_factor}")
            
            # Step 2: Convert to grayscale
            if len(processed_image.shape) == 3:
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
                steps.append("converted to grayscale")
            
            # Step 3: Denoising
            if self.config.apply_denoising:
                processed_image = self._denoise_image(processed_image)
                steps.append("applied denoising")
            
            # Step 4: Enhance contrast
            if self.config.enhance_contrast:
                processed_image = self._enhance_contrast(processed_image)
                steps.append("enhanced contrast")
            
            # Step 5: Thresholding
            processed_image = self._apply_threshold(processed_image)
            steps.append(f"applied {self.config.threshold_method} thresholding")
            
            logger.info(f"Preprocessing completed: {', '.join(steps)}")
            
            return PreprocessingResult(
                processed_image=processed_image,
                original_shape=original_shape,
                processing_steps=steps
            )
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return PreprocessingResult(
                processed_image=image,
                original_shape=original_shape,
                processing_steps=["preprocessing failed"]
            )
    
    def _resize_image(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Resize image by given factor."""
        height, width = image.shape[:2]
        new_width = int(width * factor)
        new_height = int(height * factor)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Apply denoising filter."""
        return cv2.fastNlMeansDenoising(image)
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE."""
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    
    def _apply_threshold(self, image: np.ndarray) -> np.ndarray:
        """Apply thresholding based on configuration."""
        if self.config.threshold_method == "adaptive":
            return cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        elif self.config.threshold_method == "otsu":
            _, threshed = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return threshed
        else:  # simple threshold
            _, threshed = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            return threshed
    
    def preprocess_from_path(self, image_path: str) -> PreprocessingResult:
        """Load and preprocess image from file path."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            return self.preprocess(image)
        except Exception as e:
            logger.error(f"Failed to preprocess image from {image_path}: {e}")
            raise
    
    def save_preprocessed_image(self, result: PreprocessingResult, output_path: str):
        """Save preprocessed image to file."""
        try:
            cv2.imwrite(output_path, result.processed_image)
            logger.info(f"Saved preprocessed image to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save preprocessed image: {e}")
            raise