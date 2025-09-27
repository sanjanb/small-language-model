"""Main pipeline for document text extraction and processing."""
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import json
import time

from .config import Config, load_config
from .ocr import OCRProcessor, OCRResult
from .preprocessing import ImagePreprocessor, PreprocessingResult
from .ner import DocumentNER, NERResult

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Complete processing result for a document."""
    file_path: str
    ocr_result: OCRResult
    ner_result: NERResult
    preprocessing_steps: List[str]
    processing_time: float
    success: bool
    error_message: Optional[str] = None


class DocumentProcessor:
    """Main document processing pipeline."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the document processor."""
        self.config = load_config(config_path)
        self.ocr_processor = OCRProcessor(self.config.ocr)
        self.preprocessor = ImagePreprocessor(self.config.preprocessing)
        self.ner_processor = DocumentNER(self.config.ner)
        
        # Setup logging
        self._setup_logging()
        
        logger.info("Document processor initialized successfully")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('document_processor.log')
            ]
        )
    
    def process_document(self, file_path: str, output_dir: Optional[str] = None) -> ProcessingResult:
        """Process a single document through the complete pipeline."""
        start_time = time.time()
        file_path = Path(file_path)
        
        try:
            logger.info(f"Starting processing of document: {file_path}")
            
            # Step 1: Preprocessing
            logger.info("Step 1: Image preprocessing")
            preprocessing_result = self.preprocessor.preprocess_from_path(str(file_path))
            
            # Step 2: OCR
            logger.info("Step 2: OCR text extraction")
            ocr_result = self.ocr_processor.process_image_array(preprocessing_result.processed_image)
            
            if not ocr_result.text.strip():
                logger.warning("No text extracted from document")
                return ProcessingResult(
                    file_path=str(file_path),
                    ocr_result=ocr_result,
                    ner_result=NERResult(),
                    preprocessing_steps=preprocessing_result.processing_steps,
                    processing_time=time.time() - start_time,
                    success=False,
                    error_message="No text extracted from document"
                )
            
            # Step 3: NER
            logger.info("Step 3: Named Entity Recognition")
            ner_result = self.ner_processor.process_text(ocr_result.text)
            
            processing_time = time.time() - start_time
            
            # Save results if output directory specified
            if output_dir:
                self._save_results(file_path, ocr_result, ner_result, preprocessing_result, output_dir)
            
            result = ProcessingResult(
                file_path=str(file_path),
                ocr_result=ocr_result,
                ner_result=ner_result,
                preprocessing_steps=preprocessing_result.processing_steps,
                processing_time=processing_time,
                success=True
            )
            
            logger.info(f"Successfully processed {file_path} in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error processing document: {str(e)}"
            logger.error(error_msg)
            
            return ProcessingResult(
                file_path=str(file_path),
                ocr_result=OCRResult("", 0.0),
                ner_result=NERResult(),
                preprocessing_steps=[],
                processing_time=processing_time,
                success=False,
                error_message=error_msg
            )
    
    def process_batch(self, input_dir: str, output_dir: str, file_patterns: List[str] = None) -> Dict[str, ProcessingResult]:
        """Process multiple documents in a directory."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if file_patterns is None:
            file_patterns = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.bmp', '*.pdf']
        
        # Find all matching files
        files = []
        for pattern in file_patterns:
            files.extend(input_path.glob(pattern))
        
        logger.info(f"Found {len(files)} files to process")
        
        results = {}
        for file_path in files:
            result = self.process_document(str(file_path), str(output_path))
            results[str(file_path)] = result
        
        # Generate batch summary
        self._generate_batch_summary(results, output_path)
        
        return results
    
    def _save_results(self, file_path: Path, ocr_result: OCRResult, ner_result: NERResult, 
                     preprocessing_result: PreprocessingResult, output_dir: str):
        """Save processing results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        base_name = file_path.stem
        
        # Save raw text
        with open(output_path / f"{base_name}_text.txt", 'w', encoding='utf-8') as f:
            f.write(ocr_result.text)
        
        # Save structured data as JSON
        if self.config.output.format.lower() == 'json':
            with open(output_path / f"{base_name}_structured.json", 'w', encoding='utf-8') as f:
                json.dump(ner_result.structured_data, f, indent=2, ensure_ascii=False)
        
        # Save entities as CSV
        elif self.config.output.format.lower() == 'csv':
            csv_content = self.ner_processor.export_to_csv(ner_result)
            with open(output_path / f"{base_name}_entities.csv", 'w', encoding='utf-8') as f:
                f.write(csv_content)
        
        # Save preprocessing info
        preprocessing_info = {
            "original_shape": preprocessing_result.original_shape,
            "processing_steps": preprocessing_result.processing_steps,
            "ocr_confidence": ocr_result.confidence,
            "ner_confidence": ner_result.confidence_score
        }
        
        with open(output_path / f"{base_name}_processing_info.json", 'w', encoding='utf-8') as f:
            json.dump(preprocessing_info, f, indent=2)
        
        logger.info(f"Saved results for {file_path.name} to {output_path}")
    
    def _generate_batch_summary(self, results: Dict[str, ProcessingResult], output_path: Path):
        """Generate a summary report for batch processing."""
        successful = sum(1 for r in results.values() if r.success)
        failed = len(results) - successful
        
        summary = {
            "batch_summary": {
                "total_files": len(results),
                "successful": successful,
                "failed": failed,
                "success_rate": successful / len(results) if results else 0,
                "total_processing_time": sum(r.processing_time for r in results.values()),
                "average_processing_time": sum(r.processing_time for r in results.values()) / len(results) if results else 0
            },
            "file_results": {}
        }
        
        for file_path, result in results.items():
            summary["file_results"][file_path] = {
                "success": result.success,
                "processing_time": result.processing_time,
                "ocr_confidence": result.ocr_result.confidence,
                "ner_confidence": result.ner_result.confidence_score,
                "entities_found": len(result.ner_result.entities),
                "error_message": result.error_message
            }
        
        with open(output_path / "batch_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Generated batch summary: {successful}/{len(results)} files processed successfully")
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported image formats."""
        return ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']
    
    def validate_input(self, file_path: str) -> bool:
        """Validate if input file is supported."""
        path = Path(file_path)
        return path.exists() and path.suffix.lower() in self.get_supported_formats()