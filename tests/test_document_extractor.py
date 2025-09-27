"""Unit tests for the document extraction system."""
import unittest
import sys
from pathlib import Path
import tempfile
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestConfig(unittest.TestCase):
    """Test configuration management."""
    
    def test_default_config_creation(self):
        """Test creating default configuration."""
        try:
            # Import inside try block to handle missing dependencies gracefully
            from document_extractor.config import Config
            config = Config()
            self.assertIsNotNone(config)
            self.assertEqual(config.ocr.engine, "tesseract")
            self.assertEqual(config.output.format, "json")
        except ImportError as e:
            self.skipTest(f"Dependencies not installed: {e}")
    
    def test_config_yaml_roundtrip(self):
        """Test saving and loading config to/from YAML."""
        try:
            from document_extractor.config import Config
            import tempfile
            
            config = Config()
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                config.to_yaml(f.name)
                
                # Load it back
                loaded_config = Config.from_yaml(f.name)
                self.assertEqual(config.ocr.engine, loaded_config.ocr.engine)
                self.assertEqual(config.output.format, loaded_config.output.format)
                
                # Clean up
                Path(f.name).unlink()
                
        except ImportError as e:
            self.skipTest(f"Dependencies not installed: {e}")


class TestNER(unittest.TestCase):
    """Test Named Entity Recognition functionality."""
    
    def setUp(self):
        """Setup test data."""
        self.sample_text = """
        Invoice #INV-12345
        Date: January 15, 2024
        Amount: $1,234.56
        Customer: John Smith
        Company: Acme Corp
        Email: john@example.com
        Phone: (555) 123-4567
        """
    
    def test_ner_entity_extraction(self):
        """Test entity extraction from sample text."""
        try:
            from document_extractor.config import Config
            from document_extractor.ner import DocumentNER
            
            config = Config()
            ner = DocumentNER(config.ner)
            
            result = ner.process_text(self.sample_text)
            
            self.assertIsNotNone(result)
            self.assertGreater(len(result.entities), 0)
            
            # Check that we found some expected entities
            entity_texts = [e.text for e in result.entities]
            entity_labels = [e.label for e in result.entities]
            
            # Should find at least some entities
            self.assertTrue(any("john@example.com" in text.lower() for text in entity_texts))
            self.assertTrue(any("EMAIL" in label for label in entity_labels))
            
        except ImportError as e:
            self.skipTest(f"Dependencies not installed: {e}")
    
    def test_ner_structured_output(self):
        """Test structured data output."""
        try:
            from document_extractor.config import Config
            from document_extractor.ner import DocumentNER
            
            config = Config()
            ner = DocumentNER(config.ner)
            
            result = ner.process_text(self.sample_text)
            
            self.assertIn("document_type", result.structured_data)
            self.assertIn("entities_by_type", result.structured_data)
            self.assertIn("key_information", result.structured_data)
            
        except ImportError as e:
            self.skipTest(f"Dependencies not installed: {e}")


class TestPreprocessing(unittest.TestCase):
    """Test image preprocessing functionality."""
    
    def test_preprocessing_result_structure(self):
        """Test preprocessing result structure."""
        try:
            from document_extractor.config import Config
            from document_extractor.preprocessing import ImagePreprocessor
            import numpy as np
            
            config = Config()
            preprocessor = ImagePreprocessor(config.preprocessing)
            
            # Create a dummy image
            dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            
            result = preprocessor.preprocess(dummy_image)
            
            self.assertIsNotNone(result.processed_image)
            self.assertIsNotNone(result.original_shape)
            self.assertIsInstance(result.processing_steps, list)
            self.assertGreater(len(result.processing_steps), 0)
            
        except ImportError as e:
            self.skipTest(f"Dependencies not installed: {e}")


class TestOCR(unittest.TestCase):
    """Test OCR functionality."""
    
    def test_ocr_result_structure(self):
        """Test OCR result structure."""
        try:
            from document_extractor.ocr import OCRResult
            
            result = OCRResult("Sample text", 0.95)
            
            self.assertEqual(result.text, "Sample text")
            self.assertEqual(result.confidence, 0.95)
            self.assertIsNone(result.bbox)
            
        except ImportError as e:
            self.skipTest(f"Dependencies not installed: {e}")


class TestPipeline(unittest.TestCase):
    """Test the main processing pipeline."""
    
    def test_pipeline_initialization(self):
        """Test pipeline can be initialized."""
        try:
            from document_extractor.pipeline import DocumentProcessor
            
            # This should not raise an exception
            processor = DocumentProcessor()
            self.assertIsNotNone(processor)
            self.assertIsNotNone(processor.config)
            
        except ImportError as e:
            self.skipTest(f"Dependencies not installed: {e}")
    
    def test_supported_formats(self):
        """Test supported file formats."""
        try:
            from document_extractor.pipeline import DocumentProcessor
            
            processor = DocumentProcessor()
            formats = processor.get_supported_formats()
            
            self.assertIn('.jpg', formats)
            self.assertIn('.png', formats)
            self.assertIsInstance(formats, list)
            
        except ImportError as e:
            self.skipTest(f"Dependencies not installed: {e}")


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_full_text_processing_pipeline(self):
        """Test processing text through NER pipeline."""
        try:
            from document_extractor.config import Config
            from document_extractor.ner import DocumentNER
            
            config = Config()
            ner = DocumentNER(config.ner)
            
            sample_text = "Invoice INV-2024-001 for $500.00 dated March 15, 2024"
            result = ner.process_text(sample_text)
            
            # Should extract some entities
            self.assertGreater(len(result.entities), 0)
            
            # Should have structured data
            self.assertIn("document_type", result.structured_data)
            
            # Should be able to export to JSON
            json_output = ner.export_to_json(result)
            parsed = json.loads(json_output)
            self.assertIsInstance(parsed, dict)
            
        except ImportError as e:
            self.skipTest(f"Dependencies not installed: {e}")


if __name__ == '__main__':
    print("Running document extraction system tests...")
    print("=" * 50)
    
    # Create test suite
    test_loader = unittest.TestLoader()
    test_suite = test_loader.loadTestsFromModule(sys.modules[__name__])
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    if result.skipped:
        print(f"\nSKIPPED (likely due to missing dependencies):")
        for test, reason in result.skipped:
            print(f"- {test}: {reason}")
    
    if result.wasSuccessful():
        print(f"\n🎉 All tests passed!")
    else:
        print(f"\n❌ Some tests failed. Install dependencies and check the code.")
        sys.exit(1)