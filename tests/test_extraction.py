"""
Test cases for the document text extraction system.
"""

import unittest
import json
from pathlib import Path
import tempfile
import os

from src.data_preparation import DocumentProcessor, NERDatasetCreator
from src.model import ModelConfig, create_model_and_trainer
from src.inference import DocumentInference


class TestDocumentProcessor(unittest.TestCase):
    """Test cases for document processing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = DocumentProcessor()
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        dirty_text = "  This   is    a    test  text!!!  "
        clean_text = self.processor.clean_text(dirty_text)
        self.assertEqual(clean_text, "This is a test text!")
    
    def test_entity_patterns(self):
        """Test entity pattern matching."""
        test_text = "Invoice sent to John Doe on 01/15/2025 Invoice No: INV-1001 Amount: $1,500.00"
        
        # Test that patterns exist
        self.assertIn('NAME', self.processor.entity_patterns)
        self.assertIn('DATE', self.processor.entity_patterns)
        self.assertIn('INVOICE_NO', self.processor.entity_patterns)
        self.assertIn('AMOUNT', self.processor.entity_patterns)


class TestNERDatasetCreator(unittest.TestCase):
    """Test cases for NER dataset creation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = DocumentProcessor()
        self.dataset_creator = NERDatasetCreator(self.processor)
    
    def test_auto_label_text(self):
        """Test automatic text labeling."""
        test_text = "Invoice sent to Robert White on 15/09/2025 Amount: $1,250"
        labeled_tokens = self.dataset_creator.auto_label_text(test_text)
        
        # Check that we get tokens and labels
        self.assertIsInstance(labeled_tokens, list)
        self.assertGreater(len(labeled_tokens), 0)
        
        # Check that each item is a (token, label) tuple
        for token, label in labeled_tokens:
            self.assertIsInstance(token, str)
            self.assertIsInstance(label, str)
    
    def test_create_training_example(self):
        """Test training example creation."""
        test_text = "Invoice INV-1001 for $500"
        example = self.dataset_creator.create_training_example(test_text)
        
        # Check required fields
        self.assertIn('tokens', example)
        self.assertIn('labels', example)
        self.assertIn('text', example)
        
        # Check that tokens and labels have the same length
        self.assertEqual(len(example['tokens']), len(example['labels']))
    
    def test_create_sample_dataset(self):
        """Test sample dataset creation."""
        dataset = self.dataset_creator.create_sample_dataset()
        
        # Check that we get a non-empty dataset
        self.assertIsInstance(dataset, list)
        self.assertGreater(len(dataset), 0)
        
        # Check first example structure
        first_example = dataset[0]
        self.assertIn('tokens', first_example)
        self.assertIn('labels', first_example)
        self.assertIn('text', first_example)


class TestModelConfig(unittest.TestCase):
    """Test cases for model configuration."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = ModelConfig()
        
        # Check default values
        self.assertEqual(config.model_name, "distilbert-base-uncased")
        self.assertEqual(config.max_length, 512)
        self.assertEqual(config.batch_size, 16)
        
        # Check entity labels
        self.assertIsInstance(config.entity_labels, list)
        self.assertGreater(len(config.entity_labels), 0)
        self.assertIn('O', config.entity_labels)
        
        # Check label mappings
        self.assertIsInstance(config.label2id, dict)
        self.assertIsInstance(config.id2label, dict)
        self.assertEqual(len(config.label2id), len(config.entity_labels))
    
    def test_custom_config(self):
        """Test custom configuration."""
        custom_labels = ['O', 'B-TEST', 'I-TEST']
        config = ModelConfig(
            batch_size=32,
            learning_rate=1e-5,
            entity_labels=custom_labels
        )
        
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.learning_rate, 1e-5)
        self.assertEqual(config.entity_labels, custom_labels)
        self.assertEqual(config.num_labels, 3)


class TestModelCreation(unittest.TestCase):
    """Test cases for model creation."""
    
    def test_create_model_and_trainer(self):
        """Test model and trainer creation."""
        config = ModelConfig(
            batch_size=4,  # Small batch for testing
            num_epochs=1,
            entity_labels=['O', 'B-TEST', 'I-TEST']
        )
        
        model, trainer = create_model_and_trainer(config)
        
        # Check that objects are created
        self.assertIsNotNone(model)
        self.assertIsNotNone(trainer)
        
        # Check configuration
        self.assertEqual(trainer.config.batch_size, 4)
        self.assertEqual(trainer.config.num_epochs, 1)


class TestInference(unittest.TestCase):
    """Test cases for inference pipeline."""
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level fixtures."""
        # Create a minimal trained model for testing
        # This is a placeholder - in real testing, you'd use a pre-trained model
        cls.model_path = "test_model"
        cls.test_text = "Invoice sent to John Doe on 01/15/2025 Amount: $500.00"
    
    def test_entity_validation(self):
        """Test entity validation patterns."""
        # We can test the patterns without loading a full model
        test_patterns = {
            'DATE': ['01/15/2025', '2025-01-15', 'January 15, 2025'],
            'AMOUNT': ['$500.00', '$1,250.50', '1000.00 USD'],
            'EMAIL': ['test@email.com', 'user.name@domain.co.uk'],
            'PHONE': ['(555) 123-4567', '+1-555-987-6543', '555-123-4567']
        }
        
        # This test checks that our regex patterns work
        import re
        
        date_pattern = r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b'
        self.assertTrue(re.search(date_pattern, '01/15/2025'))
        
        amount_pattern = r'\$\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?'
        self.assertTrue(re.search(amount_pattern, '$1,250.50'))
        
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        self.assertTrue(re.search(email_pattern, 'test@email.com'))


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests."""
    
    def test_data_preparation_flow(self):
        """Test the complete data preparation flow."""
        # Create processor and dataset creator
        processor = DocumentProcessor()
        dataset_creator = NERDatasetCreator(processor)
        
        # Create sample dataset
        dataset = dataset_creator.create_sample_dataset()
        
        # Verify dataset structure
        self.assertIsInstance(dataset, list)
        self.assertGreater(len(dataset), 0)
        
        for example in dataset:
            self.assertIn('tokens', example)
            self.assertIn('labels', example)
            self.assertIn('text', example)
            self.assertEqual(len(example['tokens']), len(example['labels']))
    
    def test_model_config_flow(self):
        """Test model configuration and creation flow."""
        # Create configuration
        config = ModelConfig(batch_size=4, num_epochs=1)
        
        # Create model and trainer
        model, trainer = create_model_and_trainer(config)
        
        # Verify objects exist and have correct configuration
        self.assertIsNotNone(model)
        self.assertIsNotNone(trainer)
        self.assertEqual(trainer.config.batch_size, 4)
        self.assertEqual(trainer.config.num_epochs, 1)
    
    def test_save_and_load_dataset(self):
        """Test saving and loading dataset."""
        # Create dataset
        processor = DocumentProcessor()
        dataset_creator = NERDatasetCreator(processor)
        dataset = dataset_creator.create_sample_dataset()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
            json.dump(dataset, f, indent=2)
        
        try:
            # Load and verify
            with open(temp_path, 'r') as f:
                loaded_dataset = json.load(f)
            
            self.assertEqual(len(loaded_dataset), len(dataset))
            self.assertEqual(loaded_dataset[0]['text'], dataset[0]['text'])
            
        finally:
            # Clean up
            os.unlink(temp_path)


def run_tests():
    """Run all tests."""
    print("🧪 Running Document Text Extraction Tests")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestDocumentProcessor,
        TestNERDatasetCreator,
        TestModelConfig,
        TestModelCreation,
        TestInference,
        TestEndToEnd
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    if result.wasSuccessful():
        print(f"\n✅ All tests passed! ({result.testsRun} tests)")
    else:
        print(f"\n❌ {len(result.failures)} failures, {len(result.errors)} errors")
        
        if result.failures:
            print("\nFailures:")
            for test, failure in result.failures:
                print(f"  {test}: {failure}")
        
        if result.errors:
            print("\nErrors:")
            for test, error in result.errors:
                print(f"  {test}: {error}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_tests()