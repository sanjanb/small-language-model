"""Simple test to verify the basic functionality without external dependencies."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    try:
        from document_extractor.config import Config, load_config
        from document_extractor.pipeline import DocumentProcessor
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_config():
    """Test configuration loading."""
    try:
        from document_extractor.config import Config
        config = Config()
        print(f"✓ Default config created: OCR engine = {config.ocr.engine}")
        return True
    except Exception as e:
        print(f"✗ Config error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without processing actual images."""
    try:
        from document_extractor.config import Config
        from document_extractor.ner import DocumentNER, NERResult
        
        # Test NER on sample text
        config = Config()
        ner = DocumentNER(config.ner)
        
        sample_text = """
        Invoice #INV-12345
        Date: 2024-01-15
        Amount: $1,234.56
        Contact: john@example.com
        Phone: (555) 123-4567
        """
        
        result = ner.process_text(sample_text)
        print(f"✓ NER processing successful: found {len(result.entities)} entities")
        
        # Print found entities
        for entity in result.entities[:3]:  # Show first 3
            print(f"   - {entity.label}: {entity.text}")
        
        return True
    except Exception as e:
        print(f"✗ NER test error: {e}")
        return False

if __name__ == "__main__":
    print("Running basic functionality tests...")
    print("=" * 40)
    
    tests = [test_imports, test_config, test_basic_functionality]
    passed = 0
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! The system is ready to use.")
    else:
        print("❌ Some tests failed. Check the requirements and installation.")