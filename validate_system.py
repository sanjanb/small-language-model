#!/usr/bin/env python3
"""
System validation script for the document extraction system.
Checks dependencies, installation, and basic functionality.
"""
import sys
import subprocess
import importlib
import platform
from pathlib import Path


def check_python_version():
    """Check Python version."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"   ✓ Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(f"   ✗ Python {version.major}.{version.minor}.{version.micro} (Requires 3.8+)")
        return False


def check_system_dependencies():
    """Check system-level dependencies."""
    print("\n🔧 Checking system dependencies...")
    
    dependencies = {
        'tesseract': 'tesseract --version',
        'git': 'git --version'
    }
    
    results = {}
    
    for dep, command in dependencies.items():
        try:
            result = subprocess.run(command.split(), 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            if result.returncode == 0:
                version = result.stdout.split('\n')[0]
                print(f"   ✓ {dep}: {version}")
                results[dep] = True
            else:
                print(f"   ✗ {dep}: Not found or error")
                results[dep] = False
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            print(f"   ⚠ {dep}: Not found (optional for some features)")
            results[dep] = False
    
    return results


def check_python_packages():
    """Check Python package dependencies."""
    print("\n📦 Checking Python packages...")
    
    required_packages = [
        'numpy', 'opencv-python', 'Pillow', 'pydantic', 'click', 'PyYAML'
    ]
    
    optional_packages = [
        'pytesseract', 'easyocr', 'spacy', 'transformers', 'torch', 'pandas'
    ]
    
    results = {'required': {}, 'optional': {}}
    
    # Check required packages
    print("   Required packages:")
    for package in required_packages:
        try:
            # Handle different import names
            import_name = package
            if package == 'opencv-python':
                import_name = 'cv2'
            elif package == 'Pillow':
                import_name = 'PIL'
            elif package == 'PyYAML':
                import_name = 'yaml'
            
            module = importlib.import_module(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"     ✓ {package}: {version}")
            results['required'][package] = True
        except ImportError:
            print(f"     ✗ {package}: Not installed")
            results['required'][package] = False
    
    # Check optional packages
    print("   Optional packages:")
    for package in optional_packages:
        try:
            import_name = package
            if package == 'opencv-python':
                import_name = 'cv2'
            elif package == 'Pillow':
                import_name = 'PIL'
            
            module = importlib.import_module(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"     ✓ {package}: {version}")
            results['optional'][package] = True
        except ImportError:
            print(f"     ⚠ {package}: Not installed (optional)")
            results['optional'][package] = False
    
    return results


def check_spacy_models():
    """Check spaCy language models."""
    print("\n🧠 Checking spaCy models...")
    
    try:
        import spacy
        
        models = ['en_core_web_sm', 'en_core_web_md', 'en_core_web_lg']
        found_models = []
        
        for model in models:
            try:
                nlp = spacy.load(model)
                print(f"   ✓ {model}: Available")
                found_models.append(model)
            except OSError:
                print(f"   ⚠ {model}: Not installed")
        
        if found_models:
            print(f"   Found {len(found_models)} spaCy model(s)")
            return True
        else:
            print(f"   ⚠ No spaCy models found. Run: python -m spacy download en_core_web_sm")
            return False
            
    except ImportError:
        print(f"   ✗ spaCy not installed")
        return False


def test_basic_functionality():
    """Test basic system functionality."""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Add src to path
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        
        # Test configuration
        try:
            from document_extractor.config import Config
            config = Config()
            print(f"   ✓ Configuration system working")
        except Exception as e:
            print(f"   ✗ Configuration test failed: {e}")
            return False
        
        # Test NER with sample text
        try:
            from document_extractor.ner import DocumentNER
            ner = DocumentNER(config.ner)
            
            sample_text = "Invoice INV-123 for $100.00 on 2024-01-01"
            result = ner.process_text(sample_text)
            
            if len(result.entities) > 0:
                print(f"   ✓ NER working: found {len(result.entities)} entities")
            else:
                print(f"   ⚠ NER working but no entities found in test")
            
        except Exception as e:
            print(f"   ✗ NER test failed: {e}")
            return False
        
        # Test OCR result structure
        try:
            from document_extractor.ocr import OCRResult
            result = OCRResult("test", 0.9)
            print(f"   ✓ OCR result structure working")
        except Exception as e:
            print(f"   ✗ OCR test failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ✗ Basic functionality test failed: {e}")
        return False


def check_file_permissions():
    """Check file permissions in current directory."""
    print("\n📁 Checking file permissions...")
    
    try:
        # Test write permissions
        test_file = Path("test_write_permission.tmp")
        test_file.write_text("test")
        test_file.unlink()
        print("   ✓ Write permissions: OK")
        
        # Check if src directory exists
        src_dir = Path("src")
        if src_dir.exists():
            print("   ✓ Source directory: Found")
        else:
            print("   ⚠ Source directory: Not found")
        
        # Check if examples directory exists
        examples_dir = Path("examples")
        if examples_dir.exists():
            print("   ✓ Examples directory: Found")
        else:
            print("   ⚠ Examples directory: Not found")
        
        return True
        
    except Exception as e:
        print(f"   ✗ File permissions check failed: {e}")
        return False


def print_installation_instructions(results):
    """Print installation instructions based on what's missing."""
    print("\n📋 Installation Instructions:")
    print("=" * 50)
    
    # Python packages
    missing_required = [pkg for pkg, installed in results['packages']['required'].items() if not installed]
    missing_optional = [pkg for pkg, installed in results['packages']['optional'].items() if not installed]
    
    if missing_required or missing_optional:
        print("\n1. Install Python packages:")
        if missing_required or missing_optional:
            print("   pip install -r requirements.txt")
        
        if not results['spacy_models']:
            print("\n2. Install spaCy language model:")
            print("   python -m spacy download en_core_web_sm")
    
    if not results['system']['tesseract']:
        print("\n3. Install Tesseract OCR (optional but recommended):")
        if platform.system() == "Linux":
            print("   sudo apt-get install tesseract-ocr")
        elif platform.system() == "Darwin":
            print("   brew install tesseract")
        elif platform.system() == "Windows":
            print("   Download from: https://github.com/UB-Mannheim/tesseract/wiki")
    
    print("\n4. Test the installation:")
    print("   python validate_system.py")
    print("   python demo.py")
    print("   python main.py --help")


def main():
    """Main validation function."""
    print("🔍 Document Extraction System - System Validation")
    print("=" * 60)
    
    results = {
        'python_version': check_python_version(),
        'system': check_system_dependencies(),
        'packages': check_python_packages(),
        'spacy_models': check_spacy_models(),
        'functionality': test_basic_functionality(),
        'permissions': check_file_permissions()
    }
    
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    # Count checks
    total_checks = 0
    passed_checks = 0
    
    # Python version
    total_checks += 1
    if results['python_version']:
        passed_checks += 1
        print("✓ Python version: PASS")
    else:
        print("✗ Python version: FAIL")
    
    # System dependencies
    for dep, status in results['system'].items():
        total_checks += 1
        if status:
            passed_checks += 1
            print(f"✓ {dep}: PASS")
        else:
            print(f"⚠ {dep}: OPTIONAL")
    
    # Required packages
    required_ok = all(results['packages']['required'].values())
    total_checks += 1
    if required_ok:
        passed_checks += 1
        print("✓ Required packages: PASS")
    else:
        print("✗ Required packages: FAIL")
    
    # SpaCy models
    total_checks += 1
    if results['spacy_models']:
        passed_checks += 1
        print("✓ SpaCy models: PASS")
    else:
        print("⚠ SpaCy models: OPTIONAL")
    
    # Basic functionality
    total_checks += 1
    if results['functionality']:
        passed_checks += 1
        print("✓ Basic functionality: PASS")
    else:
        print("✗ Basic functionality: FAIL")
    
    # File permissions
    total_checks += 1
    if results['permissions']:
        passed_checks += 1
        print("✓ File permissions: PASS")
    else:
        print("✗ File permissions: FAIL")
    
    print(f"\nScore: {passed_checks}/{total_checks} checks passed")
    
    if passed_checks == total_checks:
        print("\n🎉 All checks passed! The system is ready to use.")
        print("   Run 'python main.py --help' to get started.")
        return True
    elif required_ok and results['python_version'] and results['functionality']:
        print("\n⚠ Core functionality is working, but some optional components are missing.")
        print("  The system should work for basic text processing.")
        print_installation_instructions(results)
        return True
    else:
        print("\n❌ Critical components are missing. Please install dependencies.")
        print_installation_instructions(results)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)