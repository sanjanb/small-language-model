#!/usr/bin/env python3
"""
Setup script for the Document Text Extraction system.
Creates directories, checks dependencies, and initializes the project.
"""

import os
import sys
import subprocess
from pathlib import Path
import importlib.util


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True


def create_directories():
    """Create necessary project directories."""
    directories = [
        "data/raw",
        "data/processed", 
        "models",
        "results/plots",
        "results/metrics",
        "logs"
    ]
    
    print("\n📁 Creating project directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}")


def check_dependencies():
    """Check if required dependencies are installed."""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('PIL', 'Pillow'),
        ('cv2', 'OpenCV'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('sklearn', 'Scikit-learn')
    ]
    
    missing_packages = []
    
    for package, name in required_packages:
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing_packages.append(name)
            print(f"   ❌ {name} not found")
        else:
            print(f"   ✅ {name}")
    
    return missing_packages


def check_ocr_dependencies():
    """Check OCR-related dependencies."""
    print("\n🔍 Checking OCR dependencies...")
    
    # Check EasyOCR
    try:
        import easyocr
        print("   ✅ EasyOCR")
    except ImportError:
        print("   ❌ EasyOCR not found")
    
    # Check Tesseract
    try:
        import pytesseract
        print("   ✅ PyTesseract")
        
        # Try to run tesseract
        try:
            pytesseract.get_tesseract_version()
            print("   ✅ Tesseract OCR engine")
        except Exception:
            print("   ⚠️  Tesseract OCR engine not found or not in PATH")
            print("      Please install Tesseract OCR:")
            print("      - Windows: https://github.com/UB-Mannheim/tesseract/wiki")
            print("      - Ubuntu: sudo apt install tesseract-ocr")
            print("      - macOS: brew install tesseract")
            
    except ImportError:
        print("   ❌ PyTesseract not found")


def install_dependencies():
    """Install missing dependencies."""
    print("\n🔧 Installing dependencies from requirements.txt...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True, check=True)
        
        print("   ✅ Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to install dependencies: {e}")
        print(f"   Output: {e.stdout}")
        print(f"   Error: {e.stderr}")
        return False


def check_gpu_support():
    """Check if GPU support is available."""
    print("\n🖥️  Checking GPU support...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"   ✅ CUDA available - {gpu_count} GPU(s)")
            print(f"   🎯 Primary GPU: {gpu_name}")
        else:
            print("   ⚠️  CUDA not available - will use CPU")
    except ImportError:
        print("   ❌ PyTorch not installed")


def create_sample_documents():
    """Create sample documents for testing."""
    print("\n📄 Creating sample test documents...")
    
    sample_texts = [
        "Invoice sent to John Doe on 01/15/2025\nInvoice No: INV-1001\nAmount: $1,500.00\nPhone: (555) 123-4567",
        "Bill for Dr. Sarah Johnson dated March 10, 2025.\nInvoice Number: BL-2045.\nTotal: $2,300.50\nEmail: sarah@email.com",
        "Receipt for Michael Brown\n456 Oak Street, Boston MA 02101\nInvoice: REC-3089\nDate: 2025-04-22\nAmount: $890.75"
    ]
    
    sample_dir = Path("data/raw/samples")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    for i, text in enumerate(sample_texts, 1):
        sample_file = sample_dir / f"sample_document_{i}.txt"
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"   ✅ {sample_file.name}")


def run_initial_test():
    """Run a basic test to verify setup."""
    print("\n🧪 Running initial setup test...")
    
    try:
        # Test imports
        from src.data_preparation import DocumentProcessor, NERDatasetCreator
        from src.model import ModelConfig
        print("   ✅ Core modules imported successfully")
        
        # Test document processor
        processor = DocumentProcessor()
        test_text = "Invoice sent to John Doe on 01/15/2025 Amount: $500.00"
        cleaned_text = processor.clean_text(test_text)
        print("   ✅ Document processor working")
        
        # Test dataset creator
        dataset_creator = NERDatasetCreator(processor)
        sample_dataset = dataset_creator.create_sample_dataset()
        print(f"   ✅ Dataset creator working - {len(sample_dataset)} samples")
        
        # Test model config
        config = ModelConfig()
        print(f"   ✅ Model config created - {config.num_labels} labels")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Setup test failed: {e}")
        return False


def display_next_steps():
    """Display next steps for the user."""
    print("\n" + "🎉" * 30)
    print("SETUP COMPLETED SUCCESSFULLY!")
    print("🎉" * 30)
    
    print("\n📋 Next Steps:")
    print("1. 🚀 Quick Demo:")
    print("   python demo.py")
    
    print("\n2. 🔧 Train Your Model:")
    print("   # Add your documents to data/raw/")
    print("   # Then run:")
    print("   python src/training_pipeline.py")
    
    print("\n3. 🌐 Start Web API:")
    print("   python api/app.py")
    print("   # Then open: http://localhost:8000")
    
    print("\n4. 🧪 Run Tests:")
    print("   python tests/test_extraction.py")
    
    print("\n5. 📚 Documentation:")
    print("   # View README.md for detailed usage")
    print("   # API docs: http://localhost:8000/docs")
    
    print("\n💡 Pro Tips:")
    print("   - Place your documents in data/raw/ for training")
    print("   - Use GPU for faster training (if available)")
    print("   - Adjust batch_size in config if you get memory errors")
    print("   - Check logs/ directory for debugging information")


def main():
    """Main setup function."""
    print("🎯 DOCUMENT TEXT EXTRACTION - SETUP SCRIPT")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create directories
    create_directories()
    
    # Check and install dependencies
    missing_packages = check_dependencies()
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        install_deps = input("Install missing dependencies? (y/n): ").lower().strip()
        
        if install_deps == 'y':
            if not install_dependencies():
                print("❌ Failed to install dependencies. Please install manually:")
                print("   pip install -r requirements.txt")
                return False
        else:
            print("⚠️  Some features may not work without required dependencies.")
    
    # Check OCR dependencies
    check_ocr_dependencies()
    
    # Check GPU support
    check_gpu_support()
    
    # Create sample documents
    create_sample_documents()
    
    # Run initial test
    if not run_initial_test():
        print("⚠️  Setup test failed. Some features may not work correctly.")
        print("   Check error messages above and ensure all dependencies are installed.")
    
    # Display next steps
    display_next_steps()
    
    return True


if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n✅ Setup completed! Ready to extract text from documents! 🚀")
    else:
        print(f"\n❌ Setup encountered issues. Please check the messages above.")
        sys.exit(1)