# Document Text Extraction System

An end-to-end system for automated document text extraction using OCR technology, image preprocessing, and Named Entity Recognition (NER) with Small Language Models. The system can extract and structure key information from documents such as invoices, forms, reports, and scanned papers into machine-readable formats.

## Features

- **OCR Integration**: Support for both Tesseract and EasyOCR engines
- **Image Preprocessing**: Advanced preprocessing pipeline for better OCR results
  - Image resizing and enhancement
  - Noise reduction and contrast enhancement  
  - Adaptive thresholding
- **Named Entity Recognition**: Extract structured information using spaCy and custom patterns
  - Pre-trained models for common entities (PERSON, ORG, DATE, MONEY)
  - Custom regex patterns for document-specific entities (INVOICE_NUMBER, AMOUNT, etc.)
- **Multiple Output Formats**: JSON and CSV export options
- **Batch Processing**: Process multiple documents in parallel
- **CLI Interface**: Easy-to-use command-line interface
- **Configurable Pipeline**: YAML-based configuration system

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sanjanb/small-language-model.git
cd small-language-model
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR (optional, for Tesseract engine):
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows - download from: https://github.com/UB-Mannheim/tesseract/wiki
```

4. Download spaCy language model:
```bash
python -m spacy download en_core_web_sm
```

## Quick Start

1. **Test the installation**:
```bash
python test_basic.py
```

2. **Process a single document**:
```bash
python main.py process path/to/document.jpg --output results/
```

3. **Process multiple documents**:
```bash
python main.py batch input_folder/ output_folder/
```

4. **Create a configuration file**:
```bash
python main.py init-config my_config.yaml
```

## Usage

### Command Line Interface

The system provides a comprehensive CLI with the following commands:

#### Process Single Document
```bash
python main.py process INPUT_FILE [OPTIONS]

Options:
  -o, --output PATH    Output directory for results
  -f, --format [json|csv]  Output format (default: json)
  -c, --config PATH   Configuration file path
  -v, --verbose       Enable verbose logging
```

Example:
```bash
python main.py process invoice.jpg -o results/ -f json
```

#### Batch Processing
```bash
python main.py batch INPUT_DIR OUTPUT_DIR [OPTIONS]

Options:
  -p, --pattern TEXT   File patterns to process (e.g., *.jpg, *.png)
  -f, --format [json|csv]  Output format (default: json)
```

Example:
```bash
python main.py batch documents/ results/ -p "*.jpg" -p "*.png"
```

#### Other Commands
- `init-config CONFIG_PATH` - Create default configuration file
- `info` - Display system information
- `validate INPUT_FILE` - Check if file can be processed

### Python API

You can also use the system programmatically:

```python
from document_extractor.pipeline import DocumentProcessor

# Initialize processor
processor = DocumentProcessor("config.yaml")

# Process single document
result = processor.process_document("invoice.jpg", "output/")

# Access results
print(f"Extracted text: {result.ocr_result.text}")
print(f"Entities found: {len(result.ner_result.entities)}")
print(f"Key information: {result.ner_result.structured_data['key_information']}")

# Batch processing
results = processor.process_batch("input/", "output/")
```

## Configuration

The system uses YAML configuration files. Create one with:

```bash
python main.py init-config config.yaml
```

Configuration sections:

### OCR Settings
```yaml
ocr:
  engine: "tesseract"  # or "easyocr"
  tesseract_config: "--oem 3 --psm 6"
  language: "eng"
  confidence_threshold: 0.6
```

### Preprocessing Settings
```yaml
preprocessing:
  resize_factor: 2.0
  apply_denoising: true
  enhance_contrast: true
  threshold_method: "adaptive"
```

### NER Settings
```yaml
ner:
  model_name: "en_core_web_sm"
  custom_entities:
    - "INVOICE_NUMBER"
    - "AMOUNT"
    - "DATE"
  confidence_threshold: 0.8
```

### Output Settings
```yaml
output:
  format: "json"
  structured: true
  include_confidence: true
```

## Supported Document Types

The system can process various document types:

- **Invoices**: Extracts invoice numbers, amounts, dates, vendor information
- **Forms**: Identifies form fields and their values
- **Reports**: Extracts key metrics and structured data
- **Receipts**: Captures transaction details
- **Contracts**: Identifies parties, dates, and key terms

## Supported File Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- TIFF (.tiff, .tif)
- BMP (.bmp)

## Output Examples

### JSON Output
```json
{
  "document_type": "invoice",
  "entities_by_type": {
    "INVOICE_NUMBER": [
      {
        "text": "INV-12345",
        "confidence": 0.95,
        "position": {"start": 15, "end": 24}
      }
    ],
    "AMOUNT": [
      {
        "text": "$1,234.56",
        "confidence": 0.92,
        "position": {"start": 45, "end": 54}
      }
    ]
  },
  "key_information": {
    "invoice_number": {
      "value": "INV-12345",
      "confidence": 0.95
    },
    "total_amount": {
      "value": "$1,234.56", 
      "confidence": 0.92
    }
  }
}
```

### CSV Output
```csv
text,label,start,end,confidence
INV-12345,INVOICE_NUMBER,15,24,0.95
$1234.56,AMOUNT,45,54,0.92
2024-01-15,DATE,60,70,0.88
```

## Project Structure

```
small-language-model/
├── src/document_extractor/
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── pipeline.py        # Main processing pipeline
│   ├── cli.py            # Command-line interface
│   ├── ocr/              # OCR engines
│   ├── preprocessing/    # Image preprocessing
│   └── ner/              # Named entity recognition
├── examples/
├── tests/
├── main.py               # Entry point
├── config.yaml           # Default configuration
├── requirements.txt      # Python dependencies
└── README.md
```

## Development

### Adding Custom Entity Types

1. Add patterns to the NER configuration:
```yaml
ner:
  custom_entities:
    - "CUSTOM_ENTITY"
```

2. Add regex patterns in `src/document_extractor/ner/__init__.py`:
```python
patterns = {
    "CUSTOM_ENTITY": [
        r"pattern1",
        r"pattern2"
    ]
}
```

### Extending OCR Engines

Inherit from the `OCREngine` base class:
```python
class CustomOCR(OCREngine):
    def extract_text(self, image: np.ndarray) -> OCRResult:
        # Implement custom OCR logic
        pass
```

## Requirements

- Python 3.8+
- OpenCV
- Tesseract OCR (optional)
- spaCy with English model
- See `requirements.txt` for complete list

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Troubleshooting

### Common Issues

1. **"Tesseract not found"**: Install Tesseract OCR system package
2. **"spaCy model not found"**: Run `python -m spacy download en_core_web_sm`
3. **Low OCR accuracy**: Try adjusting preprocessing settings or using EasyOCR
4. **Memory issues**: Process images in smaller batches

### Performance Tips

- Use image preprocessing to improve OCR accuracy
- Adjust confidence thresholds based on your data quality
- Use EasyOCR for better accuracy on low-quality images
- Enable verbose logging to debug processing issues
