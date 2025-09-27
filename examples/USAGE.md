# Usage Examples

This document provides comprehensive examples of how to use the Document Text Extraction System.

## Basic Usage

### 1. Processing a Single Document

```bash
# Process a single image file
python main.py process invoice.jpg -o results/

# Process with specific output format
python main.py process receipt.png -f csv -o results/
```

### 2. Batch Processing

```bash
# Process all images in a directory
python main.py batch input_folder/ output_folder/

# Process specific file types
python main.py batch documents/ results/ -p "*.jpg" -p "*.png"
```

### 3. Using Custom Configuration

```bash
# Create a custom config file
python main.py init-config my_config.yaml

# Edit the config file to your needs
# Then use it:
python main.py process invoice.jpg -c my_config.yaml -o results/
```

## Python API Examples

### Basic Processing

```python
from document_extractor.pipeline import DocumentProcessor

# Initialize with default configuration
processor = DocumentProcessor()

# Process a single document
result = processor.process_document("path/to/document.jpg", "output/")

if result.success:
    print(f"Processing completed in {result.processing_time:.2f}s")
    print(f"OCR confidence: {result.ocr_result.confidence:.2f}")
    print(f"Extracted text: {result.ocr_result.text}")
    print(f"Entities found: {len(result.ner_result.entities)}")
else:
    print(f"Processing failed: {result.error_message}")
```

### Batch Processing

```python
# Process multiple documents
results = processor.process_batch("input_dir/", "output_dir/")

# Analyze results
successful = sum(1 for r in results.values() if r.success)
print(f"Successfully processed {successful}/{len(results)} files")

# Get detailed results for each file
for file_path, result in results.items():
    if result.success:
        print(f"✓ {file_path}: {len(result.ner_result.entities)} entities")
    else:
        print(f"✗ {file_path}: {result.error_message}")
```

### Custom Configuration

```python
from document_extractor.config import Config
from document_extractor.pipeline import DocumentProcessor

# Create custom configuration
config = Config()
config.ocr.engine = "easyocr"  # Use EasyOCR instead of Tesseract
config.ocr.confidence_threshold = 0.8
config.ner.custom_entities = ["INVOICE_NUMBER", "AMOUNT", "DATE"]
config.output.format = "csv"

# Use custom configuration
processor = DocumentProcessor()
processor.config = config

result = processor.process_document("invoice.jpg")
```

### Working with Results

```python
# Access OCR results
ocr_result = result.ocr_result
print(f"Extracted text: {ocr_result.text}")
print(f"OCR confidence: {ocr_result.confidence}")
print(f"Word confidences: {ocr_result.word_confidences}")

# Access NER results
ner_result = result.ner_result
print(f"Document type: {ner_result.structured_data['document_type']}")

# Get entities by type
entities_by_type = ner_result.structured_data['entities_by_type']
for entity_type, entities in entities_by_type.items():
    print(f"{entity_type}: {entities}")

# Get key information
key_info = ner_result.structured_data['key_information']
for key, value in key_info.items():
    if isinstance(value, dict) and 'value' in value:
        print(f"{key}: {value['value']} (confidence: {value['confidence']:.2f})")
    else:
        print(f"{key}: {value}")

# Export results
json_output = ner_result.structured_data
csv_output = processor.ner_processor.export_to_csv(ner_result)
```

## Configuration Examples

### config.yaml

```yaml
# OCR Configuration
ocr:
  engine: "tesseract"  # Options: "tesseract", "easyocr"
  tesseract_config: "--oem 3 --psm 6"
  language: "eng"  # Language code for OCR
  confidence_threshold: 0.6  # Minimum confidence for OCR results

# Image Preprocessing
preprocessing:
  resize_factor: 2.0  # Resize image for better OCR
  apply_denoising: true  # Apply noise reduction
  enhance_contrast: true  # Enhance image contrast
  threshold_method: "adaptive"  # Options: "adaptive", "otsu", "simple"

# Named Entity Recognition
ner:
  model_name: "en_core_web_sm"  # spaCy model name
  custom_entities:
    - "INVOICE_NUMBER"
    - "AMOUNT"
    - "DATE"
    - "VENDOR"
    - "CUSTOMER"
    - "EMAIL"
    - "PHONE"
  confidence_threshold: 0.8

# Output Configuration
output:
  format: "json"  # Options: "json", "csv"
  structured: true
  include_confidence: true
```

## Advanced Usage

### Custom Entity Patterns

You can add custom regex patterns for entity extraction by modifying the NER module:

```python
# Add to src/document_extractor/ner/__init__.py
patterns = {
    "CUSTOM_ID": [
        r"ID[\s#:]*([A-Z0-9\-]{5,15})",
        r"Reference[\s:]*([A-Z0-9\-]{5,15})"
    ],
    "CUSTOM_AMOUNT": [
        r"Total[\s:]*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)"
    ]
}
```

### Error Handling

```python
try:
    result = processor.process_document("document.jpg")
    if not result.success:
        print(f"Processing failed: {result.error_message}")
        # Handle specific error cases
        if "No text extracted" in result.error_message:
            print("Try adjusting image preprocessing settings")
        elif "OCR" in result.error_message:
            print("Check OCR engine installation")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Performance Optimization

```python
# For batch processing, monitor performance
import time

start_time = time.time()
results = processor.process_batch("large_dataset/", "results/")

# Calculate statistics
total_time = time.time() - start_time
successful = sum(1 for r in results.values() if r.success)
avg_time = total_time / len(results) if results else 0

print(f"Processed {len(results)} files in {total_time:.2f}s")
print(f"Average time per file: {avg_time:.2f}s")
print(f"Success rate: {successful/len(results)*100:.1f}%")
```

## Output Examples

### JSON Output Structure

```json
{
  "document_type": "invoice",
  "extraction_timestamp": "2024-01-15T10:30:00",
  "entities_by_type": {
    "INVOICE_NUMBER": [
      {
        "text": "INV-2024-001",
        "confidence": 0.95,
        "position": {"start": 15, "end": 27}
      }
    ],
    "AMOUNT": [
      {
        "text": "$1,234.56",
        "confidence": 0.92,
        "position": {"start": 45, "end": 54}
      }
    ],
    "DATE": [
      {
        "text": "January 15, 2024",
        "confidence": 0.88,
        "position": {"start": 60, "end": 76}
      }
    ]
  },
  "key_information": {
    "invoice_number": {
      "value": "INV-2024-001",
      "confidence": 0.95
    },
    "total_amount": {
      "value": "$1,234.56",
      "confidence": 0.92
    },
    "invoice_date": {
      "value": "January 15, 2024",
      "confidence": 0.88
    }
  },
  "confidence_scores": {
    "overall": 0.92,
    "ocr_quality": 0.94,
    "entity_extraction": 0.90
  },
  "processing_metadata": {
    "text_length": 450,
    "total_entities": 8,
    "entity_types_found": 5
  }
}
```

### CSV Output Structure

```csv
text,label,start,end,confidence
INV-2024-001,INVOICE_NUMBER,15,27,0.95
$1234.56,AMOUNT,45,54,0.92
January 15 2024,DATE,60,76,0.88
John Smith,PERSON,80,90,0.85
john@example.com,EMAIL,95,110,0.90
```

## Troubleshooting

### Common Issues and Solutions

1. **"Tesseract not found" Error**
   ```bash
   # Install Tesseract OCR
   sudo apt-get install tesseract-ocr  # Ubuntu/Debian
   brew install tesseract              # macOS
   ```

2. **"spaCy model not found" Error**
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **Low OCR Accuracy**
   - Increase `resize_factor` in preprocessing
   - Enable `apply_denoising` and `enhance_contrast`
   - Try switching to EasyOCR engine
   - Adjust `threshold_method`

4. **Missing Entities**
   - Lower `confidence_threshold` for NER
   - Add custom regex patterns for specific entity types
   - Check that text was correctly extracted by OCR

5. **Performance Issues**
   - Process files in smaller batches
   - Reduce image size with `resize_factor`
   - Use SSD storage for better I/O performance

### Debugging

Enable verbose logging to debug issues:

```bash
python main.py process document.jpg -v -o results/
```

Or in Python:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

processor = DocumentProcessor()
result = processor.process_document("document.jpg")
```