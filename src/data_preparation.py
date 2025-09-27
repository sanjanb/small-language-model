"""
Data preparation module for document text extraction.
Handles OCR, text cleaning, and dataset creation for NER training.
"""

import os
import json
import re
import pytesseract
from PIL import Image
import pandas as pd
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import fitz  # PyMuPDF for PDF processing
from docx import Document
import easyocr


class DocumentProcessor:
    """Handles document processing, OCR, and text extraction."""
    
    def __init__(self, tesseract_path: Optional[str] = None):
        """Initialize document processor with OCR settings."""
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Initialize EasyOCR reader
        self.ocr_reader = easyocr.Reader(['en'])
        
        # Entity patterns for initial labeling
        self.entity_patterns = {
            'NAME': [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last
                r'(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+[A-Z][a-z]+ [A-Z][a-z]+',  # Title + Name
            ],
            'DATE': [
                r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b',  # DD/MM/YYYY
                r'\b\d{4}[/\-]\d{1,2}[/\-]\d{1,2}\b',    # YYYY/MM/DD
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}\b'
            ],
            'INVOICE_NO': [
                r'(?:Invoice\s+(?:No|Number|#):\s*)?([A-Z]{2,4}[-]?\d{3,6})',
                r'(?:INV[-]?\d{3,6})',
            ],
            'AMOUNT': [
                r'\$\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?',  # $1,000.00
                r'\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP)',  # 1000.00 USD
            ],
            'ADDRESS': [
                r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln).*',
            ],
            'PHONE': [
                r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
                r'\(\d{3}\)\s*\d{3}-\d{4}',
            ],
            'EMAIL': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            ]
        }
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file."""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            print(f"Error extracting text from PDF {pdf_path}: {e}")
            return ""
    
    def extract_text_from_docx(self, docx_path: str) -> str:
        """Extract text from DOCX file."""
        try:
            doc = Document(docx_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            print(f"Error extracting text from DOCX {docx_path}: {e}")
            return ""
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for better OCR results."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        return thresh
    
    def extract_text_with_tesseract(self, image_path: str) -> str:
        """Extract text using Tesseract OCR."""
        try:
            preprocessed_img = self.preprocess_image(image_path)
            
            # Configure Tesseract
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(preprocessed_img, config=custom_config)
            
            return text
        except Exception as e:
            print(f"Error with Tesseract OCR on {image_path}: {e}")
            return ""
    
    def extract_text_with_easyocr(self, image_path: str) -> str:
        """Extract text using EasyOCR."""
        try:
            results = self.ocr_reader.readtext(image_path)
            text = " ".join([result[1] for result in results])
            return text
        except Exception as e:
            print(f"Error with EasyOCR on {image_path}: {e}")
            return ""
    
    def extract_text_from_image(self, image_path: str, use_easyocr: bool = True) -> str:
        """Extract text from image using OCR."""
        if use_easyocr:
            text = self.extract_text_with_easyocr(image_path)
            if not text.strip():  # Fallback to Tesseract
                text = self.extract_text_with_tesseract(image_path)
        else:
            text = self.extract_text_with_tesseract(image_path)
            if not text.strip():  # Fallback to EasyOCR
                text = self.extract_text_with_easyocr(image_path)
        
        return text
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\,\:\;\-\$\(\)\[\]\/]', '', text)
        
        # Normalize whitespace around punctuation
        text = re.sub(r'\s*([,.;:])\s*', r'\1 ', text)
        
        return text.strip()
    
    def process_document(self, file_path: str) -> str:
        """Process any document type and extract text."""
        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.pdf':
            text = self.extract_text_from_pdf(str(file_path))
        elif file_ext == '.docx':
            text = self.extract_text_from_docx(str(file_path))
        elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            text = self.extract_text_from_image(str(file_path))
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        return self.clean_text(text)


class NERDatasetCreator:
    """Creates NER training datasets from processed documents."""
    
    def __init__(self, document_processor: DocumentProcessor):
        self.document_processor = document_processor
        self.entity_labels = ['O', 'B-NAME', 'I-NAME', 'B-DATE', 'I-DATE', 
                             'B-INVOICE_NO', 'I-INVOICE_NO', 'B-AMOUNT', 'I-AMOUNT',
                             'B-ADDRESS', 'I-ADDRESS', 'B-PHONE', 'I-PHONE',
                             'B-EMAIL', 'I-EMAIL']
    
    def auto_label_text(self, text: str) -> List[Tuple[str, str]]:
        """Automatically label text using regex patterns."""
        words = text.split()
        labels = ['O'] * len(words)
        
        # Track word positions in original text
        word_positions = []
        start = 0
        for word in words:
            pos = text.find(word, start)
            word_positions.append((pos, pos + len(word)))
            start = pos + len(word)
        
        # Apply entity patterns
        for entity_type, patterns in self.document_processor.entity_patterns.items():
            for pattern in patterns:
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                for match in matches:
                    match_start, match_end = match.span()
                    
                    # Find which words overlap with this match
                    first_word_idx = None
                    last_word_idx = None
                    
                    for i, (word_start, word_end) in enumerate(word_positions):
                        if word_start >= match_start and word_end <= match_end:
                            if first_word_idx is None:
                                first_word_idx = i
                            last_word_idx = i
                        elif word_start < match_end and word_end > match_start:
                            # Partial overlap
                            if first_word_idx is None:
                                first_word_idx = i
                            last_word_idx = i
                    
                    # Apply BIO labeling
                    if first_word_idx is not None:
                        labels[first_word_idx] = f'B-{entity_type}'
                        for i in range(first_word_idx + 1, last_word_idx + 1):
                            labels[i] = f'I-{entity_type}'
        
        return list(zip(words, labels))
    
    def create_training_example(self, text: str) -> Dict:
        """Create a training example from text."""
        labeled_tokens = self.auto_label_text(text)
        
        tokens = [token for token, _ in labeled_tokens]
        labels = [label for _, label in labeled_tokens]
        
        return {
            'tokens': tokens,
            'labels': labels,
            'text': text
        }
    
    def create_sample_dataset(self) -> List[Dict]:
        """Create sample training data for demonstration."""
        sample_texts = [
            "Invoice sent to Robert White on 15/09/2025 Invoice No: INV-1024 Amount: $1,250",
            "Bill for Sarah Johnson dated March 10, 2025. Invoice Number: BL-2045. Total: $2,300.50",
            "Payment due from Michael Brown on 01/12/2025. Reference: PAY-3067. Sum: $890.00",
            "Receipt for Emma Wilson Invoice: REC-4089 Date: 2025-04-22 Amount: $1,750.25",
            "Dr. James Smith 123 Main Street Boston MA 02101 Phone: (555) 123-4567 Email: james@email.com",
            "Ms. Lisa Anderson 456 Oak Avenue New York NY 10001 Contact: +1-555-987-6543",
            "Invoice INV-5678 issued to David Lee on February 5, 2025 for $3,400.00",
            "Bill #BIL-9012 for Jennifer Garcia dated 2025-05-15. Total amount: $567.89"
        ]
        
        dataset = []
        for text in sample_texts:
            example = self.create_training_example(text)
            dataset.append(example)
        
        return dataset
    
    def process_documents_folder(self, folder_path: str) -> List[Dict]:
        """Process all documents in a folder and create training dataset."""
        folder_path = Path(folder_path)
        dataset = []
        
        if not folder_path.exists():
            print(f"Folder {folder_path} does not exist. Creating sample dataset instead.")
            return self.create_sample_dataset()
        
        supported_extensions = ['.pdf', '.docx', '.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        
        for file_path in folder_path.rglob('*'):
            if file_path.suffix.lower() in supported_extensions:
                try:
                    print(f"Processing {file_path.name}...")
                    text = self.document_processor.process_document(str(file_path))
                    
                    if text.strip():  # Only process non-empty texts
                        example = self.create_training_example(text)
                        example['source_file'] = str(file_path)
                        dataset.append(example)
                        print(f"Processed {file_path.name}")
                    else:
                        print(f"No text extracted from {file_path.name}")
                        
                except Exception as e:
                    print(f"Error processing {file_path.name}: {e}")
        
        if not dataset:
            print("No documents processed. Creating sample dataset.")
            return self.create_sample_dataset()
        
        return dataset
    
    def save_dataset(self, dataset: List[Dict], output_path: str):
        """Save dataset to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"Dataset saved to {output_path}")
        print(f"Total examples: {len(dataset)}")
        
        # Print statistics
        all_labels = []
        for example in dataset:
            all_labels.extend(example['labels'])
        
        label_counts = {}
        for label in all_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print("\nLabel distribution:")
        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count}")


def main():
    """Main function to demonstrate data preparation."""
    # Initialize components
    processor = DocumentProcessor()
    dataset_creator = NERDatasetCreator(processor)
    
    # Process documents (or create sample data)
    raw_data_path = "data/raw"
    dataset = dataset_creator.process_documents_folder(raw_data_path)
    
    # Save processed dataset
    output_path = "data/processed/ner_dataset.json"
    dataset_creator.save_dataset(dataset, output_path)
    
    print(f"\nData preparation completed!")
    print(f"Processed {len(dataset)} documents")


if __name__ == "__main__":
    main()