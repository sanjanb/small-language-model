"""
Inference pipeline for document text extraction.
Processes new documents and extracts structured information using trained SLM.
"""

import json
import torch
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np

from src.data_preparation import DocumentProcessor
from src.model import DocumentNERModel, NERTrainer, ModelConfig


class DocumentInference:
    """Inference pipeline for extracting structured data from documents."""
    
    def __init__(self, model_path: str):
        """Initialize inference pipeline with trained model."""
        self.model_path = model_path
        self.config = self._load_config()
        self.model = None
        self.trainer = None
        self.document_processor = DocumentProcessor()
        
        # Load the trained model
        self._load_model()
        
        # Post-processing patterns for field validation and formatting
        self.postprocess_patterns = {
            'DATE': [
                r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b',
                r'\b\d{4}[/\-]\d{1,2}[/\-]\d{1,2}\b',
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}\b'
            ],
            'AMOUNT': [
                r'\$\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?',
                r'\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP)'
            ],
            'PHONE': [
                r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
                r'\(\d{3}\)\s*\d{3}-\d{4}'
            ],
            'EMAIL': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ]
        }
    
    def _load_config(self) -> ModelConfig:
        """Load training configuration."""
        config_path = Path(self.model_path) / "training_config.json"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
                config = ModelConfig(**config_dict)
        else:
            print("⚠️ No training config found. Using default configuration.")
            config = ModelConfig()
        
        return config
    
    def _load_model(self):
        """Load the trained model and tokenizer."""
        try:
            # Create model and trainer
            self.model = DocumentNERModel(self.config)
            self.trainer = NERTrainer(self.model, self.config)
            
            # Load the trained weights
            self.trainer.load_model(self.model_path)
            
            print(f"✅ Model loaded successfully from {self.model_path}")
            
        except Exception as e:
            raise Exception(f"Failed to load model from {self.model_path}: {e}")
    
    def predict_entities(self, text: str) -> List[Dict[str, Any]]:
        """Predict entities from text using the trained model."""
        # Tokenize the text
        tokens = text.split()
        
        # Prepare input for the model
        inputs = self.trainer.tokenizer(
            tokens,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.trainer.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            predictions, probabilities = self.model.predict(
                inputs['input_ids'], 
                inputs['attention_mask']
            )
        
        # Convert predictions to labels
        word_ids = inputs['input_ids'][0].cpu().numpy()
        pred_labels = predictions[0].cpu().numpy()
        probs = probabilities[0].cpu().numpy()
        
        # Align predictions with original tokens
        word_ids_list = self.trainer.tokenizer.convert_ids_to_tokens(word_ids)
        
        # Extract entities
        entities = self._extract_entities_from_predictions(
            tokens, pred_labels, probs, word_ids_list
        )
        
        return entities
    
    def _extract_entities_from_predictions(self, tokens: List[str], 
                                         pred_labels: np.ndarray,
                                         probs: np.ndarray,
                                         word_ids_list: List[str]) -> List[Dict[str, Any]]:
        """Extract entities from model predictions."""
        entities = []
        current_entity = None
        
        # Map tokenizer output back to original tokens
        token_idx = 0
        
        for i, (token_id, label_id) in enumerate(zip(word_ids_list, pred_labels)):
            if token_id in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            
            label = self.config.id2label.get(label_id, 'O')
            confidence = float(np.max(probs[i]))
            
            if label.startswith('B-'):
                # Start of new entity
                if current_entity:
                    entities.append(current_entity)
                
                entity_type = label[2:]  # Remove 'B-' prefix
                current_entity = {
                    'entity': entity_type,
                    'text': token_id if not token_id.startswith('##') else token_id[2:],
                    'start': token_idx,
                    'end': token_idx + 1,
                    'confidence': confidence
                }
            
            elif label.startswith('I-') and current_entity:
                # Continue current entity
                entity_type = label[2:]  # Remove 'I-' prefix
                if current_entity['entity'] == entity_type:
                    if token_id.startswith('##'):
                        current_entity['text'] += token_id[2:]
                    else:
                        current_entity['text'] += ' ' + token_id
                    current_entity['end'] = token_idx + 1
                    current_entity['confidence'] = min(current_entity['confidence'], confidence)
            
            else:
                # 'O' label or end of entity
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
            
            if not token_id.startswith('##'):
                token_idx += 1
        
        # Add the last entity if it exists
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def postprocess_entities(self, entities: List[Dict[str, Any]], 
                           original_text: str) -> Dict[str, Any]:
        """Post-process and structure extracted entities."""
        structured_data = {}
        
        for entity in entities:
            entity_type = entity['entity']
            entity_text = entity['text']
            confidence = entity['confidence']
            
            # Apply post-processing patterns for validation
            if entity_type in self.postprocess_patterns:
                is_valid = self._validate_entity(entity_text, entity_type)
                if not is_valid:
                    continue
            
            # Format the entity value
            formatted_value = self._format_entity_value(entity_text, entity_type)
            
            # Store the best entity for each type (highest confidence)
            if entity_type not in structured_data or confidence > structured_data[entity_type]['confidence']:
                structured_data[entity_type] = {
                    'value': formatted_value,
                    'confidence': confidence,
                    'original_text': entity_text
                }
        
        # Convert to final format
        final_data = {}
        entity_mapping = {
            'NAME': 'Name',
            'DATE': 'Date',
            'INVOICE_NO': 'InvoiceNo',
            'AMOUNT': 'Amount',
            'ADDRESS': 'Address',
            'PHONE': 'Phone',
            'EMAIL': 'Email'
        }
        
        for entity_type, entity_data in structured_data.items():
            human_readable_key = entity_mapping.get(entity_type, entity_type)
            final_data[human_readable_key] = entity_data['value']
        
        return final_data
    
    def _validate_entity(self, text: str, entity_type: str) -> bool:
        """Validate entity using regex patterns."""
        patterns = self.postprocess_patterns.get(entity_type, [])
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _format_entity_value(self, text: str, entity_type: str) -> str:
        """Format entity value based on its type."""
        text = text.strip()
        
        if entity_type == 'DATE':
            # Normalize date format
            date_patterns = [
                (r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})', r'\1/\2/\3'),
                (r'(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})', r'\3/\2/\1')
            ]
            
            for pattern, replacement in date_patterns:
                match = re.search(pattern, text)
                if match:
                    return re.sub(pattern, replacement, text)
        
        elif entity_type == 'AMOUNT':
            # Normalize amount format
            amount_match = re.search(r'[\$\d,\.]+', text)
            if amount_match:
                return amount_match.group()
        
        elif entity_type == 'PHONE':
            # Normalize phone format
            digits = re.sub(r'[^\d]', '', text)
            if len(digits) == 10:
                return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
            elif len(digits) == 11 and digits[0] == '1':
                return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
        
        elif entity_type == 'NAME':
            # Capitalize name properly
            return ' '.join(word.capitalize() for word in text.split())
        
        return text
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process a document and extract structured information."""
        print(f"Processing document: {file_path}")
        
        try:
            # Extract text from document
            text = self.document_processor.process_document(file_path)
            
            if not text.strip():
                return {
                    'error': 'No text could be extracted from the document',
                    'file_path': file_path
                }
            
            # Predict entities
            entities = self.predict_entities(text)
            
            # Post-process and structure data
            structured_data = self.postprocess_entities(entities, text)
            
            # Create result
            result = {
                'file_path': file_path,
                'extracted_text': text[:500] + '...' if len(text) > 500 else text,
                'entities': entities,
                'structured_data': structured_data,
                'processing_timestamp': datetime.now().isoformat(),
                'model_path': self.model_path
            }
            
            print(f"✅ Successfully processed {file_path}")
            print(f"   Found {len(entities)} entities")
            print(f"   Structured fields: {list(structured_data.keys())}")
            
            return result
            
        except Exception as e:
            error_result = {
                'error': str(e),
                'file_path': file_path,
                'processing_timestamp': datetime.now().isoformat()
            }
            print(f"❌ Error processing {file_path}: {e}")
            return error_result
    
    def process_text_directly(self, text: str) -> Dict[str, Any]:
        """Process text directly without file operations."""
        print("Processing text directly...")
        
        try:
            # Clean the text
            cleaned_text = self.document_processor.clean_text(text)
            
            # Predict entities
            entities = self.predict_entities(cleaned_text)
            
            # Post-process and structure data
            structured_data = self.postprocess_entities(entities, cleaned_text)
            
            # Create result
            result = {
                'original_text': text,
                'cleaned_text': cleaned_text,
                'entities': entities,
                'structured_data': structured_data,
                'processing_timestamp': datetime.now().isoformat(),
                'model_path': self.model_path
            }
            
            print(f"✅ Successfully processed text")
            print(f"   Found {len(entities)} entities")
            print(f"   Structured fields: {list(structured_data.keys())}")
            
            return result
            
        except Exception as e:
            error_result = {
                'error': str(e),
                'original_text': text,
                'processing_timestamp': datetime.now().isoformat()
            }
            print(f"❌ Error processing text: {e}")
            return error_result
    
    def batch_process_documents(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Process multiple documents in batch."""
        print(f"Processing {len(file_paths)} documents...")
        
        results = []
        for i, file_path in enumerate(file_paths):
            print(f"\nProcessing {i+1}/{len(file_paths)}: {Path(file_path).name}")
            result = self.process_document(file_path)
            results.append(result)
        
        print(f"\n✅ Batch processing completed!")
        print(f"   Successfully processed: {sum(1 for r in results if 'error' not in r)}")
        print(f"   Errors: {sum(1 for r in results if 'error' in r)}")
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """Save processing results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Results saved to: {output_path}")


def create_demo_inference(model_path: str = "models/document_ner_model") -> DocumentInference:
    """Create inference pipeline for demonstration."""
    try:
        inference = DocumentInference(model_path)
        return inference
    except Exception as e:
        print(f"❌ Failed to create inference pipeline: {e}")
        print("💡 Make sure you have trained the model first by running training_pipeline.py")
        raise


def demo_text_extraction():
    """Demonstrate text extraction with sample texts."""
    print("🎯 DOCUMENT TEXT EXTRACTION - INFERENCE DEMO")
    print("=" * 60)
    
    # Sample texts for demonstration
    sample_texts = [
        "Invoice sent to Robert White on 15/09/2025 Invoice No: INV-1024 Amount: $1,250",
        "Bill for Dr. Sarah Johnson dated March 10, 2025. Invoice Number: BL-2045. Total: $2,300.50 Phone: (555) 123-4567",
        "Receipt for Michael Brown 456 Oak Street Boston MA Email: michael@email.com Invoice: REC-3089 Date: 2025-04-22 Amount: $890.75"
    ]
    
    # Create inference pipeline
    try:
        inference = create_demo_inference()
        
        results = []
        for i, text in enumerate(sample_texts):
            print(f"\n📄 Processing Sample Text {i+1}:")
            print("-" * 40)
            print(f"Text: {text}")
            
            result = inference.process_text_directly(text)
            results.append(result)
            
            if 'error' not in result:
                print(f"Structured Output: {json.dumps(result['structured_data'], indent=2)}")
            else:
                print(f"Error: {result['error']}")
        
        # Save results
        inference.save_results(results, "results/demo_extraction_results.json")
        
        print("\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")


def main():
    """Main function for inference demonstration."""
    demo_text_extraction()


if __name__ == "__main__":
    main()