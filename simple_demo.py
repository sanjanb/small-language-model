"""
Simplified demo of document text extraction without heavy ML dependencies.
This demonstrates the core workflow and patterns without requiring PyTorch/Transformers.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any


class SimpleDocumentProcessor:
    """Simplified document processor for demo purposes."""
    
    def __init__(self):
        """Initialize with regex patterns for entity extraction."""
        self.entity_patterns = {
            'NAME': [
                r'\b(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+([A-Z][a-z]+ [A-Z][a-z]+)\b',
                r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b',
            ],
            'DATE': [
                r'\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\b',
                r'\b(\d{4}[/\-]\d{1,2}[/\-]\d{1,2})\b',
                r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})\b'
            ],
            'INVOICE_NO': [
                r'(?:Invoice\s+(?:No|Number|#):\s*)?([A-Z]{2,4}[-]?\d{3,6})',
                r'(INV[-]?\d{3,6})',
                r'(BL[-]?\d{3,6})',
                r'(REC[-]?\d{3,6})',
            ],
            'AMOUNT': [
                r'(\$\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP))',
            ],
            'PHONE': [
                r'(\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})',
                r'(\(\d{3}\)\s*\d{3}-\d{4})',
            ],
            'EMAIL': [
                r'\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b',
            ]
        }
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text using regex patterns."""
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity_text = match.group(1) if match.groups() else match.group(0)
                    entities.append({
                        'entity': entity_type,
                        'text': entity_text.strip(),
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': self.get_confidence_score(entity_type)
                    })
        
        return entities
    
    def get_confidence_score(self, entity_type: str) -> float:
        """Get confidence score for entity type."""
        confidence_map = {
            'NAME': 0.80,
            'DATE': 0.85,
            'AMOUNT': 0.85,
            'INVOICE_NO': 0.90,
            'EMAIL': 0.95,
            'PHONE': 0.90,
            'ADDRESS': 0.75
        }
        return confidence_map.get(entity_type, 0.70)
    
    def create_structured_data(self, entities: List[Dict[str, Any]]) -> Dict[str, str]:
        """Create structured data from entities."""
        structured = {}
        
        # Group entities by type
        entity_groups = {}
        for entity in entities:
            entity_type = entity['entity']
            if entity_type not in entity_groups:
                entity_groups[entity_type] = []
            entity_groups[entity_type].append(entity)
        
        # Select best entity for each type
        for entity_type, group in entity_groups.items():
            if group:
                # Sort by confidence and length, take the best one
                best_entity = max(group, key=lambda x: (x['confidence'], len(x['text'])))
                
                # Map to structured field names
                field_mapping = {
                    'NAME': 'Name',
                    'DATE': 'Date', 
                    'AMOUNT': 'Amount',
                    'INVOICE_NO': 'InvoiceNo',
                    'EMAIL': 'Email',
                    'PHONE': 'Phone',
                    'ADDRESS': 'Address'
                }
                
                field_name = field_mapping.get(entity_type, entity_type)
                structured[field_name] = best_entity['text']
        
        return structured
    
    def process_document(self, text: str) -> Dict[str, Any]:
        """Process document text and extract information."""
        entities = self.extract_entities(text)
        structured_data = self.create_structured_data(entities)
        
        return {
            'text': text,
            'entities': entities,
            'structured_data': structured_data,
            'entity_count': len(entities),
            'entity_types': list(set(e['entity'] for e in entities))
        }


def run_demo():
    """Run the simplified document extraction demo."""
    
    print("SIMPLIFIED DOCUMENT TEXT EXTRACTION DEMO")
    print("=" * 60)
    print("This demo shows the core extraction logic using regex patterns")
    print("(without the full ML pipeline for demonstration purposes)")
    print()
    
    # Initialize processor
    processor = SimpleDocumentProcessor()
    
    # Sample documents
    sample_documents = [
        {
            "name": "Invoice Example 1",
            "text": "Invoice sent to Robert White on 15/09/2025 Invoice No: INV-1024 Amount: $1,250.00 Phone: (555) 123-4567 Email: robert.white@email.com"
        },
        {
            "name": "Invoice Example 2", 
            "text": "Bill for Dr. Sarah Johnson dated March 10, 2025. Invoice Number: BL-2045. Total: $2,300.50 Email: sarah.johnson@email.com"
        },
        {
            "name": "Receipt Example",
            "text": "Receipt for Michael Brown Invoice: REC-3089 Date: 2025-04-22 Amount: $890.75 Contact: +1-555-987-6543"
        },
        {
            "name": "Business Document",
            "text": "Ms. Emma Wilson 456 Oak Street Payment due: January 15, 2025 Reference: INV-4567 Total: $1,750.25"
        }
    ]
    
    # Process each document
    all_results = []
    total_entities = 0
    all_entity_types = set()
    
    for i, doc in enumerate(sample_documents, 1):
        print(f"\nDocument {i}: {doc['name']}")
        print("-" * 50)
        print(f"Text: {doc['text']}")
        print()
        
        # Process document
        result = processor.process_document(doc['text'])
        all_results.append(result)
        
        # Update totals
        total_entities += result['entity_count']
        all_entity_types.update(result['entity_types'])
        
        print(f"Extraction Results:")
        print(f"   Found {result['entity_count']} entities")
        print(f"   Entity types: {', '.join(result['entity_types'])}")
        
        # Show structured data if available
        if result['structured_data']:
            print(f"\nStructured Information:")
            for key, value in result['structured_data'].items():
                print(f"   {key}: {value}")
        
        # Show detailed entities
        if result['entities']:
            print(f"\nDetailed Entities:")
            for entity in result['entities']:
                print(f"   {entity['entity']}: '{entity['text']}' (confidence: {entity['confidence']*100:.0f}%)")
    
    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "demo_extraction_results.json"
    
    # Prepare output data
    output_data = {
        'demo_info': {
            'timestamp': datetime.now().isoformat(),
            'documents_processed': len(sample_documents),
            'total_entities_found': total_entities,
            'unique_entity_types': sorted(list(all_entity_types))
        },
        'results': all_results
    }
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    
    print(f"\nDemo Summary:")
    print(f"   Documents processed: {len(sample_documents)}")
    print(f"   Total entities found: {total_entities}")
    print(f"   Total structured fields: {sum(len(r['structured_data']) for r in all_results)}")
    print(f"   Unique entity types: {', '.join(sorted(all_entity_types))}")
    
    print(f"\nDemo completed successfully!")
    
    print(f"\nThis demonstrates the core extraction logic.")
    print(f"   The full system would add:")
    print(f"   - OCR for scanned documents")
    print(f"   - ML model (DistilBERT) for better accuracy")
    print(f"   - Web API for file uploads")
    print(f"   - Training pipeline for custom domains")
    
    # Simulate API functionality
    print(f"\nAPI FUNCTIONALITY SIMULATION")
    print("=" * 40)
    
    sample_text = "Invoice sent to John Doe on 01/15/2025 Invoice No: INV-1001 Amount: $1,500.00"
    
    print('API Request (POST /extract-from-text):')
    print('  {')
    print(f'  "text": "{sample_text}"')
    print('}')
    
    print(f"\nAPI Response:")
    api_result = processor.process_document(sample_text)
    
    api_response = {
        "status": "success",
        "data": {
            "original_text": sample_text,
            "entities": api_result['entities'],
            "structured_data": api_result['structured_data'],
            "processing_timestamp": datetime.now().isoformat(),
            "total_entities_found": api_result['entity_count'],
            "entity_types_found": api_result['entity_types']
        }
    }
    
    print(json.dumps(api_response, indent=2))
    
    print(f"\nTo run the full system:")
    print(f"   1. Install ML dependencies: pip install torch transformers")
    print(f"   2. Run training: python src/training_pipeline.py")
    print(f"   3. Start API: python api/app.py")
    print(f"   4. Open browser: http://localhost:8000")


if __name__ == "__main__":
    run_demo()
    """Simplified document processor for demo purposes."""
    
    def __init__(self):
        """Initialize with regex patterns for entity extraction."""
        self.entity_patterns = {
            'NAME': [
                r'\b(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+([A-Z][a-z]+ [A-Z][a-z]+)\b',
                r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b',
            ],
            'DATE': [
                r'\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\b',
                r'\b(\d{4}[/\-]\d{1,2}[/\-]\d{1,2})\b',
                r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})\b'
            ],
            'INVOICE_NO': [
                r'(?:Invoice\s+(?:No|Number|#):\s*)?([A-Z]{2,4}[-]?\d{3,6})',
                r'(INV[-]?\d{3,6})',
                r'(BL[-]?\d{3,6})',
                r'(REC[-]?\d{3,6})',
            ],
            'AMOUNT': [
                r'(\$\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP))',
            ],
            'PHONE': [
                r'(\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})',
                r'(\(\d{3}\)\s*\d{3}-\d{4})',
            ],
            'EMAIL': [
                r'\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b',
            ]
        }
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text using regex patterns."""
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity_text = match.group(1) if match.groups() else match.group(0)
                    
                    # Calculate position
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    # Assign confidence based on pattern strength
                    confidence = self._calculate_confidence(entity_type, entity_text, pattern)
                    
                    entity = {
                        'entity': entity_type,
                        'text': entity_text.strip(),
                        'start': start_pos,
                        'end': end_pos,
                        'confidence': confidence
                    }
                    
                    # Avoid duplicates
                    if not self._is_duplicate(entity, entities):
                        entities.append(entity)
        
        return entities
    
    def _calculate_confidence(self, entity_type: str, text: str, pattern: str) -> float:
        """Calculate confidence score for extracted entity."""
        base_confidence = 0.8
        
        # Boost confidence for specific patterns
        if entity_type == 'EMAIL' and '@' in text:
            base_confidence = 0.95
        elif entity_type == 'PHONE' and len(re.sub(r'[^\d]', '', text)) >= 10:
            base_confidence = 0.90
        elif entity_type == 'AMOUNT' and '$' in text:
            base_confidence = 0.85
        elif entity_type == 'DATE':
            base_confidence = 0.85
        elif entity_type == 'INVOICE_NO' and any(prefix in text.upper() for prefix in ['INV', 'BL', 'REC']):
            base_confidence = 0.90
        
        return min(base_confidence, 0.99)
    
    def _is_duplicate(self, new_entity: Dict, existing_entities: List[Dict]) -> bool:
        """Check if entity is duplicate."""
        for existing in existing_entities:
            if (existing['entity'] == new_entity['entity'] and 
                existing['text'].lower() == new_entity['text'].lower()):
                return True
        return False
    
    def postprocess_entities(self, entities: List[Dict], text: str) -> Dict[str, str]:
        """Convert entities to structured data format."""
        structured_data = {}
        
        # Group entities by type and pick the best one
        entity_groups = {}
        for entity in entities:
            entity_type = entity['entity']
            if entity_type not in entity_groups:
                entity_groups[entity_type] = []
            entity_groups[entity_type].append(entity)
        
        # Select best entity for each type
        for entity_type, group in entity_groups.items():
            best_entity = max(group, key=lambda x: x['confidence'])
            
            # Format the value
            formatted_value = self._format_entity_value(best_entity['text'], entity_type)
            
            # Map to human-readable keys
            readable_key = {
                'NAME': 'Name',
                'DATE': 'Date', 
                'INVOICE_NO': 'InvoiceNo',
                'AMOUNT': 'Amount',
                'PHONE': 'Phone',
                'EMAIL': 'Email'
            }.get(entity_type, entity_type)
            
            structured_data[readable_key] = formatted_value
        
        return structured_data
    
    def _format_entity_value(self, text: str, entity_type: str) -> str:
        """Format entity value based on type."""
        text = text.strip()
        
        if entity_type == 'NAME':
            return ' '.join(word.capitalize() for word in text.split())
        elif entity_type == 'PHONE':
            digits = re.sub(r'[^\d]', '', text)
            if len(digits) == 10:
                return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
            elif len(digits) == 11 and digits[0] == '1':
                return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
        elif entity_type == 'AMOUNT':
            # Ensure proper formatting
            if not text.startswith('$'):
                return f"${text}"
        
        return text
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """Process text and return extraction results."""
        # Extract entities
        entities = self.extract_entities(text)
        
        # Create structured data
        structured_data = self.postprocess_entities(entities, text)
        
        # Return complete result
        return {
            'original_text': text,
            'entities': entities,
            'structured_data': structured_data,
            'processing_timestamp': datetime.now().isoformat(),
            'total_entities_found': len(entities),
            'entity_types_found': list(set(e['entity'] for e in entities))
        }


def run_demo():
    """Run the document extraction demo."""
    print("SIMPLIFIED DOCUMENT TEXT EXTRACTION DEMO")
    print("=" * 60)
    print("This demo shows the core extraction logic using regex patterns")
    print("(without the full ML pipeline for demonstration purposes)")
    print()
    
    # Initialize processor
    processor = SimpleDocumentProcessor()
    
    # Sample documents
    sample_docs = [
        {
            "name": "Invoice Example 1",
            "text": "Invoice sent to Robert White on 15/09/2025 Invoice No: INV-1024 Amount: $1,250.00 Phone: (555) 123-4567"
        },
        {
            "name": "Invoice Example 2", 
            "text": "Bill for Dr. Sarah Johnson dated March 10, 2025. Invoice Number: BL-2045. Total: $2,300.50 Email: sarah.johnson@email.com"
        },
        {
            "name": "Receipt Example",
            "text": "Receipt for Michael Brown Invoice: REC-3089 Date: 2025-04-22 Amount: $890.75 Contact: +1-555-987-6543"
        },
        {
            "name": "Business Document",
            "text": "Ms. Emma Wilson 456 Oak Street Payment due: January 15, 2025 Reference: INV-4567 Total: $1,750.25"
        }
    ]
    
    results = []
    
    for i, doc in enumerate(sample_docs, 1):
        print(f"\nDocument {i}: {doc['name']}")
        print("-" * 50)
        print(f"Text: {doc['text']}")
        
        # Process the document
        result = processor.process_text(doc['text'])
        results.append({
            'document_name': doc['name'],
            **result
        })
        
        # Display results
        print(f"\nExtraction Results:")
        print(f"   Found {result['total_entities_found']} entities")
        print(f"   Entity types: {', '.join(result['entity_types_found'])}")
        
        # Show structured data
        if result['structured_data']:
            print(f"\nStructured Information:")
            for key, value in result['structured_data'].items():
                print(f"   {key}: {value}")
        
        # Show detailed entities
        if result['entities']:
            print(f"\nDetailed Entities:")
            for entity in result['entities']:
                confidence_pct = int(entity['confidence'] * 100)
                print(f"   {entity['entity']}: '{entity['text']}' (confidence: {confidence_pct}%)")
    
    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "demo_extraction_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Summary statistics
    total_entities = sum(len(r['entities']) for r in results)
    total_structured_fields = sum(len(r['structured_data']) for r in results)
    unique_entity_types = set()
    for r in results:
        unique_entity_types.update(r['entity_types_found'])
    
    print(f"\nDemo Summary:")
    print(f"   Documents processed: {len(results)}")
    print(f"   Total entities found: {total_entities}")
    print(f"   Total structured fields: {total_structured_fields}")
    print(f"   Unique entity types: {', '.join(sorted(unique_entity_types))}")
    
    print(f"\nDemo completed successfully!")
    print(f"\nThis demonstrates the core extraction logic.")
    print(f"   The full system would add:")
    print(f"   - OCR for scanned documents")
    print(f"   - ML model (DistilBERT) for better accuracy")
    print(f"   - Web API for file uploads")
    print(f"   - Training pipeline for custom domains")
    
    return results


def show_api_simulation():
    """Simulate the API functionality."""
    print(f"\n🌐 API FUNCTIONALITY SIMULATION")
    print("=" * 40)
    
    processor = SimpleDocumentProcessor()
    
    # Simulate API request
    sample_request = {
        "text": "Invoice sent to John Doe on 01/15/2025 Invoice No: INV-1001 Amount: $1,500.00"
    }
    
    print(f"API Request (POST /extract-from-text):")
    print(f"  {json.dumps(sample_request, indent=2)}")
    
    # Process
    result = processor.process_text(sample_request["text"])
    
    # Simulate API response
    api_response = {
        "status": "success",
        "data": result
    }
    
    print(f"\nAPI Response:")
    print(f"  {json.dumps(api_response, indent=2)}")


if __name__ == "__main__":
    # Run the main demo
    results = run_demo()
    
    # Show API simulation
    show_api_simulation()
    
    print(f"\nTo run the full system:")
    print(f"   1. Install ML dependencies: pip install torch transformers")
    print(f"   2. Run training: python src/training_pipeline.py")
    print(f"   3. Start API: python api/app.py")
    print(f"   4. Open browser: http://localhost:8000")