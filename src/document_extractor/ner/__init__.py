"""Named Entity Recognition module for extracting structured information."""
import logging
from typing import Dict, List, Tuple, Optional, Any
import re
from dataclasses import dataclass, field
from datetime import datetime
import spacy
from spacy.tokens import Doc
import pandas as pd

from ..config import NERConfig

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents an extracted entity."""
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0


@dataclass
class NERResult:
    """Result of NER processing."""
    entities: List[Entity] = field(default_factory=list)
    structured_data: Dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    confidence_score: float = 0.0


class DocumentNER:
    """Named Entity Recognition for document processing."""
    
    def __init__(self, config: NERConfig):
        self.config = config
        self.nlp = self._load_model()
        self.custom_patterns = self._create_custom_patterns()
    
    def _load_model(self):
        """Load spaCy model."""
        try:
            nlp = spacy.load(self.config.model_name)
            logger.info(f"Loaded spaCy model: {self.config.model_name}")
            return nlp
        except OSError:
            logger.warning(f"Model {self.config.model_name} not found. Using blank English model.")
            # Try to use a basic English model
            try:
                nlp = spacy.load("en_core_web_sm")
                return nlp
            except OSError:
                # Create blank model as fallback
                logger.warning("Creating blank English model as fallback")
                nlp = spacy.blank("en")
                return nlp
    
    def _create_custom_patterns(self) -> Dict[str, List[str]]:
        """Create custom regex patterns for document-specific entities."""
        patterns = {
            "INVOICE_NUMBER": [
                r"(?i)(?:invoice|inv|bill)[\s#:]*([A-Z0-9\-]{3,20})",
                r"(?i)(?:number|no|#)[\s:]*([A-Z0-9\-]{3,20})"
            ],
            "AMOUNT": [
                r"\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",
                r"(?i)(?:total|amount|sum)[\s:]*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)"
            ],
            "DATE": [
                r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})",
                r"(\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})",
                r"(?i)(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s,]*\d{1,2}[,\s]*\d{4}"
            ],
            "EMAIL": [
                r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"
            ],
            "PHONE": [
                r"(\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4})"
            ]
        }
        return patterns
    
    def process_text(self, text: str) -> NERResult:
        """Process text and extract named entities."""
        try:
            # Process with spaCy
            doc = self.nlp(text)
            entities = []
            
            # Extract spaCy entities
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "DATE", "MONEY", "CARDINAL"]:
                    entities.append(Entity(
                        text=ent.text,
                        label=ent.label_,
                        start=ent.start_char,
                        end=ent.end_char,
                        confidence=1.0  # spaCy doesn't provide confidence scores by default
                    ))
            
            # Extract custom entities using regex
            custom_entities = self._extract_custom_entities(text)
            entities.extend(custom_entities)
            
            # Structure the data
            structured_data = self._structure_entities(entities, text)
            
            # Calculate overall confidence
            avg_confidence = sum(e.confidence for e in entities) / len(entities) if entities else 0.0
            
            result = NERResult(
                entities=entities,
                structured_data=structured_data,
                raw_text=text,
                confidence_score=avg_confidence
            )
            
            logger.info(f"Extracted {len(entities)} entities with avg confidence {avg_confidence:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"NER processing failed: {e}")
            return NERResult(raw_text=text)
    
    def _extract_custom_entities(self, text: str) -> List[Entity]:
        """Extract entities using custom regex patterns."""
        entities = []
        
        for entity_type, patterns in self.custom_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    entity_text = match.group(1) if match.groups() else match.group(0)
                    entities.append(Entity(
                        text=entity_text.strip(),
                        label=entity_type,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.9  # High confidence for regex matches
                    ))
        
        return entities
    
    def _structure_entities(self, entities: List[Entity], text: str) -> Dict[str, Any]:
        """Structure entities into a meaningful format."""
        structured = {
            "document_type": self._infer_document_type(text),
            "entities_by_type": {},
            "key_information": {},
            "metadata": {
                "extraction_timestamp": datetime.now().isoformat(),
                "total_entities": len(entities)
            }
        }
        
        # Group entities by type
        for entity in entities:
            if entity.label not in structured["entities_by_type"]:
                structured["entities_by_type"][entity.label] = []
            structured["entities_by_type"][entity.label].append({
                "text": entity.text,
                "confidence": entity.confidence,
                "position": {"start": entity.start, "end": entity.end}
            })
        
        # Extract key information based on document type
        structured["key_information"] = self._extract_key_information(entities, text)
        
        return structured
    
    def _infer_document_type(self, text: str) -> str:
        """Infer document type from text content."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["invoice", "bill", "receipt"]):
            return "invoice"
        elif any(word in text_lower for word in ["form", "application", "request"]):
            return "form"
        elif any(word in text_lower for word in ["report", "analysis", "summary"]):
            return "report"
        elif any(word in text_lower for word in ["contract", "agreement", "terms"]):
            return "contract"
        else:
            return "document"
    
    def _extract_key_information(self, entities: List[Entity], text: str) -> Dict[str, Any]:
        """Extract key information based on entity types."""
        key_info = {}
        
        # Find highest confidence entities for key fields
        for entity in entities:
            if entity.label == "INVOICE_NUMBER":
                if "invoice_number" not in key_info or entity.confidence > key_info["invoice_number"]["confidence"]:
                    key_info["invoice_number"] = {
                        "value": entity.text,
                        "confidence": entity.confidence
                    }
            elif entity.label == "AMOUNT":
                if "total_amount" not in key_info or entity.confidence > key_info["total_amount"]["confidence"]:
                    key_info["total_amount"] = {
                        "value": entity.text,
                        "confidence": entity.confidence
                    }
            elif entity.label in ["DATE", "DATE"]:
                if "date" not in key_info or entity.confidence > key_info["date"]["confidence"]:
                    key_info["date"] = {
                        "value": entity.text,
                        "confidence": entity.confidence
                    }
            elif entity.label in ["PERSON", "ORG"]:
                if "vendor_customer" not in key_info:
                    key_info["vendor_customer"] = []
                key_info["vendor_customer"].append({
                    "name": entity.text,
                    "type": entity.label,
                    "confidence": entity.confidence
                })
        
        return key_info
    
    def export_to_json(self, result: NERResult) -> str:
        """Export NER result to JSON string."""
        import json
        return json.dumps(result.structured_data, indent=2, ensure_ascii=False)
    
    def export_to_csv(self, result: NERResult) -> str:
        """Export entities to CSV format."""
        if not result.entities:
            return "text,label,start,end,confidence\n"
        
        rows = []
        for entity in result.entities:
            rows.append([
                entity.text.replace('"', '""'),  # Escape quotes
                entity.label,
                entity.start,
                entity.end,
                entity.confidence
            ])
        
        df = pd.DataFrame(rows, columns=["text", "label", "start", "end", "confidence"])
        return df.to_csv(index=False)