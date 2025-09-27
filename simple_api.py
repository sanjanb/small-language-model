#!/usr/bin/env python3
"""
Simplified Document Text Extraction API
Uses regex patterns instead of ML model for demonstration
"""

import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from fastapi import FastAPI, HTTPException, File, UploadFile
    from fastapi.responses import HTMLResponse, FileResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    print("FastAPI not installed. Install with: pip install fastapi uvicorn python-multipart")
    HAS_FASTAPI = False

class SimpleDocumentProcessor:
    """Simplified document processor using regex patterns"""
    
    def __init__(self):
        # Define regex patterns for different entity types
        self.patterns = {
            'NAME': [
                r'\b(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                r'\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
                r'(?:Invoice|Bill|Receipt)\s+(?:sent\s+)?(?:to\s+|for\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            ],
            'DATE': [
                r'\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b',
                r'\b(\d{2,4}[\/\-]\d{1,2}[\/\-]\d{1,2})\b',
                r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{2,4})\b',
                r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{2,4})\b',
            ],
            'AMOUNT': [
                r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                r'(?:Amount|Total|Sum):\s*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|dollars?))',
            ],
            'INVOICE_NO': [
                r'(?:Invoice|Bill|Receipt)(?:\s+No\.?|#|Number):\s*([A-Z]{2,4}[-\s]?\d{3,6})',
                r'(?:INV|BL|REC)[-\s]?(\d{3,6})',
                r'Reference:\s*([A-Z]{2,4}[-\s]?\d{3,6})',
            ],
            'EMAIL': [
                r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b',
            ],
            'PHONE': [
                r'\b(\+?1[-.\s]?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b',
                r'\b(\([2-9]\d{2}\)\s*[2-9]\d{2}[-.\s]?\d{4})\b',
                r'\b([2-9]\d{2}[-.\s]?[2-9]\d{2}[-.\s]?\d{4})\b',
            ],
            'ADDRESS': [
                r'\b(\d+\s+[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Boulevard|Blvd|Way))\b',
            ]
        }
        
        # Confidence scores for different entity types
        self.confidence_scores = {
            'NAME': 0.80,
            'DATE': 0.85,
            'AMOUNT': 0.85,
            'INVOICE_NO': 0.90,
            'EMAIL': 0.95,
            'PHONE': 0.90,
            'ADDRESS': 0.75
        }

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text using regex patterns"""
        entities = []
        
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity = {
                        'entity': entity_type,
                        'text': match.group(1) if match.groups() else match.group(0),
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': self.confidence_scores[entity_type]
                    }
                    entities.append(entity)
        
        return entities

    def create_structured_data(self, entities: List[Dict]) -> Dict[str, str]:
        """Create structured data from extracted entities"""
        structured = {}
        
        # Get the best entity for each type
        entity_groups = {}
        for entity in entities:
            entity_type = entity['entity']
            if entity_type not in entity_groups:
                entity_groups[entity_type] = []
            entity_groups[entity_type].append(entity)
        
        # Select best entity for each type
        for entity_type, group in entity_groups.items():
            if group:
                # Sort by confidence and take the best one
                best_entity = max(group, key=lambda x: x['confidence'])
                
                # Format field names
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

    def process_text(self, text: str) -> Dict[str, Any]:
        """Process text and extract structured information"""
        entities = self.extract_entities(text)
        structured_data = self.create_structured_data(entities)
        
        # Get unique entity types
        entity_types = list(set(entity['entity'] for entity in entities))
        
        return {
            'status': 'success',
            'data': {
                'original_text': text,
                'entities': entities,
                'structured_data': structured_data,
                'processing_timestamp': datetime.now().isoformat(),
                'total_entities_found': len(entities),
                'entity_types_found': sorted(entity_types)
            }
        }

# Pydantic models for API
if HAS_FASTAPI:
    class TextRequest(BaseModel):
        text: str

def create_app():
    """Create and configure FastAPI app"""
    if not HAS_FASTAPI:
        raise ImportError("FastAPI dependencies not installed")
    
    app = FastAPI(
        title="Simple Document Text Extraction API",
        description="Extract structured information from documents using regex patterns",
        version="1.0.0"
    )
    
    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize processor
    processor = SimpleDocumentProcessor()
    
    @app.get("/", response_class=HTMLResponse)
    async def get_interface():
        """Serve the web interface"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Document Text Extraction Demo</title>
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    max-width: 1200px; 
                    margin: 0 auto; 
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: #333;
                }
                .container {
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                }
                .header {
                    text-align: center;
                    margin-bottom: 30px;
                }
                .header h1 {
                    color: #2c3e50;
                    font-size: 2.5em;
                    margin-bottom: 10px;
                }
                .header p {
                    color: #7f8c8d;
                    font-size: 1.2em;
                }
                .tabs {
                    display: flex;
                    margin-bottom: 20px;
                }
                .tab {
                    flex: 1;
                    text-align: center;
                    padding: 15px;
                    background: #ecf0f1;
                    border: none;
                    cursor: pointer;
                    font-size: 16px;
                    transition: background 0.3s;
                }
                .tab.active {
                    background: #3498db;
                    color: white;
                }
                .tab:hover {
                    background: #3498db;
                    color: white;
                }
                .tab-content {
                    display: none;
                    padding: 20px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }
                .tab-content.active {
                    display: block;
                }
                textarea {
                    width: 100%;
                    height: 150px;
                    margin-bottom: 15px;
                    padding: 10px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    font-size: 14px;
                }
                input[type="file"] {
                    margin-bottom: 15px;
                    padding: 10px;
                }
                button {
                    background: #27ae60;
                    color: white;
                    padding: 12px 25px;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 16px;
                    transition: background 0.3s;
                }
                button:hover {
                    background: #2ecc71;
                }
                .results {
                    margin-top: 20px;
                    padding: 20px;
                    background: #f8f9fa;
                    border-radius: 5px;
                    border-left: 4px solid #27ae60;
                }
                .entity {
                    background: #e8f4fd;
                    padding: 8px 12px;
                    margin: 5px;
                    border-radius: 20px;
                    display: inline-block;
                    font-size: 12px;
                    border: 1px solid #3498db;
                }
                .entity.NAME { background: #ffeb3b; border-color: #ff9800; }
                .entity.DATE { background: #4caf50; border-color: #2e7d32; color: white; }
                .entity.AMOUNT { background: #f44336; border-color: #c62828; color: white; }
                .entity.INVOICE_NO { background: #9c27b0; border-color: #6a1b9a; color: white; }
                .entity.EMAIL { background: #00bcd4; border-color: #00838f; color: white; }
                .entity.PHONE { background: #ff5722; border-color: #d84315; color: white; }
                .entity.ADDRESS { background: #795548; border-color: #5d4037; color: white; }
                .structured-data {
                    background: #e8f5e8;
                    padding: 15px;
                    border-radius: 5px;
                    margin-top: 15px;
                }
                .examples {
                    background: #fff3cd;
                    padding: 15px;
                    border-radius: 5px;
                    margin-top: 20px;
                }
                .example-btn {
                    background: #6c757d;
                    font-size: 12px;
                    padding: 5px 10px;
                    margin: 2px;
                }
                pre {
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    overflow-x: auto;
                    font-size: 12px;
                    border: 1px solid #dee2e6;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1> Document Text Extraction</h1>
                    <p>Extract structured information from documents using AI patterns</p>
                </div>
                
                <div class="tabs">
                    <button class="tab active" onclick="showTab('text')">Enter Text</button>
                    <button class="tab" onclick="showTab('file')">Upload File</button>
                    <button class="tab" onclick="showTab('api')">API Docs</button>
                </div>
                
                <div id="text-tab" class="tab-content active">
                    <h3>Enter Text to Extract:</h3>
                    <textarea id="textInput" placeholder="Paste your document text here...">Invoice sent to Robert White on 15/09/2025 Invoice No: INV-1024 Amount: $1,250.00 Phone: (555) 123-4567 Email: robert.white@email.com</textarea>
                    <button onclick="extractFromText()">Extract Information</button>
                    
                    <div class="examples">
                        <h4>Try These Examples:</h4>
                        <button class="example-btn" onclick="useExample(0)">Invoice Example</button>
                        <button class="example-btn" onclick="useExample(1)">Receipt Example</button>
                        <button class="example-btn" onclick="useExample(2)">Business Document</button>
                        <button class="example-btn" onclick="useExample(3)">Payment Notice</button>
                    </div>
                </div>
                
                <div id="file-tab" class="tab-content">
                    <h3>Upload Document:</h3>
                    <input type="file" id="fileInput" accept=".pdf,.docx,.txt,.jpg,.png,.tiff">
                    <br>
                    <button onclick="extractFromFile()">Upload & Extract</button>
                    <p><em>Note: File upload processing is simplified in this demo</em></p>
                </div>
                
                <div id="api-tab" class="tab-content">
                    <h3>API Documentation</h3>
                    <h4>Endpoints:</h4>
                    <pre><strong>POST /extract-from-text</strong>
Content-Type: application/json
{
  "text": "Invoice sent to John Doe on 01/15/2025 Invoice No: INV-1001 Amount: $1,500.00"
}</pre>
                    
                    <pre><strong>POST /extract-from-file</strong>
Content-Type: multipart/form-data
file: [uploaded file]</pre>
                    
                    <h4>Response Format:</h4>
                    <pre>{
  "status": "success",
  "data": {
    "original_text": "...",
    "entities": [...],
    "structured_data": {...},
    "processing_timestamp": "2025-09-27T...",
    "total_entities_found": 7,
    "entity_types_found": ["NAME", "DATE", "AMOUNT", "INVOICE_NO"]
  }
}</pre>
                </div>
                
                <div id="results"></div>
            </div>
            
            <script>
                const examples = [
                    "Invoice sent to Robert White on 15/09/2025 Invoice No: INV-1024 Amount: $1,250.00 Phone: (555) 123-4567 Email: robert.white@email.com",
                    "Receipt for Michael Brown Invoice: REC-3089 Date: 2025-04-22 Amount: $890.75 Contact: +1-555-987-6543",
                    "Ms. Emma Wilson 456 Oak Street Payment due: January 15, 2025 Reference: INV-4567 Total: $1,750.25",
                    "Bill for Dr. Sarah Johnson dated March 10, 2025. Invoice Number: BL-2045. Total: $2,300.50 Email: sarah.johnson@email.com"
                ];
                
                function showTab(tabName) {
                    // Hide all tabs
                    document.querySelectorAll('.tab-content').forEach(content => {
                        content.classList.remove('active');
                    });
                    document.querySelectorAll('.tab').forEach(tab => {
                        tab.classList.remove('active');
                    });
                    
                    // Show selected tab
                    document.getElementById(tabName + '-tab').classList.add('active');
                    event.target.classList.add('active');
                }
                
                function useExample(index) {
                    document.getElementById('textInput').value = examples[index];
                }
                
                async function extractFromText() {
                    const text = document.getElementById('textInput').value;
                    if (!text.trim()) {
                        alert('Please enter some text');
                        return;
                    }
                    
                    try {
                        const response = await fetch('/extract-from-text', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({ text: text })
                        });
                        
                        const result = await response.json();
                        displayResults(result);
                    } catch (error) {
                        alert('Error: ' + error.message);
                    }
                }
                
                async function extractFromFile() {
                    const fileInput = document.getElementById('fileInput');
                    if (!fileInput.files[0]) {
                        alert('Please select a file');
                        return;
                    }
                    
                    // For demo purposes, show that file upload would work
                    alert('File upload processing would happen here. For now, using sample text extraction.');
                    document.getElementById('textInput').value = examples[0];
                    showTab('text');
                    extractFromText();
                }
                
                function displayResults(result) {
                    const resultsDiv = document.getElementById('results');
                    
                    if (result.status !== 'success') {
                        resultsDiv.innerHTML = '<div class="results"><h3>❌ Error</h3><p>' + result.message + '</p></div>';
                        return;
                    }
                    
                    const data = result.data;
                    let html = '<div class="results">';
                    html += '<h3>✅ Extraction Results</h3>';
                    html += '<p><strong>Found:</strong> ' + data.total_entities_found + ' entities of ' + data.entity_types_found.length + ' types</p>';
                    
                    // Show entities
                    html += '<h4>🏷️ Detected Entities:</h4>';
                    data.entities.forEach(entity => {
                        html += '<span class="entity ' + entity.entity + '">' + entity.entity + ': ' + entity.text + ' (' + Math.round(entity.confidence * 100) + '%)</span> ';
                    });
                    
                    // Show structured data
                    if (Object.keys(data.structured_data).length > 0) {
                        html += '<div class="structured-data">';
                        html += '<h4>📊 Structured Information:</h4>';
                        html += '<ul>';
                        for (const [key, value] of Object.entries(data.structured_data)) {
                            html += '<li><strong>' + key + ':</strong> ' + value + '</li>';
                        }
                        html += '</ul>';
                        html += '</div>';
                    }
                    
                    // Show processing info
                    html += '<p><small>🕒 Processed at: ' + new Date(data.processing_timestamp).toLocaleString() + '</small></p>';
                    html += '</div>';
                    
                    resultsDiv.innerHTML = html;
                }
            </script>
        </body>
        </html>
        """

    @app.post("/extract-from-text")
    async def extract_from_text(request: TextRequest):
        """Extract entities from text"""
        try:
            result = processor.process_text(request.text)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/extract-from-file")
    async def extract_from_file(file: UploadFile = File(...)):
        """Extract entities from uploaded file"""
        try:
            # Read file content
            content = await file.read()
            
            # For demo purposes, convert to text (simplified)
            if file.filename.lower().endswith('.txt'):
                text = content.decode('utf-8')
            else:
                # For other file types, use sample text in demo
                text = "Demo processing for " + file.filename + ": Invoice sent to John Doe on 01/15/2025 Invoice No: INV-1001 Amount: $1,500.00"
            
            result = processor.process_text(text)
            return result
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}

    return app

def main():
    """Main function to run the API server"""
    if not HAS_FASTAPI:
        print("❌ FastAPI dependencies not installed.")
        print("📦 Install with: pip install fastapi uvicorn python-multipart")
        return
    
    print("🚀 Starting Simple Document Text Extraction API...")
    print("📍 Access the web interface at: http://localhost:8001")
    print("📖 API documentation at: http://localhost:8001/docs")
    print("🔧 Health check at: http://localhost:8001/health")
    print("\n⚡ Server starting...")
    
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")

if __name__ == "__main__":
    main()