"""
FastAPI web service for document text extraction.
Provides REST API endpoints for uploading and processing documents.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import tempfile
import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
import shutil

from src.inference import DocumentInference


# Initialize FastAPI app
app = FastAPI(
    title="Document Text Extraction API",
    description="Extract structured information from documents using Small Language Model (SLM)",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference pipeline
inference_pipeline: Optional[DocumentInference] = None

def get_inference_pipeline() -> DocumentInference:
    """Get or initialize the inference pipeline."""
    global inference_pipeline
    
    if inference_pipeline is None:
        model_path = "models/document_ner_model"
        
        if not Path(model_path).exists():
            raise HTTPException(
                status_code=503,
                detail="Model not found. Please train the model first by running training_pipeline.py"
            )
        
        try:
            inference_pipeline = DocumentInference(model_path)
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Failed to load model: {str(e)}"
            )
    
    return inference_pipeline


@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    try:
        get_inference_pipeline()
        print("✅ Model loaded successfully on startup")
    except Exception as e:
        print(f"⚠️ Failed to load model on startup: {e}")
        print("Model will be loaded on first request")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML interface."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Document Text Extraction</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .header {
                text-align: center;
                color: #333;
                margin-bottom: 30px;
            }
            .upload-area {
                border: 2px dashed #ccc;
                padding: 40px;
                text-align: center;
                margin: 20px 0;
                border-radius: 8px;
                background-color: #fafafa;
            }
            .upload-area:hover {
                border-color: #007bff;
                background-color: #f0f8ff;
            }
            .btn {
                background-color: #007bff;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }
            .btn:hover {
                background-color: #0056b3;
            }
            .result {
                margin-top: 20px;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 5px;
                border: 1px solid #dee2e6;
            }
            .json-output {
                background-color: #f4f4f4;
                padding: 15px;
                border-radius: 5px;
                font-family: monospace;
                white-space: pre-wrap;
                overflow-x: auto;
                max-height: 400px;
                overflow-y: auto;
            }
            .text-input {
                width: 100%;
                height: 100px;
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 5px;
                font-family: monospace;
                resize: vertical;
            }
            .tab-container {
                margin: 20px 0;
            }
            .tabs {
                display: flex;
                border-bottom: 1px solid #ccc;
            }
            .tab {
                padding: 10px 20px;
                cursor: pointer;
                border-bottom: 2px solid transparent;
                background-color: #f8f9fa;
                margin-right: 5px;
            }
            .tab.active {
                border-bottom-color: #007bff;
                background-color: white;
            }
            .tab-content {
                display: none;
                padding: 20px 0;
            }
            .tab-content.active {
                display: block;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎯 Document Text Extraction</h1>
                <p>Extract structured information from documents using AI</p>
            </div>
            
            <div class="tab-container">
                <div class="tabs">
                    <div class="tab active" onclick="showTab('file')">📄 Upload File</div>
                    <div class="tab" onclick="showTab('text')">✏️ Enter Text</div>
                </div>
                
                <div id="file-tab" class="tab-content active">
                    <form id="uploadForm" enctype="multipart/form-data">
                        <div class="upload-area">
                            <p>📁 Choose a document to extract information</p>
                            <p><small>Supported: PDF, DOCX, Images (PNG, JPG, etc.)</small></p>
                            <input type="file" id="fileInput" name="file" accept=".pdf,.docx,.png,.jpg,.jpeg,.tiff,.bmp" style="margin: 10px 0;">
                            <br>
                            <button type="submit" class="btn">🚀 Extract Information</button>
                        </div>
                    </form>
                </div>
                
                <div id="text-tab" class="tab-content">
                    <form id="textForm">
                        <p>Enter text directly for information extraction:</p>
                        <textarea id="textInput" class="text-input" placeholder="Enter document text here, e.g.:&#10;Invoice sent to John Doe on 01/15/2025&#10;Invoice No: INV-1001&#10;Amount: $1,500.00"></textarea>
                        <br><br>
                        <button type="submit" class="btn">🔍 Extract from Text</button>
                    </form>
                </div>
            </div>
            
            <div id="result" class="result" style="display: none;">
                <h3>📊 Extraction Results</h3>
                <div id="resultContent"></div>
            </div>
        </div>

        <script>
            function showTab(tabName) {
                // Hide all tab contents
                document.querySelectorAll('.tab-content').forEach(content => {
                    content.classList.remove('active');
                });
                
                // Remove active class from all tabs
                document.querySelectorAll('.tab').forEach(tab => {
                    tab.classList.remove('active');
                });
                
                // Show selected tab content
                document.getElementById(tabName + '-tab').classList.add('active');
                
                // Add active class to selected tab
                event.target.classList.add('active');
            }

            // File upload form handler
            document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const fileInput = document.getElementById('fileInput');
                if (!fileInput.files[0]) {
                    alert('Please select a file');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                
                try {
                    showResult('⏳ Processing document, please wait...');
                    
                    const response = await fetch('/extract-from-file', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    displayResult(result);
                    
                } catch (error) {
                    showResult('❌ Error: ' + error.message);
                }
            });
            
            // Text form handler
            document.getElementById('textForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const text = document.getElementById('textInput').value;
                if (!text.trim()) {
                    alert('Please enter some text');
                    return;
                }
                
                try {
                    showResult('⏳ Processing text, please wait...');
                    
                    const response = await fetch('/extract-from-text', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ text: text })
                    });
                    
                    const result = await response.json();
                    displayResult(result);
                    
                } catch (error) {
                    showResult('❌ Error: ' + error.message);
                }
            });
            
            function showResult(message) {
                const resultDiv = document.getElementById('result');
                const contentDiv = document.getElementById('resultContent');
                contentDiv.innerHTML = message;
                resultDiv.style.display = 'block';
            }
            
            function displayResult(result) {
                let html = '';
                
                if (result.error) {
                    html = `<div style="color: red;">❌ Error: ${result.error}</div>`;
                } else {
                    // Show structured data
                    if (result.structured_data && Object.keys(result.structured_data).length > 0) {
                        html += '<h4>✅ Extracted Information:</h4>';
                        html += '<table style="width: 100%; border-collapse: collapse; margin: 10px 0;">';
                        html += '<tr style="background-color: #f8f9fa;"><th style="padding: 8px; border: 1px solid #dee2e6; text-align: left;">Field</th><th style="padding: 8px; border: 1px solid #dee2e6; text-align: left;">Value</th></tr>';
                        
                        for (const [key, value] of Object.entries(result.structured_data)) {
                            html += `<tr><td style="padding: 8px; border: 1px solid #dee2e6; font-weight: bold;">${key}</td><td style="padding: 8px; border: 1px solid #dee2e6;">${value}</td></tr>`;
                        }
                        html += '</table>';
                    } else {
                        html += '<div style="color: orange;">⚠️ No structured information found in the document.</div>';
                    }
                    
                    // Show entities
                    if (result.entities && result.entities.length > 0) {
                        html += '<h4>🏷️ Detected Entities:</h4>';
                        html += '<div style="margin: 10px 0;">';
                        result.entities.forEach(entity => {
                            const confidence = Math.round(entity.confidence * 100);
                            html += `<span style="display: inline-block; margin: 2px 4px; padding: 4px 8px; background-color: #e3f2fd; border: 1px solid #2196f3; border-radius: 15px; font-size: 12px;">
                                ${entity.entity}: "${entity.text}" (${confidence}%)</span>`;
                        });
                        html += '</div>';
                    }
                    
                    // Show raw JSON
                    html += '<h4>📋 Full Response:</h4>';
                    html += `<div class="json-output">${JSON.stringify(result, null, 2)}</div>`;
                }
                
                showResult(html);
            }
        </script>
    </body>
    </html>
    """
    return html_content


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        get_inference_pipeline()
        return {"status": "healthy", "message": "Model loaded successfully"}
    except Exception as e:
        return {"status": "unhealthy", "message": str(e)}


@app.post("/extract-from-file")
async def extract_from_file(file: UploadFile = File(...)):
    """Extract structured information from an uploaded file."""
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file type
    allowed_extensions = {'.pdf', '.docx', '.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_extension}. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name
    
    try:
        # Process the document
        inference = get_inference_pipeline()
        result = inference.process_document(temp_file_path)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_file_path)
        except:
            pass


@app.post("/extract-from-text")
async def extract_from_text(request: Dict[str, str]):
    """Extract structured information from text."""
    text = request.get("text", "").strip()
    
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    try:
        # Process the text
        inference = get_inference_pipeline()
        result = inference.process_text_directly(text)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported file formats."""
    return {
        "supported_formats": [
            {"extension": ".pdf", "description": "PDF documents"},
            {"extension": ".docx", "description": "Microsoft Word documents"},
            {"extension": ".png", "description": "PNG images"},
            {"extension": ".jpg", "description": "JPEG images"},
            {"extension": ".jpeg", "description": "JPEG images"},
            {"extension": ".tiff", "description": "TIFF images"},
            {"extension": ".bmp", "description": "BMP images"}
        ],
        "entity_types": [
            "Name", "Date", "InvoiceNo", "Amount", "Address", "Phone", "Email"
        ]
    }


@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model."""
    try:
        inference = get_inference_pipeline()
        return {
            "model_path": inference.model_path,
            "model_name": inference.config.model_name,
            "max_length": inference.config.max_length,
            "entity_labels": inference.config.entity_labels,
            "num_labels": inference.config.num_labels
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {str(e)}")


def main():
    """Run the FastAPI server."""
    print("🚀 Starting Document Text Extraction API Server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📄 Web interface: http://localhost:8000")
    print("📋 API docs: http://localhost:8000/docs")
    print("❤️ Health check: http://localhost:8000/health")
    
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()