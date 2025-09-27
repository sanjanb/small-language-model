"""
Simple demo script for document text extraction.
Demonstrates the complete workflow from training to inference.
"""

import os
import sys
from pathlib import Path
import json

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from src.data_preparation import DocumentProcessor, NERDatasetCreator
from src.training_pipeline import TrainingPipeline, create_custom_config
from src.inference import DocumentInference


def run_quick_demo():
    """Run a quick demonstration of the text extraction system."""
    print("🎯 DOCUMENT TEXT EXTRACTION - QUICK DEMO")
    print("=" * 60)
    
    # Sample documents for demonstration
    demo_texts = [
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
            "text": "Receipt for Michael Brown 456 Oak Street Boston MA 02101 Invoice: REC-3089 Date: 2025-04-22 Amount: $890.75"
        }
    ]
    
    print("\n📄 Sample Documents:")
    for i, doc in enumerate(demo_texts, 1):
        print(f"{i}. {doc['name']}: {doc['text'][:60]}...")
    
    # Check if model exists
    model_path = "models/document_ner_model"
    if not Path(model_path).exists():
        print(f"\n🔧 Model not found at {model_path}")
        print("Training a new model first...")
        
        # Train model
        config = create_custom_config()
        config.num_epochs = 2  # Quick training for demo
        config.batch_size = 8
        
        pipeline = TrainingPipeline(config)
        model_path = pipeline.run_complete_pipeline()
        
        print(f"✅ Model trained and saved to {model_path}")
    
    # Load inference pipeline
    print(f"\n🔍 Loading inference pipeline from {model_path}")
    try:
        inference = DocumentInference(model_path)
        print("✅ Inference pipeline loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load inference pipeline: {e}")
        return
    
    # Process demo texts
    print(f"\n🚀 Processing {len(demo_texts)} demo documents...")
    results = []
    
    for i, doc in enumerate(demo_texts, 1):
        print(f"\n📋 Processing Document {i}: {doc['name']}")
        print("-" * 50)
        print(f"Text: {doc['text']}")
        
        # Extract information
        result = inference.process_text_directly(doc['text'])
        results.append({
            'document_name': doc['name'],
            'original_text': doc['text'],
            'result': result
        })
        
        # Display results
        if 'error' not in result:
            structured_data = result.get('structured_data', {})
            entities = result.get('entities', [])
            
            print(f"\n✅ Extraction Results:")
            if structured_data:
                print("📊 Structured Data:")
                for key, value in structured_data.items():
                    print(f"   {key}: {value}")
            else:
                print("   No structured data extracted")
            
            if entities:
                print(f"🏷️  Found {len(entities)} entities:")
                for entity in entities:
                    confidence = int(entity['confidence'] * 100)
                    print(f"   {entity['entity']}: '{entity['text']}' ({confidence}%)")
        else:
            print(f"❌ Error: {result['error']}")
    
    # Save results
    output_path = "results/demo_results.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Demo results saved to: {output_path}")
    
    # Summary
    successful_extractions = sum(1 for r in results if 'error' not in r['result'])
    total_entities = sum(len(r['result'].get('entities', [])) for r in results if 'error' not in r['result'])
    total_structured_fields = sum(len(r['result'].get('structured_data', {})) for r in results if 'error' not in r['result'])
    
    print(f"\n📈 Demo Summary:")
    print(f"   Successfully processed: {successful_extractions}/{len(demo_texts)} documents")
    print(f"   Total entities found: {total_entities}")
    print(f"   Total structured fields: {total_structured_fields}")
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"💡 You can now:")
    print(f"   - Run the web API: python api/app.py")
    print(f"   - Process your own documents using inference.py")
    print(f"   - Retrain with your data using training_pipeline.py")


def train_model_only():
    """Train the model without running inference demo."""
    print("🔧 TRAINING MODEL ONLY")
    print("=" * 40)
    
    config = create_custom_config()
    pipeline = TrainingPipeline(config)
    
    model_path = pipeline.run_complete_pipeline()
    
    print(f"✅ Model training completed!")
    print(f"📂 Model saved to: {model_path}")


def test_specific_text():
    """Test extraction on user-provided text."""
    print("✏️ CUSTOM TEXT EXTRACTION")
    print("=" * 40)
    
    # Check if model exists
    model_path = "models/document_ner_model"
    if not Path(model_path).exists():
        print("❌ No trained model found. Please run training first.")
        return
    
    # Get text from user
    print("Enter text to extract information from:")
    print("(Example: Invoice sent to John Doe on 01/15/2025 Invoice No: INV-1001 Amount: $1,500.00)")
    text = input("Text: ").strip()
    
    if not text:
        print("❌ No text provided.")
        return
    
    # Load inference and process
    try:
        inference = DocumentInference(model_path)
        result = inference.process_text_directly(text)
        
        print(f"\n📊 Extraction Results:")
        if 'error' not in result:
            structured_data = result.get('structured_data', {})
            if structured_data:
                print("Structured Information:")
                for key, value in structured_data.items():
                    print(f"  {key}: {value}")
            else:
                print("No structured information found.")
            
            entities = result.get('entities', [])
            if entities:
                print(f"\nEntities Found ({len(entities)}):")
                for entity in entities:
                    confidence = int(entity['confidence'] * 100)
                    print(f"  {entity['entity']}: '{entity['text']}' ({confidence}%)")
        else:
            print(f"❌ Error: {result['error']}")
    
    except Exception as e:
        print(f"❌ Failed to process text: {e}")


def main():
    """Main demo function with options."""
    print("🎯 DOCUMENT TEXT EXTRACTION SYSTEM")
    print("=" * 50)
    print("Choose an option:")
    print("1. Run complete demo (train + inference)")
    print("2. Train model only")
    print("3. Test specific text (requires trained model)")
    print("4. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            run_quick_demo()
            break
        elif choice == '2':
            train_model_only()
            break
        elif choice == '3':
            test_specific_text()
            break
        elif choice == '4':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()