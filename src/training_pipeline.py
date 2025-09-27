"""
Complete training pipeline for document text extraction using SLM.
Handles data loading, model training, evaluation, and saving.
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report as seq_classification_report

from src.data_preparation import DocumentProcessor, NERDatasetCreator
from src.model import DocumentNERModel, NERTrainer, ModelConfig, create_model_and_trainer


class TrainingPipeline:
    """Complete training pipeline for document NER."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize training pipeline."""
        self.config = config or ModelConfig()
        self.model = None
        self.trainer = None
        self.history = {}
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories for training."""
        directories = [
            "data/raw",
            "data/processed",
            "models",
            "results/plots",
            "results/metrics"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def prepare_data(self, data_path: Optional[str] = None) -> List[Dict]:
        """Prepare training data from documents or create sample data."""
        print("=" * 60)
        print("STEP 1: DATA PREPARATION")
        print("=" * 60)
        
        # Initialize document processor and dataset creator
        processor = DocumentProcessor()
        dataset_creator = NERDatasetCreator(processor)
        
        # Process documents or create sample data
        if data_path and Path(data_path).exists():
            print(f"Processing documents from: {data_path}")
            dataset = dataset_creator.process_documents_folder(data_path)
        else:
            print("No document path provided or path doesn't exist.")
            print("Creating sample dataset for demonstration...")
            dataset = dataset_creator.create_sample_dataset()
        
        # Save processed dataset
        output_path = "data/processed/ner_dataset.json"
        dataset_creator.save_dataset(dataset, output_path)
        
        print(f"✓ Data preparation completed!")
        print(f"✓ Dataset saved to: {output_path}")
        print(f"✓ Total examples: {len(dataset)}")
        
        return dataset
    
    def initialize_model(self):
        """Initialize model and trainer."""
        print("\n" + "=" * 60)
        print("STEP 2: MODEL INITIALIZATION")
        print("=" * 60)
        
        self.model, self.trainer = create_model_and_trainer(self.config)
        
        print(f"✓ Model initialized: {self.config.model_name}")
        print(f"✓ Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"✓ Device: {self.trainer.device}")
        print(f"✓ Number of entity labels: {self.config.num_labels}")
        
        return self.model, self.trainer
    
    def train_model(self, dataset: List[Dict]) -> Dict[str, List[float]]:
        """Train the NER model."""
        print("\n" + "=" * 60)
        print("STEP 3: MODEL TRAINING")
        print("=" * 60)
        
        # Prepare dataloaders
        print("Preparing training and validation data...")
        train_dataloader, val_dataloader = self.trainer.prepare_dataloaders(dataset)
        
        print(f"✓ Training samples: {len(train_dataloader.dataset)}")
        print(f"✓ Validation samples: {len(val_dataloader.dataset)}")
        print(f"✓ Training batches: {len(train_dataloader)}")
        print(f"✓ Validation batches: {len(val_dataloader)}")
        
        # Start training
        print(f"\nStarting training for {self.config.num_epochs} epochs...")
        self.history = self.trainer.train(train_dataloader, val_dataloader)
        
        print(f"✓ Training completed!")
        return self.history
    
    def evaluate_model(self, dataset: List[Dict]) -> Dict:
        """Evaluate the trained model."""
        print("\n" + "=" * 60)
        print("STEP 4: MODEL EVALUATION")
        print("=" * 60)
        
        # Prepare test data
        _, test_dataloader = self.trainer.prepare_dataloaders(dataset, test_size=0.3)
        
        # Evaluate
        evaluation_results = self._detailed_evaluation(test_dataloader)
        
        # Save evaluation results
        results_path = "results/metrics/evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        print(f"✓ Evaluation completed!")
        print(f"✓ Results saved to: {results_path}")
        
        return evaluation_results
    
    def _detailed_evaluation(self, test_dataloader) -> Dict:
        """Perform detailed evaluation of the model."""
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_tokens = []
        
        print("Running evaluation on test set...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                # Move to device
                batch = {k: v.to(self.trainer.device) for k, v in batch.items()}
                
                # Get predictions
                predictions, probabilities = self.model.predict(
                    batch['input_ids'], 
                    batch['attention_mask']
                )
                
                # Convert to numpy
                pred_np = predictions.cpu().numpy()
                labels_np = batch['labels'].cpu().numpy()
                
                # Process each sequence in the batch
                for i in range(pred_np.shape[0]):
                    pred_seq = []
                    label_seq = []
                    
                    for j in range(pred_np.shape[1]):
                        if labels_np[i][j] != -100:  # Valid label
                            pred_label = self.config.id2label[pred_np[i][j]]
                            true_label = self.config.id2label[labels_np[i][j]]
                            
                            pred_seq.append(pred_label)
                            label_seq.append(true_label)
                    
                    if pred_seq and label_seq:  # Non-empty sequences
                        all_predictions.append(pred_seq)
                        all_labels.append(label_seq)
        
        print(f"Processed {len(all_predictions)} sequences")
        
        # Calculate metrics using seqeval
        f1 = f1_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        
        # Detailed classification report
        report = seq_classification_report(all_labels, all_predictions)
        
        evaluation_results = {
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'classification_report': report,
            'num_test_sequences': len(all_predictions)
        }
        
        # Print results
        print(f"\nEvaluation Results:")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"\nDetailed Classification Report:")
        print(report)
        
        return evaluation_results
    
    def plot_training_history(self):
        """Plot training history."""
        if not self.history:
            print("No training history available.")
            return
        
        print("\n" + "=" * 60)
        print("STEP 5: PLOTTING TRAINING HISTORY")
        print("=" * 60)
        
        # Create plots
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        epochs = range(1, len(self.history['train_loss']) + 1)
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Training Loss')
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(epochs, self.history['val_accuracy'], 'g-', label='Validation Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = "results/plots/training_history.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Training history plot saved to: {plot_path}")
    
    def save_model(self, model_name: str = "document_ner_model"):
        """Save the trained model."""
        print("\n" + "=" * 60)
        print("STEP 6: SAVING MODEL")
        print("=" * 60)
        
        save_path = f"models/{model_name}"
        self.trainer.save_model(save_path)
        
        # Save training history
        history_path = f"{save_path}/training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"✓ Model saved to: {save_path}")
        print(f"✓ Training history saved to: {history_path}")
        
        return save_path
    
    def run_complete_pipeline(self, data_path: Optional[str] = None, 
                            model_name: str = "document_ner_model") -> str:
        """Run the complete training pipeline."""
        print("🚀 STARTING COMPLETE TRAINING PIPELINE")
        print("=" * 80)
        
        try:
            # Step 1: Prepare data
            dataset = self.prepare_data(data_path)
            
            # Step 2: Initialize model
            self.initialize_model()
            
            # Step 3: Train model
            self.train_model(dataset)
            
            # Step 4: Evaluate model
            self.evaluate_model(dataset)
            
            # Step 5: Plot training history
            self.plot_training_history()
            
            # Step 6: Save model
            model_path = self.save_model(model_name)
            
            print("\n" + "🎉" * 20)
            print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            print("🎉" * 20)
            print(f"✅ Model saved to: {model_path}")
            print(f"✅ Training completed in {self.config.num_epochs} epochs")
            print(f"✅ Final validation accuracy: {self.history['val_accuracy'][-1]:.4f}")
            
            return model_path
            
        except Exception as e:
            print(f"\n❌ Error in training pipeline: {e}")
            raise


def create_custom_config() -> ModelConfig:
    """Create a custom configuration for training."""
    config = ModelConfig(
        model_name="distilbert-base-uncased",
        max_length=256,  # Shorter sequences for faster training
        batch_size=16,   # Adjust based on your GPU memory
        learning_rate=2e-5,
        num_epochs=3,
        warmup_steps=500,
        weight_decay=0.01,
        dropout_rate=0.1
    )
    
    return config


def main():
    """Main function to run the complete training pipeline."""
    print("Document Text Extraction - Training Pipeline")
    print("=" * 50)
    
    # Create custom configuration
    config = create_custom_config()
    
    # Initialize training pipeline
    pipeline = TrainingPipeline(config)
    
    # Run complete pipeline
    # You can provide a path to your document folder here
    # pipeline.run_complete_pipeline(data_path="data/raw")
    
    # For demonstration, we'll use sample data
    model_path = pipeline.run_complete_pipeline()
    
    print(f"\n🎯 Training completed! Model saved to: {model_path}")
    print("You can now use this model for document text extraction!")


if __name__ == "__main__":
    main()