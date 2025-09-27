"""
Small Language Model (SLM) architecture for document text extraction.
Uses DistilBERT with transfer learning for Named Entity Recognition.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer, 
    DistilBertForTokenClassification,
    DistilBertConfig,
    get_linear_schedule_with_warmup
)
from typing import List, Dict, Tuple, Optional
import json
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for the SLM model."""
    model_name: str = "distilbert-base-uncased"
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    dropout_rate: float = 0.3
    
    # Entity labels
    entity_labels: List[str] = None
    
    def __post_init__(self):
        if self.entity_labels is None:
            self.entity_labels = [
                'O', 'B-NAME', 'I-NAME', 'B-DATE', 'I-DATE', 
                'B-INVOICE_NO', 'I-INVOICE_NO', 'B-AMOUNT', 'I-AMOUNT',
                'B-ADDRESS', 'I-ADDRESS', 'B-PHONE', 'I-PHONE',
                'B-EMAIL', 'I-EMAIL'
            ]
    
    @property
    def num_labels(self) -> int:
        return len(self.entity_labels)
    
    @property
    def label2id(self) -> Dict[str, int]:
        return {label: i for i, label in enumerate(self.entity_labels)}
    
    @property
    def id2label(self) -> Dict[int, str]:
        return {i: label for i, label in enumerate(self.entity_labels)}


class NERDataset(Dataset):
    """PyTorch Dataset for NER training."""
    
    def __init__(self, dataset: List[Dict], tokenizer: DistilBertTokenizer, 
                 config: ModelConfig, mode: str = 'train'):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.config = config
        self.mode = mode
        
        # Prepare tokenized data
        self.tokenized_data = self._tokenize_and_align_labels()
    
    def _tokenize_and_align_labels(self) -> List[Dict]:
        """Tokenize text and align labels with subword tokens."""
        tokenized_data = []
        
        for example in self.dataset:
            tokens = example['tokens']
            labels = example['labels']
            
            # Tokenize each word and track alignments
            tokenized_inputs = self.tokenizer(
                tokens,
                is_split_into_words=True,
                padding='max_length',
                truncation=True,
                max_length=self.config.max_length,
                return_tensors='pt'
            )
            
            # Align labels with subword tokens
            word_ids = tokenized_inputs.word_ids()
            aligned_labels = []
            previous_word_idx = None
            
            for word_idx in word_ids:
                if word_idx is None:
                    # Special tokens get -100 (ignored in loss computation)
                    aligned_labels.append(-100)
                elif word_idx != previous_word_idx:
                    # First subword of a word gets the original label
                    if word_idx < len(labels):
                        label = labels[word_idx]
                        aligned_labels.append(self.config.label2id.get(label, 0))
                    else:
                        aligned_labels.append(-100)
                else:
                    # Subsequent subwords of the same word
                    if word_idx < len(labels):
                        label = labels[word_idx]
                        if label.startswith('B-'):
                            # Convert B- to I- for subword tokens
                            i_label = label.replace('B-', 'I-')
                            aligned_labels.append(self.config.label2id.get(i_label, 0))
                        else:
                            aligned_labels.append(self.config.label2id.get(label, 0))
                    else:
                        aligned_labels.append(-100)
                
                previous_word_idx = word_idx
            
            tokenized_data.append({
                'input_ids': tokenized_inputs['input_ids'].squeeze(),
                'attention_mask': tokenized_inputs['attention_mask'].squeeze(),
                'labels': torch.tensor(aligned_labels, dtype=torch.long),
                'original_tokens': tokens,
                'original_labels': labels
            })
        
        return tokenized_data
    
    def __len__(self) -> int:
        return len(self.tokenized_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'input_ids': self.tokenized_data[idx]['input_ids'],
            'attention_mask': self.tokenized_data[idx]['attention_mask'],
            'labels': self.tokenized_data[idx]['labels']
        }


class DocumentNERModel(nn.Module):
    """DistilBERT-based model for document NER."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Load pre-trained DistilBERT configuration
        bert_config = DistilBertConfig.from_pretrained(
            config.model_name,
            num_labels=config.num_labels,
            id2label=config.id2label,
            label2id=config.label2id,
            dropout=config.dropout_rate,
            attention_dropout=config.dropout_rate
        )
        
        # Initialize model with token classification head
        self.model = DistilBertForTokenClassification.from_pretrained(
            config.model_name,
            config=bert_config
        )
        
        # Additional dropout layer for regularization
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass through the model."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs
    
    def predict(self, input_ids, attention_mask):
        """Make predictions without computing loss."""
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            predictions = torch.argmax(outputs.logits, dim=-1)
            probabilities = torch.softmax(outputs.logits, dim=-1)
        
        return predictions, probabilities


class NERTrainer:
    """Trainer class for the NER model."""
    
    def __init__(self, model: DocumentNERModel, config: ModelConfig):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(config.model_name)
    
    def prepare_dataloaders(self, dataset: List[Dict], 
                          test_size: float = 0.2) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation dataloaders."""
        # Split dataset
        train_data, val_data = train_test_split(
            dataset, test_size=test_size, random_state=42
        )
        
        # Create datasets
        train_dataset = NERDataset(train_data, self.tokenizer, self.config, 'train')
        val_dataset = NERDataset(val_data, self.tokenizer, self.config, 'val')
        
        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False
        )
        
        return train_dataloader, val_dataloader
    
    def train(self, train_dataloader: DataLoader, 
              val_dataloader: DataLoader) -> Dict[str, List[float]]:
        """Train the NER model."""
        # Initialize optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        total_steps = len(train_dataloader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        print(f"Training on device: {self.device}")
        print(f"Total training steps: {total_steps}")
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print("-" * 50)
            
            # Training phase
            train_loss = self._train_epoch(train_dataloader, optimizer, scheduler)
            history['train_loss'].append(train_loss)
            
            # Validation phase
            val_loss, val_accuracy = self._validate_epoch(val_dataloader)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.4f}")
        
        return history
    
    def _train_epoch(self, dataloader: DataLoader, optimizer, scheduler) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        return total_loss / len(dataloader)
    
    def _validate_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss
                
                total_loss += loss.item()
                
                # Calculate accuracy (ignoring -100 labels)
                predictions = torch.argmax(outputs.logits, dim=-1)
                labels = batch['labels']
                
                # Mask for valid labels (not -100)
                valid_mask = labels != -100
                
                correct = (predictions == labels) & valid_mask
                total_correct += correct.sum().item()
                total_tokens += valid_mask.sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        
        return avg_loss, accuracy
    
    def save_model(self, save_path: str):
        """Save the trained model and tokenizer."""
        self.model.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save config
        config_path = f"{save_path}/training_config.json"
        with open(config_path, 'w') as f:
            json.dump(vars(self.config), f, indent=2)
        
        print(f"Model saved to {save_path}")
    
    def load_model(self, model_path: str):
        """Load a pre-trained model."""
        self.model.model = DistilBertForTokenClassification.from_pretrained(model_path)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        print(f"Model loaded from {model_path}")


def create_model_and_trainer(config: Optional[ModelConfig] = None) -> Tuple[DocumentNERModel, NERTrainer]:
    """Create model and trainer with configuration."""
    if config is None:
        config = ModelConfig()
    
    model = DocumentNERModel(config)
    trainer = NERTrainer(model, config)
    
    return model, trainer


def main():
    """Demonstrate model creation and setup."""
    # Create configuration
    config = ModelConfig(
        batch_size=8,  # Smaller batch size for demo
        num_epochs=2,
        learning_rate=3e-5
    )
    
    print("Model Configuration:")
    print(f"Model: {config.model_name}")
    print(f"Max Length: {config.max_length}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Number of Labels: {config.num_labels}")
    print(f"Entity Labels: {config.entity_labels}")
    
    # Create model and trainer
    model, trainer = create_model_and_trainer(config)
    
    print(f"\nModel created successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model, trainer


if __name__ == "__main__":
    main()