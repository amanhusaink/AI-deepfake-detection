"""
AI Text Detection Model Training Script
Trains a BERT-based model to classify text as Human-written or AI-generated.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json


class TextDataset(Dataset):
    """Custom dataset for AI text detection"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        """
        Args:
            texts: List of text strings
            labels: List of labels (0=Human, 1=AI)
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class TextDeepfakeDetector(nn.Module):
    """BERT-based binary classifier for AI text detection"""
    
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        super(TextDeepfakeDetector, self).__init__()
        
        self.bert = BertModel.from_pretrained(model_name)
        
        # Freeze BERT parameters initially
        for param in self.bert.parameters():
            param.requires_grad = False
        
        hidden_size = self.bert.config.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_labels)
        )
        
    def unfreeze_bert(self, num_layers=4):
        """Unfreeze last N layers of BERT"""
        for param in self.bert.pooler.parameters():
            param.requires_grad = True
        
        encoder_layers = len(self.bert.encoder.layer)
        for i in range(encoder_layers - num_layers, encoder_layers):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = True
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        
        return logits


def load_data_from_directory(data_dir):
    """
    Load text data from directory structure.
    
    Directory format:
    data_dir/
        human/
            *.txt
        ai/
            *.txt
    """
    texts = []
    labels = []
    
    # Load human texts (label 0)
    human_dir = os.path.join(data_dir, 'human')
    if os.path.exists(human_dir):
        for filename in os.listdir(human_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(human_dir, filename), 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    if len(text) >= 50:  # Minimum length filter
                        texts.append(text)
                        labels.append(0)
    
    # Load AI texts (label 1)
    ai_dir = os.path.join(data_dir, 'ai')
    if os.path.exists(ai_dir):
        for filename in os.listdir(ai_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(ai_dir, filename), 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    if len(text) >= 50:
                        texts.append(text)
                        labels.append(1)
    
    print(f"Loaded {len(texts)} texts from {data_dir}")
    print(f"  Human: {labels.count(0)}, AI: {labels.count(1)}")
    
    return texts, labels


def load_data_from_json(json_path):
    """
    Load text data from JSON file.
    
    JSON format:
    [
        {"text": "...", "label": 0},  # 0=Human, 1=AI
        ...
    ]
    """
    texts = []
    labels = []
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for item in data:
        text = item.get('text', '')
        label = item.get('label', 0)
        
        if len(text) >= 50:
            texts.append(text)
            labels.append(label)
    
    print(f"Loaded {len(texts)} texts from {json_path}")
    print(f"  Human: {labels.count(0)}, AI: {labels.count(1)}")
    
    return texts, labels


def train_model(
    data_dir=None,
    json_file=None,
    model_save_path='models/text_model',
    batch_size=16,
    num_epochs=5,
    learning_rate=2e-5,
    max_length=512
):
    """
    Train the AI text detection model.
    
    Args:
        data_dir: Directory with human/ and ai/ subdirectories
        json_file: Path to JSON file with labeled data
        model_save_path: Directory to save trained model
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        max_length: Maximum sequence length
    """
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    if json_file and os.path.exists(json_file):
        texts, labels = load_data_from_json(json_file)
    elif data_dir and os.path.exists(data_dir):
        texts, labels = load_data_from_directory(data_dir)
    else:
        print("\n⚠ Warning: No training data found!")
        print("\nTo train the model, provide data in one of these formats:")
        print("1. JSON file with --json-file argument")
        print("2. Directory structure: data_dir/human/*.txt and data_dir/ai/*.txt")
        print("\nYou can use datasets like:")
        print("- OpenAI Text Classifier dataset")
        "- HC3 (Human ChatGPT Comparison Corpus)")
        exit(1)
    
    if len(texts) == 0:
        print("\n⚠ Error: No valid training data loaded!")
        exit(1)
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\nTraining samples: {len(train_texts)}, Validation samples: {len(val_texts)}")
    
    # Initialize tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # Create datasets
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, max_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Initialize model
    model = TextDeepfakeDetector(model_name=model_name).to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Training tracking
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Phase 1: Train with frozen BERT (first 2 epochs)
        if epoch < 2:
            print("Phase: Training with frozen BERT backbone")
        else:
            # Unfreeze BERT for fine-tuning
            if epoch == 2:
                print("Phase: Unfreezing BERT for fine-tuning")
                model.unfreeze_bert(num_layers=4)
        
        # Training phase
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc=f'Train {epoch+1}')
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels_batch = batch['label'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(outputs, labels_batch)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Statistics
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
            
            acc = accuracy_score(all_labels, all_preds)
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.4f}'})
        
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = accuracy_score(all_labels, all_preds)
        train_losses.append(epoch_train_loss)
        
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
        
        # Validation phase
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Val {epoch+1}')
            for batch in pbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels_batch = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask, token_type_ids)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels_batch.cpu().numpy())
                
                acc = accuracy_score(all_labels, all_preds)
                pbar.set_postfix({'val_acc': f'{acc:.4f}'})
        
        epoch_val_acc = accuracy_score(all_labels, all_preds)
        val_precision = precision_score(all_labels, all_preds)
        val_recall = recall_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds)
        
        val_accuracies.append(epoch_val_acc)
        
        print(f"\nValidation Results:")
        print(f"  Accuracy:  {epoch_val_acc:.4f}")
        print(f"  Precision: {val_precision:.4f}")
        print(f"  Recall:    {val_recall:.4f}")
        print(f"  F1 Score:  {val_f1:.4f}")
        
        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_metrics = {
                'accuracy': epoch_val_acc,
                'precision': val_precision,
                'recall': val_recall,
                'f1_score': val_f1
            }
            
            # Save model
            os.makedirs(model_save_path, exist_ok=True)
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            
            # Save training metadata
            metadata = {
                'epoch': epoch,
                'val_accuracy': epoch_val_acc,
                'train_loss': epoch_train_loss,
                'metrics': best_metrics,
                'model_name': model_name,
                'max_length': max_length
            }
            
            with open(os.path.join(model_save_path, 'training_metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"\n✓ Saved best model to: {model_save_path}")
            print(f"  Best validation accuracy: {best_val_acc:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, 'g-', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('text_training_plot.png', dpi=300)
    print("\n✓ Saved training plot to text_training_plot.png")
    
    print(f"\n✓ Training completed! Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {model_save_path}")
    
    return model, best_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train AI Text Detection Model')
    parser.add_argument('--data-dir', type=str, default='data/training/text',
                       help='Directory with human/ and ai/ subdirectories')
    parser.add_argument('--json-file', type=str, default=None,
                       help='Path to JSON file with labeled data')
    parser.add_argument('--output', type=str, default='models/text_model',
                       help='Output directory for trained model')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("AI Text Detection Model Training")
    print("=" * 60)
    
    # Start training
    model, metrics = train_model(
        data_dir=args.data_dir,
        json_file=args.json_file,
        model_save_path=args.output,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr
    )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Final Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    print("=" * 60)
