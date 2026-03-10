"""
AI Text Detection Model using BERT
This module implements a BERT-based model for detecting AI-generated text.
"""

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from typing import Tuple, Optional, List
import os


class TextDeepfakeDetector(nn.Module):
    """
    BERT-based binary classifier for AI-generated text detection.
    Human (0) vs AI (1)
    """
    
    def __init__(self, model_name: str = 'bert-base-uncased', num_labels: int = 2):
        super(TextDeepfakeDetector, self).__init__()
        
        # Load pretrained BERT model
        self.bert = BertModel.from_pretrained(model_name)
        
        # Freeze BERT parameters initially (can be unfrozen for fine-tuning)
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Get hidden size from BERT config
        hidden_size = self.bert.config.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_labels)
        )
        
    def unfreeze_bert(self, num_layers: int = 4):
        """
        Unfreeze last N layers of BERT for fine-tuning.
        
        Args:
            num_layers: Number of encoder layers to unfreeze
        """
        # Unfreeze pooler and last N encoder layers
        for param in self.bert.pooler.parameters():
            param.requires_grad = True
        
        encoder_layers = len(self.bert.encoder.layer)
        for i in range(encoder_layers - num_layers, encoder_layers):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = True
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            token_type_ids: Segment IDs
            
        Returns:
            Logits for each class
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use [CLS] token representation for classification
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        
        return logits
    
    def predict_proba(self, input_ids, attention_mask=None, token_type_ids=None):
        """Return probability scores for each class"""
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask, token_type_ids)
            probabilities = torch.softmax(logits, dim=1)
        return probabilities


class TextModelHandler:
    """
    Handler class for loading model and making predictions on text.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the text model handler.
        
        Args:
            model_path: Path to trained model directory or weights
            device: Device to run inference on ('cuda' or 'cpu')
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize tokenizer
        self.model_name = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        
        # Initialize model
        self.model = TextDeepfakeDetector(model_name=self.model_name)
        self.model_path = model_path
        
        # Max sequence length
        self.max_length = 512
        
        # Load model weights if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
    def load_model(self, model_path: str):
        """
        Load trained model weights or entire model directory.
        
        Args:
            model_path: Path to model directory or .pth file
        """
        try:
            if os.path.isdir(model_path):
                # Load from HuggingFace format directory
                self.model = TextDeepfakeDetector.from_pretrained(model_path)
                print(f"✓ Loaded text model from directory: {model_path}")
            else:
                # Load from .pth file
                checkpoint = torch.load(model_path, map_location=self.device)
                
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['state_dict'])
                    else:
                        self.model.load_state_dict(checkpoint)
                else:
                    self.model.load_state_dict(checkpoint)
                    
                print(f"✓ Loaded text model from: {model_path}")
                
        except Exception as e:
            print(f"⚠ Warning: Could not load model weights: {e}")
            print("Using randomly initialized weights")
    
    def tokenize_text(self, text: str, max_length: Optional[int] = None) -> dict:
        """
        Tokenize text for BERT model.
        
        Args:
            text: Input text string
            max_length: Maximum sequence length
            
        Returns:
            Tokenized output dict with input_ids, attention_mask, token_type_ids
        """
        if max_length is None:
            max_length = self.max_length
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )
        
        return encoding
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict whether text is human-written or AI-generated.
        
        Args:
            text: Input text string
            
        Returns:
            Tuple of (prediction_label, confidence_score)
        """
        try:
            # Validate input
            if not text or not text.strip():
                raise ValueError("Empty text input")
            
            # Tokenize
            encoding = self.tokenize_text(text)
            
            # Move tensors to device
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            token_type_ids = encoding['token_type_ids'].to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                probabilities = torch.softmax(outputs, dim=1)
                
                # Get predicted class and confidence
                confidence, predicted = torch.max(probabilities, 1)
                
                prediction = "AI Generated" if predicted.item() == 1 else "Human"
                confidence_score = confidence.item() * 100  # Convert to percentage
                
            return prediction, confidence_score
            
        except Exception as e:
            raise Exception(f"Error during text prediction: {e}")
    
    def predict_batch(self, texts: List[str]) -> list:
        """
        Predict multiple texts in batch.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of (text, prediction, confidence) tuples
        """
        results = []
        
        for text in texts:
            try:
                prediction, confidence = self.predict(text)
                results.append((text[:50] + "...", prediction, confidence))
            except Exception as e:
                results.append((text[:50] + "...", f"Error: {str(e)}", 0.0))
        
        return results


def initialize_text_model(model_path: Optional[str] = None) -> TextModelHandler:
    """
    Factory function to initialize text model.
    
    Args:
        model_path: Path to trained model weights or directory
        
    Returns:
        TextModelHandler instance
    """
    return TextModelHandler(model_path=model_path)


# Example usage and testing
if __name__ == "__main__":
    # Test initialization
    print("Initializing Text Deepfake Detector...")
    handler = initialize_text_model()
    print(f"Model loaded on device: {handler.device}")
    print("Model architecture:")
    print(handler.model)
    
    # Test prediction
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is revolutionizing technology."
    ]
    
    for text in test_texts:
        prediction, confidence = handler.predict(text)
        print(f"\nText: {text}")
        print(f"Prediction: {prediction} ({confidence:.2f}%)")
