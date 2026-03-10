"""
Image Deepfake Detection Model Training Script
Trains a ResNet50-based model to classify images as Real or Fake.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path


class DeepfakeImageDataset(Dataset):
    """Custom dataset for deepfake image detection"""
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Directory with real/ and fake/ subdirectories
            transform: Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.labels = []
        
        # Load real images (label 0)
        real_dir = os.path.join(root_dir, 'real')
        if os.path.exists(real_dir):
            for img_name in os.listdir(real_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    self.samples.append(os.path.join(real_dir, img_name))
                    self.labels.append(0)
        
        # Load fake images (label 1)
        fake_dir = os.path.join(root_dir, 'fake')
        if os.path.exists(fake_dir):
            for img_name in os.listdir(fake_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    self.samples.append(os.path.join(fake_dir, img_name))
                    self.labels.append(1)
        
        print(f"Loaded {len(self.samples)} images from {root_dir}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if there's an error
            return torch.zeros((3, 224, 224)), label


class ImageDeepfakeDetector(nn.Module):
    """ResNet50-based binary classifier"""
    
    def __init__(self, pretrained=True, dropout_rate=0.5):
        super(ImageDeepfakeDetector, self).__init__()
        
        # Load pretrained ResNet50
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Replace final layer for binary classification
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 2)
        )
        
    def forward(self, x):
        return self.resnet(x)


def train_model(
    data_dir='data/training/images',
    model_save_path='models/image_model.pth',
    batch_size=32,
    num_epochs=15,
    learning_rate=0.001,
    img_size=224
):
    """
    Train the deepfake image detection model.
    
    Args:
        data_dir: Directory containing training data
        model_save_path: Path to save trained model
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        img_size: Image resize dimension
    """
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transformations
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    full_dataset = DeepfakeImageDataset(data_dir, transform=train_transform)
    
    # Split into train and validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Update validation dataset transform
    val_dataset.dataset.transform = val_transform
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Initialize model
    model = ImageDeepfakeDetector(pretrained=True).to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    # Training tracking
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Train {epoch+1}')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})
        
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100 * correct / total
        train_losses.append(epoch_train_loss)
        
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%")
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Val {epoch+1}')
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({'val_acc': f'{100*correct/total:.2f}%'})
        
        epoch_val_acc = 100 * correct / total
        val_accuracies.append(epoch_val_acc)
        
        print(f"Validation Accuracy: {epoch_val_acc:.2f}%")
        
        # Update learning rate
        scheduler.step(epoch_val_acc)
        
        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': epoch_val_acc,
                'train_loss': epoch_train_loss
            }
            
            # Create directory if needed
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(checkpoint, model_save_path)
            print(f"✓ Saved best model with val accuracy: {best_val_acc:.2f}%")
    
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
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_plot.png', dpi=300)
    print("✓ Saved training plot to training_plot.png")
    
    print(f"\n✓ Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {model_save_path}")
    
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("Deepfake Image Detection Model Training")
    print("=" * 60)
    
    # Check if training data exists
    data_dir = 'data/training/images'
    if not os.path.exists(data_dir):
        print(f"\n⚠ Warning: Training data directory '{data_dir}' not found!")
        print("\nTo train the model, you need to organize your dataset as follows:")
        print(f"{data_dir}/real/  - Place real images here")
        print(f"{data_dir}/fake/  - Place fake/deepfake images here")
        print("\nYou can use public datasets like:")
        print("- FaceForensics++")
        print("- DeepFake Detection Challenge (DFDC)")
        print("- Celeb-DF")
        exit(1)
    
    # Start training
    train_model(
        data_dir=data_dir,
        model_save_path='models/image_model.pth',
        batch_size=32,
        num_epochs=15,
        learning_rate=0.001
    )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
