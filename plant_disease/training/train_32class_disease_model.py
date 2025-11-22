"""
32-Class Plant Disease Detection Training Script

This script trains a 32-class plant disease detection model using the PlantVillage dataset.
It's specifically designed for a subset of the PlantVillage dataset with 32 classes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18, ResNet18_Weights
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

# Define the 32 plant disease classes
PLANT_DISEASE_CLASSES_32 = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___healthy'
]

def create_32class_dataset(data_dir, img_size=224):
    """Create dataset with transformations for 32-class plant disease detection"""
    
    # Define training transformations with data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),  # Resize to slightly larger size
        transforms.RandomCrop((img_size, img_size)),        # Random crop to final size
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Define validation/testing transformations
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load full dataset
    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    
    # Filter dataset to only include the 32 classes we want
    class_to_idx = {cls_name: idx for cls_name, idx in full_dataset.class_to_idx.items() 
                    if cls_name in PLANT_DISEASE_CLASSES_32}
    
    # Get indices of samples that belong to our 32 classes
    indices = [i for i, (_, label) in enumerate(full_dataset.imgs) if label in class_to_idx.values()]
    
    # Create subset dataset
    subset_dataset = torch.utils.data.Subset(full_dataset, indices)
    
    # Update class_to_idx mapping for the subset
    subset_dataset.dataset.class_to_idx = class_to_idx
    subset_dataset.dataset.classes = [cls for cls in full_dataset.classes if cls in PLANT_DISEASE_CLASSES_32]
    
    # Create separate datasets for training and validation transforms
    train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=data_dir, transform=val_transform)
    
    # Filter both datasets to only include the 32 classes
    train_indices = [i for i, (_, label) in enumerate(train_dataset.imgs) if label in class_to_idx.values()]
    val_indices = [i for i, (_, label) in enumerate(val_dataset.imgs) if label in class_to_idx.values()]
    
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
    
    # Update class mappings
    train_dataset.dataset.class_to_idx = class_to_idx
    val_dataset.dataset.class_to_idx = class_to_idx
    train_dataset.dataset.classes = PLANT_DISEASE_CLASSES_32
    val_dataset.dataset.classes = PLANT_DISEASE_CLASSES_32
    
    return train_dataset, val_dataset

def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Split dataset into train, validation, and test sets"""
    
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # Ensure we have at least one sample in each set
    if train_size == 0:
        train_size = total_size
        val_size = 0
        test_size = 0
    elif val_size == 0 and test_size == 0:
        val_size = total_size - train_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    return train_dataset, val_dataset, test_dataset

def create_model(num_classes=32, pretrained=True):
    """Create ResNet18 model for 32-class plant disease detection"""
    
    if pretrained:
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
    else:
        model = resnet18()
    
    # Replace the final fully connected layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs=25, device='cpu'):
    """Train the model with validation"""
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    model = model.to(device)
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc='Training')
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{running_loss/len(train_loader):.4f}',
                'Acc': f'{100 * correct / total:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc='Validation')
            for images, labels in progress_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                progress_bar.set_postfix({
                    'Loss': f'{val_loss/len(val_loader):.4f}',
                    'Acc': f'{100 * val_correct / val_total:.2f}%'
                })
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Step the scheduler
        scheduler.step()
    
    return train_losses, val_losses, train_accuracies, val_accuracies

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    """Plot training and validation metrics"""
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def save_model_info(model, num_classes, class_names, train_acc, val_acc, save_path):
    """Save model information to a JSON file"""
    
    model_info = {
        'num_classes': num_classes,
        'class_names': class_names,
        'final_train_accuracy': train_acc,
        'final_validation_accuracy': val_acc,
        'model_architecture': 'ResNet18',
        'training_complete': True
    }
    
    info_path = save_path.replace('.pth', '_info.json')
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"Model information saved to {info_path}")

def main():
    """Main training function"""
    
    print("32-Class Plant Disease Detection Model Training")
    print("=" * 50)
    
    # Configuration
    data_dir = "../datasets/plant_disease/images"  # Adjust path as needed
    model_save_path = "plant_disease_32class_model.pth"
    batch_size = 32
    num_epochs = 25
    learning_rate = 0.001
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    try:
        # Create datasets
        print("Loading and preparing datasets...")
        train_dataset, val_dataset = create_32class_dataset(data_dir)
        
        print(f"Dataset loaded with {len(train_dataset)} training samples")
        print(f"Number of classes: {len(PLANT_DISEASE_CLASSES_32)}")
        print("Classes:")
        for i, class_name in enumerate(PLANT_DISEASE_CLASSES_32):
            print(f"  {i+1:2d}. {class_name}")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        print(f"Data loaders created with batch size: {batch_size}")
        
        # Create model
        print("Creating model...")
        model = create_model(num_classes=32, pretrained=True)
        model = model.to(device)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
        
        print("Starting training...")
        
        # Train the model
        train_losses, val_losses, train_accuracies, val_accuracies = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler, 
            num_epochs=num_epochs, device=device
        )
        
        print("Training completed!")
        
        # Plot training history
        plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)
        
        # Save the model
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
        
        # Save model information
        final_train_acc = train_accuracies[-1] if train_accuracies else 0
        final_val_acc = val_accuracies[-1] if val_accuracies else 0
        save_model_info(model, 32, PLANT_DISEASE_CLASSES_32, 
                       final_train_acc, final_val_acc, model_save_path)
        
        print("\nTraining Summary:")
        print(f"Final Training Accuracy: {final_train_acc:.2f}%")
        print(f"Final Validation Accuracy: {final_val_acc:.2f}%")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()