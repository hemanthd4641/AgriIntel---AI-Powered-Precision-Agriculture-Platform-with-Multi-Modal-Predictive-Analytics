"""
Model Training Module for Smart Agriculture Project

This module handles the training of the CNN+RNN fusion model for crop yield prediction.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

class ModelTrainer:
    """Class for training the crop yield prediction model"""
    
    def __init__(self, model, device=None):
        """
        Initialize the model trainer
        
        Args:
            model: PyTorch model to train
            device: Device to train on (cuda or cpu)
        """
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, dataloader, optimizer, criterion):
        """
        Train the model for one epoch
        
        Args:
            dataloader: DataLoader for training data
            optimizer: Optimizer for model parameters
            criterion: Loss function
            
        Returns:
            float: Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            # Move data to device
            if isinstance(data, (list, tuple)):
                data = [d.to(self.device) for d in data]
            else:
                data = data.to(self.device)
            target = target.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            if isinstance(data, (list, tuple)):
                output = self.model(*data)
            else:
                output = self.model(data)
            
            # Calculate loss
            loss = criterion(output.squeeze(), target.float())
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
    
    def validate(self, dataloader, criterion):
        """
        Validate the model
        
        Args:
            dataloader: DataLoader for validation data
            criterion: Loss function
            
        Returns:
            float: Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in dataloader:
                # Move data to device
                if isinstance(data, (list, tuple)):
                    data = [d.to(self.device) for d in data]
                else:
                    data = data.to(self.device)
                target = target.to(self.device)
                
                # Forward pass
                if isinstance(data, (list, tuple)):
                    output = self.model(*data)
                else:
                    output = self.model(data)
                
                # Calculate loss
                loss = criterion(output.squeeze(), target.float())
                total_loss += loss.item()
                
        return total_loss / len(dataloader)
    
    def train(self, train_loader, val_loader, epochs=50, learning_rate=0.001, 
              weight_decay=1e-5, save_path='models/saved_models/best_model.pth'):
        """
        Train the model
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            save_path: Path to save the best model
        """
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        best_val_loss = float('inf')
        
        print(f"Training on device: {self.device}")
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Train for one epoch
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader, criterion)
            self.val_losses.append(val_loss)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.model.state_dict(), save_path)
                print(f"Epoch {epoch+1}: Saved new best model with validation loss: {val_loss:.4f}")
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    def evaluate(self, test_loader):
        """
        Evaluate the model on test data
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                # Move data to device
                if isinstance(data, (list, tuple)):
                    data = [d.to(self.device) for d in data]
                else:
                    data = data.to(self.device)
                target = target.to(self.device)
                
                # Forward pass
                if isinstance(data, (list, tuple)):
                    output = self.model(*data)
                else:
                    output = self.model(data)
                
                predictions.extend(output.squeeze().cpu().numpy())
                targets.extend(target.cpu().numpy())
        
        # Calculate metrics
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, predictions)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'predictions': predictions,
            'targets': targets
        }
    
    def plot_training_history(self, save_path='training_history.png'):
        """
        Plot training history
        
        Args:
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Model Training History')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        
        print(f"Training history plot saved to {save_path}")

# Example usage
if __name__ == "__main__":
    # This would typically be run from a training script
    print("Model training module ready")
    
    # Example of how to use:
    # 1. Initialize your model (CNN, RNN, or Fusion model)
    # 2. Create data loaders
    # 3. Initialize trainer
    # 4. Call train() method
    # 5. Evaluate and plot results