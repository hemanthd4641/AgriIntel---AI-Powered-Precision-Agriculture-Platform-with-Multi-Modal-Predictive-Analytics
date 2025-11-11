"""
CNN Model for Satellite Image Processing in Smart Agriculture Project

This module implements a Convolutional Neural Network for processing
satellite imagery to extract spatial features for crop yield prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SatelliteCNN(nn.Module):
    """CNN model for satellite image feature extraction"""
    
    def __init__(self, input_channels=3, num_classes=1, dropout_rate=0.5):
        """
        Initialize the CNN model
        
        Args:
            input_channels: Number of input channels (e.g., RGB = 3)
            num_classes: Number of output classes (1 for regression)
            dropout_rate: Dropout rate for regularization
        """
        super(SatelliteCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Calculate the size of flattened features after convolutions
        # Assuming input size is 256x256
        self.flattened_size = 256 * 16 * 16  # After 4 pooling operations: 256/(2^4) = 16
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Convolutional layers with ReLU activation and batch normalization
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten the tensor for fully connected layers
        x = x.view(-1, self.flattened_size)
        
        # Fully connected layers with ReLU activation and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Output layer
        x = self.fc3(x)
        
        return x

class NDVICNN(nn.Module):
    """Specialized CNN for NDVI image processing"""
    
    def __init__(self, num_classes=1, dropout_rate=0.3):
        """
        Initialize the NDVI CNN model
        
        Args:
            num_classes: Number of output classes (1 for regression)
            dropout_rate: Dropout rate for regularization
        """
        super(NDVICNN, self).__init__()
        
        # Convolutional layers for single-channel NDVI input
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Calculate the size of flattened features
        # Assuming input size is 256x256, after 3 pooling: 256/(2^3) = 32
        self.flattened_size = 128 * 32 * 32
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, 1, height, width)
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Add channel dimension if not present
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        # Convolutional layers with ReLU activation and batch normalization
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten the tensor for fully connected layers
        x = x.view(-1, self.flattened_size)
        
        # Fully connected layers with ReLU activation and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Output layer
        x = self.fc3(x)
        
        return x

# Example usage
if __name__ == "__main__":
    # Test the CNN models
    # Create a sample input tensor (batch_size=4, channels=3, height=256, width=256)
    sample_input = torch.randn(4, 3, 256, 256)
    
    # Initialize the model
    model = SatelliteCNN(input_channels=3, num_classes=1)
    
    # Forward pass
    output = model(sample_input)
    print(f"SatelliteCNN output shape: {output.shape}")
    
    # Test NDVI CNN
    ndvi_input = torch.randn(4, 1, 256, 256)
    ndvi_model = NDVICNN(num_classes=1)
    ndvi_output = ndvi_model(ndvi_input)
    print(f"NDVICNN output shape: {ndvi_output.shape}")