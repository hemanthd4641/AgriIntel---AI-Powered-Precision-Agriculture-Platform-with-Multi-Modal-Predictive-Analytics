"""
Fusion Model for Combining CNN and RNN Features in Smart Agriculture Project

This module implements a fusion layer that combines features from the CNN
(satellite imagery) and RNN (weather data) models to predict crop yield.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionModel(nn.Module):
    """Fusion model that combines CNN and RNN features"""
    
    def __init__(self, cnn_feature_size=128, rnn_feature_size=64, num_classes=1, dropout_rate=0.5):
        """
        Initialize the fusion model
        
        Args:
            cnn_feature_size: Size of features from CNN model
            rnn_feature_size: Size of features from RNN model
            num_classes: Number of output classes (1 for regression)
            dropout_rate: Dropout rate for regularization
        """
        super(FusionModel, self).__init__()
        
        self.cnn_feature_size = cnn_feature_size
        self.rnn_feature_size = rnn_feature_size
        
        # Fully connected layers for feature fusion
        self.fc1 = nn.Linear(cnn_feature_size + rnn_feature_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, cnn_features, rnn_features):
        """
        Forward pass through the fusion network
        
        Args:
            cnn_features: Features from CNN model (batch_size, cnn_feature_size)
            rnn_features: Features from RNN model (batch_size, rnn_feature_size)
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Concatenate CNN and RNN features
        x = torch.cat((cnn_features, rnn_features), dim=1)
        
        # Fully connected layers with ReLU activation, batch normalization, and dropout
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        
        # Output layer
        x = self.fc4(x)
        
        return x

class EarlyFusionModel(nn.Module):
    """Early fusion model that combines raw inputs before feature extraction"""
    
    def __init__(self, image_channels=3, image_size=256, weather_features=10, sequence_length=30, num_classes=1):
        """
        Initialize the early fusion model
        
        Args:
            image_channels: Number of channels in satellite images
            image_size: Size of satellite images (assumed square)
            weather_features: Number of weather parameters
            sequence_length: Length of weather time series
            num_classes: Number of output classes (1 for regression)
        """
        super(EarlyFusionModel, self).__init__()
        
        # CNN for image processing
        self.conv1 = nn.Conv2d(image_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        # Calculate flattened image features size
        self.image_flattened_size = 128 * (image_size // 8) * (image_size // 8)
        self.fc_image = nn.Linear(self.image_flattened_size, 128)
        
        # LSTM for weather processing
        self.lstm = nn.LSTM(
            input_size=weather_features,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Fusion layer
        self.fc_fusion1 = nn.Linear(128 + 64, 256)
        self.fc_fusion2 = nn.Linear(256, 128)
        self.fc_fusion3 = nn.Linear(128, 64)
        self.fc_output = nn.Linear(64, num_classes)
        
        self.bn_fusion1 = nn.BatchNorm1d(256)
        self.bn_fusion2 = nn.BatchNorm1d(128)
        self.bn_fusion3 = nn.BatchNorm1d(64)
        
    def forward(self, images, weather_data):
        """
        Forward pass through the early fusion network
        
        Args:
            images: Satellite images (batch_size, channels, height, width)
            weather_data: Weather time series (batch_size, sequence_length, features)
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Process images with CNN
        img_features = self.pool(F.relu(self.bn1(self.conv1(images))))
        img_features = self.pool(F.relu(self.bn2(self.conv2(img_features))))
        img_features = self.pool(F.relu(self.bn3(self.conv3(img_features))))
        
        img_features = img_features.view(-1, self.image_flattened_size)
        img_features = F.relu(self.fc_image(img_features))
        img_features = self.dropout(img_features)
        
        # Process weather data with LSTM
        h0 = torch.zeros(2, weather_data.size(0), 64).to(weather_data.device)
        c0 = torch.zeros(2, weather_data.size(0), 64).to(weather_data.device)
        
        weather_out, _ = self.lstm(weather_data, (h0, c0))
        weather_features = weather_out[:, -1, :]  # Use last time step
        
        # Fuse features
        fused = torch.cat((img_features, weather_features), dim=1)
        
        # Fully connected layers
        fused = F.relu(self.bn_fusion1(self.fc_fusion1(fused)))
        fused = self.dropout(fused)
        fused = F.relu(self.bn_fusion2(self.fc_fusion2(fused)))
        fused = self.dropout(fused)
        fused = F.relu(self.bn_fusion3(self.fc_fusion3(fused)))
        fused = self.dropout(fused)
        
        # Output
        output = self.fc_output(fused)
        
        return output

class LateFusionModel(nn.Module):
    """Late fusion model with separate processing branches"""
    
    def __init__(self, num_classes=1):
        """
        Initialize the late fusion model with pre-trained CNN and RNN components
        
        Args:
            num_classes: Number of output classes (1 for regression)
        """
        super(LateFusionModel, self).__init__()
        
        # These would typically be pre-trained models loaded separately
        # For this implementation, we'll define simplified versions
        
        # Image processing branch (simplified)
        self.image_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((64, 64)),
            nn.Flatten(),
            nn.Linear(3 * 64 * 64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Weather processing branch (simplified)
        self.weather_branch = nn.Sequential(
            nn.LSTM(10, 64, batch_first=True),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Fusion and output
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, images, weather_data):
        """
        Forward pass through the late fusion network
        
        Args:
            images: Satellite images (batch_size, channels, height, width)
            weather_data: Weather time series (batch_size, sequence_length, features)
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Process images
        image_features = self.image_branch(images)
        
        # Process weather data
        # For simplicity, we're using a basic approach here
        # In practice, you'd use the actual LSTM output
        batch_size = weather_data.size(0)
        weather_features = torch.randn(batch_size, 64).to(images.device)
        
        # Fuse features
        combined = torch.cat([image_features, weather_features], dim=1)
        output = self.fusion(combined)
        
        return output

# Example usage
if __name__ == "__main__":
    # Test the fusion model
    batch_size = 4
    
    # Sample CNN features (from image processing)
    cnn_features = torch.randn(batch_size, 128)
    
    # Sample RNN features (from weather processing)
    rnn_features = torch.randn(batch_size, 64)
    
    # Initialize the fusion model
    fusion_model = FusionModel(cnn_feature_size=128, rnn_feature_size=64, num_classes=1)
    
    # Forward pass
    output = fusion_model(cnn_features, rnn_features)
    print(f"FusionModel output shape: {output.shape}")
    
    # Test early fusion model
    images = torch.randn(batch_size, 3, 256, 256)
    weather = torch.randn(batch_size, 30, 10)
    
    early_fusion = EarlyFusionModel(image_channels=3, weather_features=10)
    early_output = early_fusion(images, weather)
    print(f"EarlyFusionModel output shape: {early_output.shape}")