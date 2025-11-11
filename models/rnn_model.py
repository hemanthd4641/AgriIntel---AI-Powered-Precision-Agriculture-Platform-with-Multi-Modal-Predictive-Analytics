"""
RNN Model for Weather Data Processing in Smart Agriculture Project

This module implements a Recurrent Neural Network (LSTM) for processing
time-series weather data to extract temporal features for crop yield prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class WeatherLSTM(nn.Module):
    """LSTM model for weather time-series feature extraction"""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=1, dropout_rate=0.3):
        """
        Initialize the LSTM model
        
        Args:
            input_size: Number of features in the input (weather parameters)
            hidden_size: Number of features in the hidden state
            num_layers: Number of recurrent layers
            num_classes: Number of output classes (1 for regression)
            dropout_rate: Dropout rate for regularization
        """
        super(WeatherLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Use the last time step output
        x = lstm_out[:, -1, :]
        
        # Fully connected layers with ReLU activation and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Output layer
        x = self.fc3(x)
        
        return x

class WeatherGRU(nn.Module):
    """GRU model for weather time-series feature extraction (alternative to LSTM)"""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=1, dropout_rate=0.3):
        """
        Initialize the GRU model
        
        Args:
            input_size: Number of features in the input (weather parameters)
            hidden_size: Number of features in the hidden state
            num_layers: Number of recurrent layers
            num_classes: Number of output classes (1 for regression)
            dropout_rate: Dropout rate for regularization
        """
        super(WeatherGRU, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # GRU forward pass
        gru_out, _ = self.gru(x, h0)
        
        # Use the last time step output
        x = gru_out[:, -1, :]
        
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
    # Test the LSTM model
    # Create a sample input tensor (batch_size=4, sequence_length=30, features=10)
    sample_input = torch.randn(4, 30, 10)  # 30 days of weather data with 10 parameters
    
    # Initialize the LSTM model
    lstm_model = WeatherLSTM(input_size=10, hidden_size=64, num_layers=2, num_classes=1)
    
    # Forward pass
    lstm_output = lstm_model(sample_input)
    print(f"WeatherLSTM output shape: {lstm_output.shape}")
    
    # Test the GRU model
    gru_model = WeatherGRU(input_size=10, hidden_size=64, num_layers=2, num_classes=1)
    gru_output = gru_model(sample_input)
    print(f"WeatherGRU output shape: {gru_output.shape}")