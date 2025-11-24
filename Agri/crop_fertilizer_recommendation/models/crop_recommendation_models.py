"""
Crop recommendation models using various ML algorithms.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import os


class CropRecommendationModel:
    """Base class for crop recommendation models."""
    
    def __init__(self):
        self.model = None
        self.model_type = None
    
    def train(self, X_train, y_train):
        """Train the model."""
        raise NotImplementedError
    
    def predict(self, X):
        """Make predictions."""
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model.predict(X), self.model.predict_proba(X)
    
    def save_model(self, file_path):
        """Save the trained model."""
        if self.model is not None:
            joblib.dump(self.model, file_path)
        else:
            raise ValueError("Model not trained yet.")
    
    def load_model(self, file_path):
        """Load a trained model."""
        if os.path.exists(file_path):
            self.model = joblib.load(file_path)
        else:
            raise FileNotFoundError(f"Model file {file_path} not found.")


class CropRecommendationRandomForest(CropRecommendationModel):
    """Random Forest classifier for crop recommendation."""
    
    def __init__(self, n_estimators=100, random_state=42):
        super().__init__()
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        self.model_type = "RandomForest"
        
    def train(self, X_train, y_train):
        """Train the Random Forest model."""
        self.model.fit(X_train, y_train)
        
    def predict(self, X):
        """Predict crop recommendations."""
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        return predictions, probabilities
    
    def get_feature_importance(self):
        """Get feature importance scores."""
        return self.model.feature_importances_


class CropRecommendationXGBoost(CropRecommendationModel):
    """XGBoost classifier for crop recommendation."""
    
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1):
        super().__init__()
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42
        )
        self.model_type = "XGBoost"
        
    def train(self, X_train, y_train):
        """Train the XGBoost model."""
        self.model.fit(X_train, y_train)
        
    def predict(self, X):
        """Predict crop recommendations."""
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        return predictions, probabilities


class CropRecommendationNN(nn.Module):
    """Neural Network for crop recommendation."""
    
    def __init__(self, input_size, num_classes, hidden_sizes=[128, 64, 32]):
        super(CropRecommendationNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
            
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class CropRecommendationNeuralNet:
    """Neural Network wrapper for crop recommendation."""
    
    def __init__(self, input_size, num_classes):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CropRecommendationNN(input_size, num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.model_type = "NeuralNetwork"
        
    def train(self, X_train, y_train, epochs=100):
        """Train the neural network model."""
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_train_tensor)
            loss = self.criterion(outputs, y_train_tensor)
            loss.backward()
            self.optimizer.step()
            
    def predict(self, X):
        """Predict crop recommendations."""
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_tensor = torch.FloatTensor(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            return predictions.cpu().numpy(), probabilities.cpu().numpy()


class FertilizerRecommendationModel:
    """Gradient Boosting classifier for fertilizer recommendation."""
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
        self.model_type = "GradientBoosting"
        
    def train(self, X_train, y_train):
        """Train the fertilizer recommendation model."""
        self.model.fit(X_train, y_train)
        
    def predict(self, X):
        """Predict fertilizer recommendations."""
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        return predictions, probabilities


class FertilizerRuleBased:
    """Rule-based approach for fertilizer recommendation."""
    
    def __init__(self):
        # Define rules based on soil nutrient levels and crop requirements
        self.rules = {
            'wheat': {'N': (80, 120), 'P': (40, 60), 'K': (40, 60)},
            'rice': {'N': (100, 150), 'P': (30, 50), 'K': (30, 50)},
            'maize': {'N': (120, 180), 'P': (60, 90), 'K': (60, 90)},
            'cotton': {'N': (100, 150), 'P': (40, 60), 'K': (80, 120)},
            'sugarcane': {'N': (150, 250), 'P': (50, 80), 'K': (150, 250)},
            # Add more crops as needed
        }
        
    def recommend(self, crop, soil_n, soil_p, soil_k):
        """
        Recommend fertilizer based on crop and soil nutrient levels.
        
        Args:
            crop (str): Crop name
            soil_n (float): Soil nitrogen level
            soil_p (float): Soil phosphorus level
            soil_k (float): Soil potassium level
            
        Returns:
            list: List of fertilizer recommendations
        """
        if crop not in self.rules:
            return ["Unknown crop. General recommendation: Apply balanced NPK fertilizer."]
            
        requirements = self.rules[crop]
        recommendations = []
        
        # Nitrogen recommendation
        if soil_n < requirements['N'][0]:
            n_deficit = requirements['N'][0] - soil_n
            recommendations.append(f"Apply {n_deficit:.1f} kg/ha of nitrogen fertilizer")
            
        # Phosphorus recommendation
        if soil_p < requirements['P'][0]:
            p_deficit = requirements['P'][0] - soil_p
            recommendations.append(f"Apply {p_deficit:.1f} kg/ha of phosphorus fertilizer")
            
        # Potassium recommendation
        if soil_k < requirements['K'][0]:
            k_deficit = requirements['K'][0] - soil_k
            recommendations.append(f"Apply {k_deficit:.1f} kg/ha of potassium fertilizer")
            
        return recommendations if recommendations else ["Soil nutrient levels are adequate for this crop"]