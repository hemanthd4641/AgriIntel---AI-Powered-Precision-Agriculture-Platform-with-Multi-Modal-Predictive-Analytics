"""
Data preprocessing for market prediction dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

class MarketPreprocessor:
    """Preprocessor for crop market prediction data"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def load_data(self, file_path):
        """
        Load market prediction data from CSV file
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pandas.DataFrame: Loaded data
        """
        try:
            data = pd.read_csv(file_path)
            print(f"Successfully loaded data with {len(data)} rows and {len(data.columns)} columns")
            return data
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    def preprocess_data(self, data, target_column='price_per_ton'):
        """
        Preprocess the market prediction data
        
        Args:
            data (pandas.DataFrame): Input data
            target_column (str): Name of the target column
            
        Returns:
            tuple: (X, y) - Processed features and target
        """
        # Separate features and target
        feature_columns = [col for col in data.columns if col != target_column]
        self.feature_columns = feature_columns
        
        X = data[feature_columns].copy()
        y = data[target_column] if target_column in data.columns else None
        
        # Handle categorical features
        categorical_columns = ['crop', 'region', 'season', 'global_demand', 'weather_impact', 'economic_condition', 'fertilizer_usage', 'irrigation_usage']
        numerical_columns = [col for col in feature_columns if col not in categorical_columns]
        
        # Encode categorical variables
        for col in categorical_columns:
            if col in X.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
                else:
                    X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        # Scale numerical features
        X[numerical_columns] = self.scaler.fit_transform(X[numerical_columns])
        
        return X, y
    
    def save_preprocessor(self, file_path):
        """
        Save the preprocessor to disk
        
        Args:
            file_path (str): Path to save the preprocessor
        """
        joblib.dump({
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }, file_path)
        print(f"Preprocessor saved to {file_path}")
    
    def load_preprocessor(self, file_path):
        """
        Load the preprocessor from disk
        
        Args:
            file_path (str): Path to load the preprocessor from
        """
        try:
            data = joblib.load(file_path)
            self.label_encoders = data['label_encoders']
            self.scaler = data['scaler']
            self.feature_columns = data['feature_columns']
            print(f"Preprocessor loaded from {file_path}")
        except Exception as e:
            print(f"Error loading preprocessor: {str(e)}")
    
    def preprocess_single_sample(self, sample_data):
        """
        Preprocess a single sample for prediction
        
        Args:
            sample_data (dict): Dictionary with feature names as keys and values as values
            
        Returns:
            numpy.ndarray: Preprocessed feature vector
        """
        # Convert to DataFrame
        df = pd.DataFrame([sample_data])
        
        # Apply the same preprocessing steps
        categorical_columns = ['crop', 'region', 'season', 'global_demand', 'weather_impact', 'economic_condition', 'fertilizer_usage', 'irrigation_usage']
        numerical_columns = [col for col in self.feature_columns if col not in categorical_columns]
        
        # Encode categorical variables
        for col in categorical_columns:
            if col in df.columns and col in self.label_encoders:
                try:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
                except ValueError as e:
                    print(f"Warning: Unknown category in {col}, using default value")
                    df[col] = 0  # Default to first category
        
        # Scale numerical features
        df[numerical_columns] = self.scaler.transform(df[numerical_columns])
        
        return df[self.feature_columns].values

# Example usage
if __name__ == "__main__":
    # Create preprocessor
    preprocessor = MarketPreprocessor()
    
    # Example of how to use the preprocessor
    print("Market prediction preprocessor ready for use")