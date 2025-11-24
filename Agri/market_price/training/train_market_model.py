"""
Training script for market prediction model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from market_price.preprocessing.market_preprocessor import MarketPreprocessor

def load_real_market_data():
    """
    Load real market data from the dataset
    """
    try:
        # Define dataset path
        dataset_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'market_price',
            'market_data.csv'
        )
        
        # Check if file exists in the expected location
        if not os.path.exists(dataset_path):
            # Try alternative path (same directory as this script)
            dataset_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'market_data.csv'
            )
        
        # Load data
        if os.path.exists(dataset_path):
            data = pd.read_csv(dataset_path)
            print(f"Loaded real market data with {len(data)} samples")
            return data
        else:
            print(f"Dataset not found at {dataset_path}")
            return None
    except Exception as e:
        print(f"Error loading real market data: {str(e)}")
        return None

def generate_sample_data():
    """
    Generate sample market data for training
    In a real implementation, this would load actual historical data
    """
    np.random.seed(42)
    
    # Sample crops
    crops = ['Wheat', 'Rice', 'Maize', 'Barley', 'Soybean', 'Cotton', 'Sugarcane', 'Potato', 'Tomato']
    regions = ['North', 'South', 'East', 'West', 'Central']
    seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
    global_demands = ['low', 'medium', 'high']
    weather_impacts = ['poor', 'normal', 'excellent']
    economic_conditions = ['recession', 'stable', 'growth']
    fertilizer_usages = ['low', 'medium', 'high']
    irrigation_usages = ['low', 'medium', 'high']
    
    # Generate sample data
    n_samples = 1000
    data = []
    
    for _ in range(n_samples):
        crop = np.random.choice(crops)
        region = np.random.choice(regions)
        season = np.random.choice(seasons)
        yield_prediction = np.random.uniform(1.0, 5.0)  # tons/ha
        global_demand = np.random.choice(global_demands)
        weather_impact = np.random.choice(weather_impacts)
        economic_condition = np.random.choice(economic_conditions)
        supply_index = np.random.uniform(30, 90)
        demand_index = np.random.uniform(30, 90)
        inventory_level = np.random.uniform(10, 80)
        export_demand = np.random.uniform(20, 90)
        production_cost = np.random.uniform(50, 500)
        days_to_harvest = np.random.randint(30, 180)
        fertilizer_usage = np.random.choice(fertilizer_usages)
        irrigation_usage = np.random.choice(irrigation_usages)
        
        # Generate price based on factors (simplified model)
        base_price = {
            'Wheat': 250, 'Rice': 400, 'Maize': 200, 'Barley': 220,
            'Soybean': 450, 'Cotton': 1800, 'Sugarcane': 50, 
            'Potato': 300, 'Tomato': 200
        }.get(crop, 300)
        
        # Adjust based on factors
        yield_factor = 1.0
        if yield_prediction < 2.0:
            yield_factor = 1.3
        elif yield_prediction > 4.0:
            yield_factor = 0.8
            
        demand_factor = {'low': 0.8, 'medium': 1.0, 'high': 1.3}.get(global_demand, 1.0)
        weather_factor = {'poor': 1.4, 'normal': 1.0, 'excellent': 0.9}.get(weather_impact, 1.0)
        economic_factor = {'recession': 0.8, 'stable': 1.0, 'growth': 1.2}.get(economic_condition, 1.0)
        
        # Supply-demand balance factor
        supply_demand_ratio = supply_index / demand_index
        supply_demand_factor = 1.0
        if supply_demand_ratio < 0.8:
            supply_demand_factor = 1.3
        elif supply_demand_ratio > 1.2:
            supply_demand_factor = 0.8
            
        # Inventory factor
        inventory_factor = 1.0
        if inventory_level < 30:
            inventory_factor = 1.2
        elif inventory_level > 70:
            inventory_factor = 0.9
            
        # Export demand factor
        export_factor = 1.0
        if export_demand > 70:
            export_factor = 1.3
        elif export_demand < 40:
            export_factor = 0.8
            
        price = (base_price * yield_factor * demand_factor * weather_factor * 
                economic_factor * supply_demand_factor * inventory_factor * export_factor)
        
        # Add some noise
        price *= np.random.normal(1.0, 0.1)
        
        data.append({
            'date': '2023-01-01',
            'crop': crop,
            'region': region,
            'season': season,
            'yield_prediction': yield_prediction,
            'global_demand': global_demand,
            'weather_impact': weather_impact,
            'economic_condition': economic_condition,
            'supply_index': supply_index,
            'demand_index': demand_index,
            'inventory_level': inventory_level,
            'export_demand': export_demand,
            'production_cost': production_cost,
            'days_to_harvest': days_to_harvest,
            'fertilizer_usage': fertilizer_usage,
            'irrigation_usage': irrigation_usage,
            'price_per_ton': round(price, 2)
        })
    
    return pd.DataFrame(data)

def preprocess_real_data(data):
    """
    Preprocess real market data for training
    
    Args:
        data: Raw market data DataFrame
        
    Returns:
        pandas.DataFrame: Preprocessed data
    """
    # Make a copy to avoid modifying original data
    df = data.copy()
    
    # Convert date to datetime if it exists
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        # Extract useful date features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        # Drop original date column
        df = df.drop('date', axis=1)
    
    return df

def train_market_model():
    """
    Train the market prediction model
    """
    print("=== TRAINING MARKET PREDICTION MODEL ===")
    
    # Try to load real data first, fall back to sample data
    print("Loading market data...")
    data = load_real_market_data()
    
    if data is None:
        print("Generating sample training data...")
        data = generate_sample_data()
    else:
        print("Using real market data for training")
        # Preprocess real data
        data = preprocess_real_data(data)
    
    print(f"Dataset shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    
    # Create preprocessor
    preprocessor = MarketPreprocessor()
    
    # Preprocess data
    print("Preprocessing data...")
    X, y = preprocessor.preprocess_data(data, target_column='price_per_ton')
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape if y is not None else 'None'}")
    
    # Split data
    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train model
    print("Training Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=200,  # Increased number of trees
        random_state=42, 
        n_jobs=-1,
        max_depth=20,  # Limit depth to prevent overfitting
        min_samples_split=5,
        min_samples_leaf=2
    )
    model.fit(X_train, y_train)
    print("Model training completed")
    
    # Evaluate model
    print("Evaluating model performance...")
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  R²: {r2:.4f}")
    
    # Feature importance
    if hasattr(preprocessor, 'feature_columns') and preprocessor.feature_columns:
        feature_importance = pd.DataFrame({
            'feature': preprocessor.feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
    
    # Save model and preprocessor
    print("Saving model and preprocessor...")
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    market_models_dir = os.path.join(models_dir, 'market_prediction')
    if not os.path.exists(market_models_dir):
        os.makedirs(market_models_dir)
    
    model_path = os.path.join(market_models_dir, 'market_model.pkl')
    preprocessor_path = os.path.join(market_models_dir, 'preprocessor.pkl')
    
    joblib.dump(model, model_path)
    preprocessor.save_preprocessor(preprocessor_path)
    
    print(f"Model saved to: {model_path}")
    print(f"Preprocessor saved to: {preprocessor_path}")
    
    # Test prediction with a sample
    print("\nTesting prediction with sample data...")
    sample_data = {
        'crop': 'Wheat',
        'region': 'North',
        'season': 'Summer',
        'yield_prediction': 3.5,
        'global_demand': 'medium',
        'weather_impact': 'normal',
        'economic_condition': 'stable',
        'supply_index': 60.0,
        'demand_index': 60.0,
        'inventory_level': 50.0,
        'export_demand': 60.0,
        'production_cost': 150.0,
        'days_to_harvest': 90,
        'fertilizer_usage': 'medium',
        'irrigation_usage': 'medium',
        'year': 2023,
        'month': 6,
        'day_of_year': 151
    }
    
    # Preprocess sample
    X_sample = preprocessor.preprocess_single_sample(sample_data)
    
    # Make prediction
    predicted_price = model.predict(X_sample)[0]
    print(f"Predicted price for sample: ${predicted_price:.2f} per ton")
    
    print("\n=== TRAINING COMPLETED SUCCESSFULLY ===")
    return model, preprocessor

# Example usage
if __name__ == "__main__":
    train_market_model()