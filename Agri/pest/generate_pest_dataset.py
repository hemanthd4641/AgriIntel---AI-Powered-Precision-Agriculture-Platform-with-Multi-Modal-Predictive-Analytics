"""
Script to generate comprehensive pest prediction dataset
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

def generate_comprehensive_pest_data(n_samples=10000):
    """
    Generate comprehensive pest data for training
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        pandas.DataFrame: Generated pest data
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Define crops
    crops = ['Wheat', 'Rice', 'Maize', 'Barley', 'Soybean', 'Cotton', 'Sugarcane', 'Potato', 'Tomato']
    
    # Define regions
    regions = ['North', 'South', 'East', 'West', 'Central', 'Northeast', 'Southeast', 'Northwest', 'Southwest']
    
    # Define seasons
    seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
    
    # Define common pests
    pests = [
        'Aphids', 'Armyworms', 'Boll Weevils', 'Corn Borers', 'Cutworms',
        'Flea Beetles', 'Grasshoppers', 'Hessian Fly', 'Japanese Beetles',
        'Leafhoppers', 'Root Knot Nematodes', 'Spider Mites', 'Stem Borers',
        'Thrips', 'Wireworms', 'White Grubs', 'Whiteflies'
    ]
    
    # Define weather conditions
    weather_conditions = ['Sunny', 'Cloudy', 'Rainy', 'Humid', 'Dry', 'Windy', 'Stormy']
    
    # Define soil types
    soil_types = ['Clay', 'Sandy', 'Loam', 'Silt', 'Peat', 'Chalky']
    
    # Define irrigation methods
    irrigation_methods = ['Drip', 'Sprinkler', 'Flood', 'Furrow', 'None']
    
    # Generate sample data
    data = []
    
    for _ in range(n_samples):
        # Randomly select factors
        crop = random.choice(crops)
        region = random.choice(regions)
        season = random.choice(seasons)
        pest = random.choice(pests)
        weather = random.choice(weather_conditions)
        soil_type = random.choice(soil_types)
        irrigation = random.choice(irrigation_methods)
        
        # Generate environmental factors
        temperature = round(np.random.uniform(15, 40), 1)  # Celsius
        humidity = round(np.random.uniform(30, 95), 1)  # Percentage
        rainfall = round(np.random.uniform(0, 150), 1)  # mm in last week
        wind_speed = round(np.random.uniform(0, 30), 1)  # km/h
        soil_moisture = round(np.random.uniform(10, 80), 1)  # Percentage
        soil_ph = round(np.random.uniform(5.0, 8.5), 1)  # pH level
        nitrogen = round(np.random.uniform(10, 200), 1)  # kg/ha
        phosphorus = round(np.random.uniform(5, 100), 1)  # kg/ha
        potassium = round(np.random.uniform(50, 300), 1)  # kg/ha
        previous_crop = random.choice(crops)
        days_since_planting = random.randint(10, 150)
        plant_density = random.randint(20000, 80000)  # plants per hectare
        
        # Generate pest probability based on factors (simplified model)
        # Base probability
        pest_probability = 0.1
        
        # Adjust based on temperature (optimal range 20-30°C)
        if 20 <= temperature <= 30:
            pest_probability += 0.2
        elif temperature > 35 or temperature < 10:
            pest_probability -= 0.1
            
        # Adjust based on humidity (pests prefer 60-80%)
        if 60 <= humidity <= 80:
            pest_probability += 0.25
        elif humidity < 40 or humidity > 90:
            pest_probability -= 0.1
            
        # Adjust based on rainfall (moderate rain increases pest activity)
        if 20 <= rainfall <= 80:
            pest_probability += 0.15
        elif rainfall > 100:
            pest_probability += 0.1  # Too much rain can wash away some pests
            
        # Adjust based on soil moisture (moderate moisture is ideal)
        if 40 <= soil_moisture <= 70:
            pest_probability += 0.2
        elif soil_moisture < 20 or soil_moisture > 85:
            pest_probability -= 0.1
            
        # Adjust based on crop type
        crop_risk = {
            'Wheat': 0.3, 'Rice': 0.5, 'Maize': 0.4, 'Barley': 0.25,
            'Soybean': 0.45, 'Cotton': 0.55, 'Sugarcane': 0.4, 
            'Potato': 0.6, 'Tomato': 0.65
        }
        pest_probability += crop_risk.get(crop, 0.4)
        
        # Adjust based on season
        season_factor = {'Spring': 0.2, 'Summer': 0.4, 'Autumn': 0.3, 'Winter': 0.1}
        pest_probability += season_factor.get(season, 0.25)
        
        # Adjust based on previous crop (some pests are host-specific)
        if previous_crop == crop:
            pest_probability += 0.15
            
        # Adjust based on plant density (higher density can harbor more pests)
        if plant_density > 60000:
            pest_probability += 0.1
        elif plant_density < 30000:
            pest_probability -= 0.05
            
        # Add some random noise
        pest_probability += np.random.normal(0, 0.1)
        
        # Ensure probability is between 0 and 1
        pest_probability = max(0, min(1, pest_probability))
        
        # Determine if pest is present based on probability
        pest_presence = 1 if np.random.random() < pest_probability else 0
        
        # Generate severity level (0-10) if pest is present
        severity = 0
        if pest_presence == 1:
            severity = max(1, min(10, int(np.random.normal(pest_probability * 10, 2))))
        
        # Generate recommended treatment based on severity
        treatment = 'None'
        if severity >= 8:
            treatment = 'Immediate pesticide application recommended'
        elif severity >= 5:
            treatment = 'Monitor closely and consider treatment'
        elif severity >= 2:
            treatment = 'Preventive measures recommended'
        
        # Generate timestamp
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2025, 12, 31)
        random_date = start_date + timedelta(
            seconds=random.randint(0, int((end_date - start_date).total_seconds()))
        )
        
        data.append({
            'date': random_date.strftime('%Y-%m-%d'),
            'crop': crop,
            'region': region,
            'season': season,
            'pest': pest,
            'temperature': temperature,
            'humidity': humidity,
            'rainfall': rainfall,
            'wind_speed': wind_speed,
            'soil_moisture': soil_moisture,
            'soil_ph': soil_ph,
            'soil_type': soil_type,
            'nitrogen': nitrogen,
            'phosphorus': phosphorus,
            'potassium': potassium,
            'weather_condition': weather,
            'irrigation_method': irrigation,
            'previous_crop': previous_crop,
            'days_since_planting': days_since_planting,
            'plant_density': plant_density,
            'pest_presence': pest_presence,
            'severity': severity,
            'recommended_treatment': treatment
        })
    
    return pd.DataFrame(data)

def save_dataset(df, output_file='pest_data.csv'):
    """
    Save the generated dataset to CSV file
    
    Args:
        df: DataFrame to save
        output_file: File path to save the dataset
    """
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Print basic statistics
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Print value counts for categorical variables
    print("\nValue Counts:")
    categorical_columns = ['crop', 'region', 'season', 'pest', 'weather_condition', 'soil_type', 'irrigation_method']
    for col in categorical_columns:
        print(f"\n{col}:")
        print(df[col].value_counts().head())

def main():
    """
    Main function to generate and save the pest dataset
    """
    print("Generating comprehensive pest prediction dataset...")
    
    # Generate dataset
    df = generate_comprehensive_pest_data(n_samples=12000)
    
    # Save dataset
    save_dataset(df, 'pest_data.csv')
    
    print("\nDataset generation completed successfully!")

# Example usage
if __name__ == "__main__":
    main()