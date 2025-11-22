"""
Script to generate comprehensive market prediction dataset
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

def generate_comprehensive_market_data(n_samples=10000):
    """
    Generate comprehensive market data for training
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        pandas.DataFrame: Generated market data
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Define crops and their base prices
    crops = {
        'Wheat': {'base_price': 250, 'volatility': 0.15},
        'Rice': {'base_price': 400, 'volatility': 0.20},
        'Maize': {'base_price': 200, 'volatility': 0.18},
        'Barley': {'base_price': 220, 'volatility': 0.16},
        'Soybean': {'base_price': 450, 'volatility': 0.22},
        'Cotton': {'base_price': 1800, 'volatility': 0.25},
        'Sugarcane': {'base_price': 50, 'volatility': 0.12},
        'Potato': {'base_price': 300, 'volatility': 0.14},
        'Tomato': {'base_price': 200, 'volatility': 0.18},
        'Coffee': {'base_price': 2500, 'volatility': 0.30},
        'Tea': {'base_price': 3000, 'volatility': 0.28}
    }
    
    # Define regions
    regions = ['North', 'South', 'East', 'West', 'Central', 'Northeast', 'Southeast', 'Northwest', 'Southwest']
    
    # Define seasons
    seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
    
    # Define global demand levels
    global_demands = ['low', 'medium', 'high']
    
    # Define weather impacts
    weather_impacts = ['poor', 'normal', 'excellent']
    
    # Define economic indicators
    economic_conditions = ['recession', 'stable', 'growth']
    
    # Generate sample data
    data = []
    
    for _ in range(n_samples):
        # Randomly select crop
        crop = random.choice(list(crops.keys()))
        crop_info = crops[crop]
        
        # Randomly select other factors
        region = random.choice(regions)
        season = random.choice(seasons)
        global_demand = random.choice(global_demands)
        weather_impact = random.choice(weather_impacts)
        economic_condition = random.choice(economic_conditions)
        
        # Generate yield prediction (tons/ha)
        yield_prediction = np.random.uniform(1.0, 8.0)
        
        # Generate supply index (0-100)
        supply_index = np.random.uniform(30, 90)
        
        # Generate demand index (0-100)
        demand_index = np.random.uniform(30, 90)
        
        # Generate inventory levels (0-100)
        inventory_level = np.random.uniform(10, 80)
        
        # Generate export demand (0-100)
        export_demand = np.random.uniform(20, 90)
        
        # Generate production cost per ton
        production_cost = np.random.uniform(crop_info['base_price'] * 0.3, crop_info['base_price'] * 0.8)
        
        # Generate price based on multiple factors
        base_price = crop_info['base_price']
        
        # Adjust based on yield prediction
        yield_factor = 1.0
        if yield_prediction < 2.0:
            yield_factor = 1.5  # Low yield increases price
        elif yield_prediction > 6.0:
            yield_factor = 0.7  # High yield decreases price
            
        # Adjust based on global demand
        demand_factor = {'low': 0.7, 'medium': 1.0, 'high': 1.4}.get(global_demand, 1.0)
        
        # Adjust based on weather impact
        weather_factor = {'poor': 1.6, 'normal': 1.0, 'excellent': 0.8}.get(weather_impact, 1.0)
        
        # Adjust based on economic condition
        economic_factor = {'recession': 0.8, 'stable': 1.0, 'growth': 1.2}.get(economic_condition, 1.0)
        
        # Adjust based on supply/demand balance
        supply_demand_ratio = supply_index / demand_index
        supply_demand_factor = 1.0
        if supply_demand_ratio < 0.8:
            supply_demand_factor = 1.3  # Low supply relative to demand increases price
        elif supply_demand_ratio > 1.2:
            supply_demand_factor = 0.8  # High supply relative to demand decreases price
            
        # Adjust based on inventory levels
        inventory_factor = 1.0
        if inventory_level < 30:
            inventory_factor = 1.2  # Low inventory increases price
        elif inventory_level > 70:
            inventory_factor = 0.9  # High inventory decreases price
            
        # Adjust based on export demand
        export_factor = 1.0
        if export_demand > 70:
            export_factor = 1.3  # High export demand increases price
        elif export_demand < 40:
            export_factor = 0.8  # Low export demand decreases price
            
        # Calculate final price
        price = (base_price * yield_factor * demand_factor * weather_factor * 
                economic_factor * supply_demand_factor * inventory_factor * export_factor)
        
        # Add some random noise
        noise = np.random.normal(1.0, crop_info['volatility'] * 0.3)
        price *= noise
        
        # Ensure price is positive
        price = max(price, base_price * 0.3)
        
        # Generate additional features
        days_to_harvest = random.randint(30, 180)
        fertilizer_usage = random.choice(['low', 'medium', 'high'])
        irrigation_usage = random.choice(['low', 'medium', 'high'])
        
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
            'yield_prediction': round(yield_prediction, 2),
            'global_demand': global_demand,
            'weather_impact': weather_impact,
            'economic_condition': economic_condition,
            'supply_index': round(supply_index, 2),
            'demand_index': round(demand_index, 2),
            'inventory_level': round(inventory_level, 2),
            'export_demand': round(export_demand, 2),
            'production_cost': round(production_cost, 2),
            'days_to_harvest': days_to_harvest,
            'fertilizer_usage': fertilizer_usage,
            'irrigation_usage': irrigation_usage,
            'price_per_ton': round(price, 2)
        })
    
    return pd.DataFrame(data)

def save_dataset(df, output_file='market_data.csv'):
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
    categorical_columns = ['crop', 'region', 'season', 'global_demand', 'weather_impact', 'economic_condition']
    for col in categorical_columns:
        print(f"\n{col}:")
        print(df[col].value_counts().head())

def main():
    """
    Main function to generate and save the market dataset
    """
    print("Generating comprehensive market prediction dataset...")
    
    # Generate dataset
    df = generate_comprehensive_market_data(n_samples=15000)
    
    # Save dataset
    save_dataset(df, 'market_data.csv')
    
    print("\nDataset generation completed successfully!")

# Example usage
if __name__ == "__main__":
    main()