# Market Prediction Module

## Overview
The Market Prediction Module provides AI-powered price forecasting for agricultural commodities. This module helps farmers make informed decisions about when to sell their crops by predicting future market prices based on various factors.

## Features
- **Price Prediction**: Predict future crop prices based on yield predictions, regional factors, and market conditions
- **Market Trend Analysis**: Determine if the market is bullish, bearish, or neutral for specific crops
- **Confidence Scoring**: Provide confidence levels for predictions
- **Historical Analysis**: Track price prediction history for crops

## Directory Structure
```
market_prediction/
├── preprocessing/           # Data preprocessing utilities
├── training/               # Model training scripts
└── README.md               # This file
```

## Components

### 1. Market Prediction Model
- Uses machine learning to predict crop prices
- Considers factors like yield predictions, regional conditions, global demand, and weather impact
- Provides confidence scores for predictions

### 2. Preprocessing Utilities
- Data preprocessing for market prediction features
- Categorical encoding and numerical scaling
- Single sample preprocessing for real-time predictions

### 3. Training Scripts
- Model training with sample data generation
- Performance evaluation metrics
- Model saving and loading utilities

## Usage

### Training the Model
To train the market prediction model:

```bash
cd market_prediction/training
python train_market_model.py
```

This will:
1. Generate sample training data
2. Train a Random Forest model
3. Evaluate model performance
4. Save the trained model and preprocessor

### Making Predictions
Market predictions can be made through the API endpoint:
```
POST /api/predict-market-price/
```

Required parameters:
- `crop`: Crop name
- `region`: Geographic region
- `season`: Growing season
- `yield_prediction`: Predicted yield in tons/hectare
- `global_demand`: Global demand level (low/medium/high)
- `weather_impact`: Weather impact on crop (poor/normal/excellent)

## API Endpoints
- `POST /api/predict-market-price/` - Predict market price for a crop
- `GET /api/market-predictions/` - Get recent market predictions
- `GET /api/market-predictions/crop/<crop_id>/` - Get market prediction history for a specific crop

## Model Performance
The current model uses Random Forest regression with the following performance metrics:
- RMSE: ~$25-35 per ton
- MAE: ~$20-30 per ton
- R²: ~0.85-0.92

## Future Improvements
- Integration with real market data feeds
- More sophisticated models (LSTM, XGBoost)
- Additional features (economic indicators, supply chain data)
- Real-time price updates