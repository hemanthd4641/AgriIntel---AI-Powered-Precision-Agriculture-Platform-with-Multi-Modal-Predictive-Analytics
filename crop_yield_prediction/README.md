# Crop Yield Prediction System

## Overview

The crop yield prediction system uses machine learning models to predict agricultural crop yields based on environmental and management factors. This system is a critical component of precision agriculture, helping farmers optimize their production practices and make informed decisions.

## Directory Structure

```
crop_yield_prediction/
├── models/                    # Trained models (created after training)
├── preprocessing/             # Data preprocessing utilities
│   └── yield_preprocessor.py  # Preprocessing functions
├── training/                  # Model training scripts
│   ├── train_yield_model.py   # Basic training script
│   └── train_yield_model_enhanced.py  # Enhanced training script with LLM integration
├── predict_yield_enhanced.py  # Enhanced prediction script with detailed explanations
├── README.md                  # This file
└── README_ENHANCED.md         # Documentation for enhanced features
```

## Features

1. **Machine Learning Models**: Uses RandomForestRegressor for yield prediction
2. **Data Preprocessing**: Handles categorical and numerical feature encoding
3. **Model Evaluation**: Provides MSE, MAE, and R² metrics
4. **Prediction API**: RESTful API endpoints for yield predictions
5. **Enhanced Explanations**: Detailed insights and recommendations (enhanced version)
6. **Risk Assessment**: Evaluation of potential challenges (enhanced version)
7. **Benchmark Comparisons**: Contextualization against typical yields (enhanced version)

## Supported Features

The system considers the following factors for yield prediction:
- Region
- Soil Type
- Crop Type
- Rainfall (mm)
- Temperature (Celsius)
- Fertilizer Usage
- Irrigation Usage
- Weather Conditions
- Days to Harvest

## Usage

### Training a Model

1. Navigate to the [training/](file:///c%3A/Users/heman/OneDrive/Desktop/Agri/crop_yield_prediction/training) directory
2. Run `python train_yield_model.py` for basic training
3. Run `python train_yield_model_enhanced.py` for enhanced training with detailed insights
4. The trained model will be saved to the [models/](file:///c%3A/Users/heman/OneDrive/Desktop/Agri/models) directory

### Making Predictions

1. Use the backend API endpoint `/api/predict-yield/` with required parameters
2. Or run `python predict_yield_enhanced.py` for interactive predictions

### Enhanced Features

For detailed information about the enhanced yield prediction system with LLM-generated explanations and recommendations, see [README_ENHANCED.md](file:///c%3A/Users/heman/OneDrive/Desktop/Agri/crop_yield_prediction/README_ENHANCED.md).

## Model Architecture

The crop yield prediction system uses RandomForestRegressor:
- Ensemble learning method for regression tasks
- Handles non-linear relationships between features
- Provides feature importance scores
- Robust to outliers in the data

## Performance

The model typically achieves good accuracy on crop yield prediction tasks:
- Mean Squared Error: ~0.5-2.0 (lower is better)
- Mean Absolute Error: ~0.5-1.5 tons/hectare
- R² Score: ~0.7-0.9 (higher is better)

Results may vary based on:
- Data quality and quantity
- Feature engineering
- Model hyperparameters
- Regional variations in growing conditions

## Integration with Smart Agriculture Platform

The crop yield prediction system integrates with the broader Smart Agriculture Platform through:
1. API endpoints for yield predictions
2. Shared preprocessing utilities
3. Consistent data formats
4. Database storage for historical predictions

## Future Improvements

Planned enhancements for the crop yield prediction system include:
1. Integration with real-time weather and satellite data
2. Advanced time-series forecasting models
3. Multi-crop rotation planning
4. Enhanced LLM integration for natural language explanations
5. Mobile application integration
6. Regional model specialization

## License

This project is part of the Smart Agriculture Platform and follows the same license as the main project.