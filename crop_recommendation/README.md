# Crop Recommendation System

This directory contains the machine learning models and scripts for crop and fertilizer recommendations.

## Directory Structure

- [models/](file:///C:/Users/heman/OneDrive/Desktop/Agri/crop_recommendation/models) - Machine learning model implementations
- [preprocessing/](file:///C:/Users/heman/OneDrive/Desktop/Agri/crop_recommendation/preprocessing) - Data preprocessing utilities
- [saved_models/](file:///C:/Users/heman/OneDrive/Desktop/Agri/crop_recommendation/saved_models) - Trained models and preprocessors
- [training/](file:///C:/Users/heman/OneDrive/Desktop/Agri/crop_recommendation/training) - Model training scripts

## Enhanced System

For more detailed information and explanations, see the [Enhanced System Documentation](README_ENHANCED.md).

## Models

The system includes implementations for:

1. Crop Recommendation Models
   - Random Forest
   - XGBoost
   - Neural Network

2. Fertilizer Recommendation Models
   - Gradient Boosting

## Training

To train the models:

```bash
python crop_recommendation/training/train_crop_model.py
python crop_recommendation/training/train_fertilizer_model.py
```

For enhanced models with detailed insights:

```bash
python crop_recommendation/training/train_crop_model_enhanced.py
python crop_recommendation/training/train_fertilizer_model_enhanced.py
```

## Usage

The models are used through the backend API endpoints in [backend/api/recommendation_views.py](file:///C:/Users/heman/OneDrive/Desktop/Agri/backend/api/recommendation_views.py).