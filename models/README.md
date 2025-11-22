# Machine Learning Models

This directory contains the trained machine learning models for the Smart Agriculture Platform.

## Overview

The Smart Agriculture Platform uses various machine learning models to provide intelligent agricultural insights and predictions. These models are trained on domain-specific datasets and optimized for agricultural applications.

## Directory Structure

```
models/
├── crop_recommendation/       # Crop recommendation models
│   ├── crop_model.pkl
│   ├── crop_model_enhanced.pkl
│   ├── crop_preprocessor.pkl
│   ├── crop_preprocessor_enhanced.pkl
│   ├── fertilizer_model.pkl
│   ├── fertilizer_model_enhanced.pkl
│   ├── crop_model_metadata.json
│   └── fertilizer_model_metadata.json
├── crop_yield_prediction/     # Crop yield prediction models
│   ├── yield_model.pkl
│   ├── yield_model_enhanced.pkl
│   ├── preprocessor.pkl
│   ├── preprocessor_enhanced.pkl
│   ├── prediction_metadata.json
│   └── sample_explanation.txt
├── plant_disease/             # Plant disease detection models
│   └── plant_disease_model.pth
├── pest_weed/                 # Pest and weed detection models
│   └── pest_weed_detection_model.pth
├── cnn_model.py               # CNN base model architecture
├── rnn_model.py               # RNN base model architecture
├── fusion_model.py            # Model fusion techniques
├── model_training.py          # Generic model training utilities
└── README.md                  # This file
```

## Model Types

### 1. Crop Recommendation Models
- **Algorithm**: XGBoost Classifier
- **Purpose**: Recommend optimal crops based on soil and environmental conditions
- **Features**: Soil nutrients, weather patterns, regional factors
- **Output**: Top crop recommendations with confidence scores
- **Enhanced Version**: Includes detailed explanations and recommendations

### 2. Crop Yield Prediction Models
- **Algorithm**: RandomForest Regressor
- **Purpose**: Predict crop yields based on environmental and management factors
- **Features**: Rainfall, temperature, soil type, fertilizer usage, irrigation
- **Output**: Yield prediction in tons/hectare with confidence score
- **Enhanced Version**: Includes detailed explanations and recommendations

### 3. Plant Disease Detection Models
- **Algorithm**: ResNet18 (Transfer Learning)
- **Purpose**: Identify plant diseases from leaf images
- **Features**: Visual symptoms, leaf patterns, color variations
- **Output**: Disease classification with confidence score

### 4. Pest and Weed Detection Models
- **Algorithm**: ResNet18 (Transfer Learning)
- **Purpose**: Detect pests and weeds in agricultural fields
- **Features**: Visual identification, size, shape, color patterns
- **Output**: Pest/weed classification with confidence score

## Enhanced AI Systems

### Enhanced Yield Prediction System

The crop yield prediction system now includes an enhanced version with:
- Detailed insights generation
- Natural language explanations
- Actionable recommendations
- Risk assessment
- Benchmark comparisons

For more information, see [crop_yield_prediction/README_ENHANCED.md](file:///c%3A/Users/heman/OneDrive/Desktop/Agri/crop_yield_prediction/README_ENHANCED.md).

### Enhanced Crop and Fertilizer Recommendation System

The crop and fertilizer recommendation system now includes enhanced versions with:
- Detailed insights generation
- Natural language explanations
- Risk assessment
- Benchmark comparisons
- Confidence scoring

For more information, see [crop_recommendation/README_ENHANCED.md](file:///c%3A/Users/heman/OneDrive/Desktop/Agri/crop_recommendation/README_ENHANCED.md).

## Model Performance

### Crop Recommendation
- Accuracy: ~85-95%
- Precision: ~80-90%
- Recall: ~85-95%

### Crop Yield Prediction
- R² Score: ~0.7-0.9
- Mean Absolute Error: ~0.5-1.5 tons/hectare
- Mean Squared Error: ~0.5-2.0

### Plant Disease Detection
- Accuracy: ~85-95%
- Precision: ~80-90%
- Recall: ~85-95%

### Pest and Weed Detection
- Accuracy: ~80-90%
- Precision: ~75-85%
- Recall: ~80-90%

## Model Training

Models are trained using domain-specific datasets:
- Crop recommendation: Custom agricultural dataset
- Crop yield prediction: Historical yield and weather data
- Plant disease detection: PlantVillage dataset
- Pest and weed detection: WEED2C and IP102 datasets

Training is done using the enhanced training scripts:
- `crop_recommendation/training/train_crop_model_enhanced.py`
- `crop_recommendation/training/train_fertilizer_model_enhanced.py`
- `crop_yield_prediction/training/train_yield_model_enhanced.py`

## Model Deployment

Models are deployed through the Django REST API:
1. Models are loaded at application startup
2. Predictions are made through API endpoints
3. Results are returned in JSON format
4. Database storage for historical predictions

## Model Updates

Models should be retrained periodically:
- When new data becomes available
- When performance degrades
- When expanding to new regions or crops
- When adding new features or capabilities

## Integration with Smart Agriculture Platform

The models integrate with the broader platform through:
1. API endpoints for predictions
2. Shared data preprocessing utilities
3. Consistent model interfaces
4. Database storage for results

## Future Improvements

Planned enhancements include:
1. Integration with real-time satellite and weather data
2. Advanced deep learning architectures
3. Model explainability and interpretability
4. Automated model retraining pipelines
5. Enhanced LLM integration for natural language explanations
6. Edge deployment for offline inference

## License

This project is part of the Smart Agriculture Platform and follows the same license as the main project.