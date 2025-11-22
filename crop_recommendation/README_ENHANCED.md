# Enhanced Crop and Fertilizer Recommendation System

This enhanced system provides more detailed information and explanations for crop and fertilizer recommendations using machine learning models with LLM-like natural language generation.

## Features

- Enhanced machine learning models for crop and fertilizer recommendations
- Detailed insights and analysis of soil and environmental conditions
- Natural language explanations of recommendations
- Risk assessment and benchmark comparisons
- Confidence scoring for all predictions
- Database integration for tracking recommendations

## Enhanced Training Scripts

### Crop Recommendation ([train_crop_model_enhanced.py](file:///C:/Users/heman/OneDrive/Desktop/Agri/crop_recommendation/training/train_crop_model_enhanced.py))

The enhanced crop recommendation training script creates machine learning models with additional metadata and insights generation.

Key features:
- Trains both Random Forest and XGBoost models
- Selects the best performing model based on accuracy
- Generates detailed model insights including feature importance
- Creates natural language explanations for recommendations
- Saves model metadata for use in predictions

### Fertilizer Recommendation ([train_fertilizer_model_enhanced.py](file:///C:/Users/heman/OneDrive/Desktop/Agri/crop_recommendation/training/train_fertilizer_model_enhanced.py))

The enhanced fertilizer recommendation training script creates machine learning models specifically for fertilizer recommendations.

Key features:
- Trains Gradient Boosting models for fertilizer recommendations
- Generates detailed model insights
- Creates natural language explanations for fertilizer choices
- Saves model metadata for use in predictions

## Enhanced Prediction Scripts

### Crop Prediction ([predict_crop_enhanced.py](file:///C:/Users/heman/OneDrive/Desktop/Agri/crop_recommendation/predict_crop_enhanced.py))

The enhanced crop prediction script provides detailed insights and explanations for crop recommendations.

Key features:
- Loads enhanced models with fallback to regular models
- Generates detailed insights about the recommendation
- Creates natural language explanations using rule-based logic (placeholder for LLM)
- Provides risk assessment and benchmark comparisons
- Returns confidence scores for all predictions

### Fertilizer Prediction ([predict_fertilizer_enhanced.py](file:///C:/Users/heman/OneDrive/Desktop/Agri/crop_recommendation/predict_fertilizer_enhanced.py))

The enhanced fertilizer prediction script provides detailed insights and explanations for fertilizer recommendations.

Key features:
- Loads enhanced models with fallback to regular models
- Generates detailed insights about the fertilizer recommendation
- Creates natural language explanations using rule-based logic (placeholder for LLM)
- Provides risk assessment and benchmark comparisons
- Returns confidence scores for all predictions

## Enhanced API Endpoints

### Enhanced Recommendations ([recommendation_views_enhanced.py](file:///C:/Users/heman/OneDrive/Desktop/Agri/backend/api/recommendation_views_enhanced.py))

The enhanced API provides detailed recommendations with natural language explanations.

Endpoint: `POST /api/enhanced-recommendations/`

Request Body:
```json
{
    "soil_nitrogen": 120,
    "soil_phosphorus": 60,
    "soil_potassium": 80,
    "soil_ph": 6.5,
    "temperature": 25,
    "humidity": 70,
    "rainfall": 150,
    "location": "California",
    "season": "Spring"
}
```

Response:
```json
{
    "location": "California",
    "season": "Spring",
    "recommendation": {
        "rank": 1,
        "crop": "maize",
        "crop_confidence": 0.85,
        "fertilizer": "NPK 15-15-15",
        "fertilizer_confidence": 0.92,
        "quantity_kg_per_ha": 120,
        "explanation": "Based on the comprehensive analysis...",
        "crop_insights": {
            "prediction_summary": "Recommended crop: maize (confidence: 85.00%)",
            "key_factors": [
                "High nitrogen levels suitable for nitrogen-demanding crops",
                "Adequate phosphorus for root development",
                "Good potassium levels for disease resistance"
            ],
            "recommendations": [
                "Plant maize during the appropriate season for your region",
                "Monitor soil moisture and adjust irrigation as needed"
            ],
            "risk_assessment": "Low risk factors identified. Growing conditions appear favorable.",
            "comparison_to_benchmarks": "maize benchmarks: N(120-180), P(60-90), K(60-90), pH(5.8-7.0)",
            "confidence_level": "High (85.00%)"
        },
        "fertilizer_insights": {
            "prediction_summary": "Recommended fertilizer: NPK 15-15-15 (confidence: 92.00%)",
            "key_factors": [
                "Adequate nitrogen levels",
                "Adequate phosphorus levels",
                "Adequate potassium levels"
            ],
            "recommendations": [
                "Apply NPK 15-15-15 according to crop requirements and soil test results",
                "Consider split applications for better nutrient uptake efficiency"
            ],
            "risk_assessment": "Low risk factors identified. Conditions appear favorable for fertilizer application.",
            "comparison_to_benchmarks": "NPK 15-15-15 characteristics: n_content=15, p_content=15, k_content=15, best_for=Balanced nutrition",
            "confidence_level": "High (92.00%)"
        }
    }
}
```

## Testing

Test scripts are provided to verify the enhanced systems:

- [test_enhanced_crop.py](file:///C:/Users/heman/OneDrive/Desktop/Agri/crop_recommendation/test_enhanced_crop.py) - Test enhanced crop recommendation system
- [test_enhanced_fertilizer.py](file:///C:/Users/heman/OneDrive/Desktop/Agri/crop_recommendation/test_enhanced_fertilizer.py) - Test enhanced fertilizer recommendation system

Run tests with:
```bash
python crop_recommendation/test_enhanced_crop.py
python crop_recommendation/test_enhanced_fertilizer.py
```

## Training Models

To train the enhanced models:

```bash
python crop_recommendation/training/train_crop_model_enhanced.py
python crop_recommendation/training/train_fertilizer_model_enhanced.py
```

The enhanced models will be saved in the `crop_recommendation/saved_models/` directory with the suffix `_enhanced`.

## Integration with Existing System

The enhanced system is designed to work alongside the existing recommendation system. It automatically falls back to regular models if enhanced models are not available, ensuring backward compatibility.