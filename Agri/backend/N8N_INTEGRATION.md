# N8N Webhook Integration Summary

## Overview
All AgriIntel prediction features are now integrated with the n8n webhook to receive AI-powered suggestions and improvements after making predictions.

## Webhook Configuration
- **URL**: `https://projectu.app.n8n.cloud/webhook/agri-intel-chat`
- **Method**: POST
- **Timeout**: 30 seconds
- **Headers**: `Content-Type: application/json`

## Integration Flow

1. **User submits prediction request** → Feature endpoint receives data
2. **ML Model processes request** → Generates prediction using trained model
3. **Results sent to n8n webhook** → Sends prediction + input data to AI
4. **AI analyzes results** → n8n workflow provides suggestions and improvements
5. **Enhanced response returned** → User receives prediction + AI suggestions

## Integrated Features

### 1. ✅ Crop Yield Prediction
- **Endpoint**: `POST /api/predict-yield/`
- **Feature Name**: "Crop Yield Prediction"
- **Data Sent to n8n**:
  - Prediction: predicted_yield, confidence_score, explanation
  - Input: region, soil_type, crop, rainfall, temperature, fertilizer, irrigation, weather, days_to_harvest
- **AI Suggestions**: Analysis, improvement suggestions, best practices, risk mitigation

### 2. ✅ Pest Prediction
- **Endpoint**: `POST /api/predict-pest/`
- **Feature Name**: "Pest Prediction"
- **Data Sent to n8n**:
  - Prediction: predicted_pest, severity, confidence_score, pest_analysis, treatment
  - Input: crop, region, season, temperature, humidity, rainfall, soil data, nutrients
- **AI Suggestions**: Pest management advice, prevention strategies, treatment optimization

### 3. ✅ Plant Disease Detection
- **Endpoint**: `POST /api/disease/predict/`
- **Feature Name**: "Plant Disease Detection"
- **Data Sent to n8n**:
  - Prediction: predicted_disease, confidence_score, top_predictions, advice
  - Input: image_uploaded, image_name
- **AI Suggestions**: Disease treatment plans, preventive measures, organic alternatives

### 4. ✅ Market Price Prediction
- **Endpoint**: `POST /api/predict-market-price/`
- **Feature Name**: "Market Price Prediction"
- **Data Sent to n8n**:
  - Prediction: predicted_price, market_trend, confidence_score, market_intelligence
  - Input: crop, region, season, yield_prediction
- **AI Suggestions**: Market timing advice, price optimization, selling strategies

### 5. ✅ Crop & Fertilizer Recommendation
- **Endpoint**: `POST /api/recommendations/combined/`
- **Feature Name**: "Crop & Fertilizer Recommendation"
- **Data Sent to n8n**:
  - Prediction: top_recommendations (crop, fertilizer, quantity), additional_crops
  - Input: location, season, soil nutrients (N,P,K, pH), temperature, humidity, rainfall
- **AI Suggestions**: Crop rotation advice, soil improvement tips, sustainable farming practices

## Response Format

All integrated endpoints now return an enhanced response with AI suggestions:

```json
{
  // Original prediction results
  "predicted_yield": 5.23,
  "confidence_score": 0.9042,
  "explanation": "Based on conditions...",
  
  // NEW: AI-powered suggestions from n8n
  "ai_suggestions": "Based on your prediction of 5.23 tons/ha for Wheat...",
  "ai_enabled": true,
  
  // Other original fields...
  "timestamp": "2025-11-22T..."
}
```

## Implementation Details

### Helper Function: `send_to_n8n_webhook()`

Located at the top of `backend/api/views.py` (after imports)

```python
def send_to_n8n_webhook(feature_name, prediction_data, input_data):
    """
    Send prediction results to n8n webhook and get AI suggestions.
    
    Args:
        feature_name: Name of the feature
        prediction_data: The prediction results
        input_data: The input parameters
    
    Returns:
        dict: {
            'ai_suggestions': str,
            'webhook_success': bool
        }
    """
```

**Features**:
- Automatic formatting of prediction and input data
- 30-second timeout with error handling
- Graceful fallback if webhook is unavailable
- Detailed error messages for debugging
- Session ID generation for tracking

### Error Handling

The integration is designed to be resilient:

1. **Webhook Timeout**: Returns fallback message, prediction still works
2. **Webhook Error**: Returns error description, prediction still works  
3. **Network Issues**: Handles connection failures gracefully
4. **Invalid Response**: Extracts suggestions from various response formats

## Webhook Payload Structure

```json
{
  "message": "Feature: Crop Yield Prediction\n\nInput Parameters:...\n\nPrediction Results:...",
  "feature": "Crop Yield Prediction",
  "prediction": {
    "predicted_yield": 5.23,
    "confidence_score": 0.9042
  },
  "input": {
    "region": "North",
    "crop": "Wheat",
    ...
  },
  "timestamp": "2025-11-22T10:30:00.000Z",
  "sessionId": "agriintel_crop_yield_prediction"
}
```

## Testing the Integration

### Test Individual Feature

```python
import requests

# Test Crop Yield Prediction with n8n integration
response = requests.post(
    'http://localhost:8000/api/predict-yield/',
    json={
        'region': 'North',
        'soil_type': 'Loamy',
        'crop': 'Wheat',
        'rainfall_mm': 800,
        'temperature_celsius': 25,
        'fertilizer_used': True,
        'irrigation_used': True,
        'weather_condition': 'Sunny',
        'days_to_harvest': 120
    }
)

result = response.json()
print("Prediction:", result['predicted_yield'])
print("AI Suggestions:", result['ai_suggestions'])
print("AI Enabled:", result['ai_enabled'])
```

### Check Webhook Status

The console logs show webhook status:
```
Sending Crop Yield Prediction data to n8n webhook...
Received AI suggestions from n8n for Crop Yield Prediction
```

## Benefits

1. **Enhanced Insights**: AI-powered analysis beyond ML model predictions
2. **Contextual Advice**: Suggestions tailored to specific conditions
3. **Best Practices**: Expert recommendations from AI knowledge base
4. **Risk Mitigation**: Proactive identification of potential issues
5. **Continuous Improvement**: AI learns from user interactions
6. **No Disruption**: Fallback ensures predictions work even if webhook is down

## Configuration

To change the webhook URL, update the constant in `backend/api/views.py`:

```python
N8N_WEBHOOK_URL = "https://projectu.app.n8n.cloud/webhook/agri-intel-chat"
```

## Status

✅ **All major prediction features integrated**
✅ **Error handling implemented**
✅ **Graceful fallback on webhook failure**
✅ **Response format includes AI suggestions**
✅ **Session tracking enabled**
✅ **Timeout protection (30s)**

## Notes

- The webhook integration adds AI suggestions without affecting core ML predictions
- If the n8n webhook is unavailable, predictions still work normally
- AI suggestions are added to all response payloads with `ai_suggestions` and `ai_enabled` fields
- Each feature has a unique session ID for tracking in n8n
- Webhook timeout is set to 30 seconds to prevent long waits
