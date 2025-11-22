# Plant Disease Detection Feature

This module extends the Smart Agriculture project with plant disease detection capabilities using computer vision and explainable AI.

## 🌿 Feature Overview

The plant disease detection feature allows farmers to:
- Upload photos of plant leaves
- Get predictions for plant diseases with confidence scores
- Receive natural language treatment advice using LLM + RAG
- View disease history for their farms
- Track disease patterns over time

## 📁 Directory Structure

```
plant_disease/
├── models/                 # Disease detection models
├── preprocessing/          # Image preprocessing utilities
├── datasets/               # Dataset handling utilities
├── training/               # Model training scripts
├── saved_models/           # Trained model weights
└── requirements.txt        # Python dependencies
```

## 🚀 Key Components

### 1. Disease Detection Model
- **Architecture**: ResNet50 with transfer learning
- **Classes**: 38 plant diseases from PlantVillage dataset
- **Input**: Leaf images (224x224 RGB)
- **Output**: Disease class + confidence score

### 2. Image Preprocessing
- **Resizing**: Standardize images to 224x224 pixels
- **Normalization**: ImageNet normalization
- **Augmentation**: Rotation, flipping, brightness/contrast adjustments

### 3. Explainable AI
- **LLM**: Integration with open-source language models
- **RAG**: Retrieval-Augmented Generation with FAISS
- **Knowledge Base**: Treatment guidelines and prevention strategies

### 4. API Endpoints
- `POST /api/disease/predict/` - Upload image and get disease prediction
- `GET /api/disease/history/<farm_id>/` - Get disease prediction history
- `POST /api/disease/advice/` - Get treatment advice for a disease
- `GET /api/disease/statistics/<farm_id>/` - Get disease statistics

## 🛠️ Installation

1. Install additional dependencies:
```bash
pip install -r plant_disease/requirements.txt
```

2. Download the PlantVillage dataset:
```bash
# Download from Kaggle: https://www.kaggle.com/emmarex/plantdisease
# Extract to: plant_disease/datasets/plantvillage/
```

3. Train the model (optional, pre-trained weights available):
```bash
python plant_disease/training/train_disease_model.py \
  --data_dir plant_disease/datasets/plantvillage/ \
  --model_save_path plant_disease/saved_models/plant_disease_model.pth
```

## 📊 Model Performance

The ResNet50-based model achieves:
- **Accuracy**: ~95% on PlantVillage test set
- **Training Time**: ~2 hours on Google Colab GPU
- **Inference Time**: <1 second per image

## 🌐 Frontend Integration

The feature is integrated into the existing web interface with:
- Dedicated "Disease Detection" section
- Image upload form
- Results display with confidence scores
- Treatment advice panel
- Disease history tracking

## 🧠 Explainable AI Components

### LLM Integration
- Uses Hugging Face models (GPT-2 by default)
- Generates natural language explanations
- Provides actionable treatment recommendations

### RAG System
- Knowledge base with disease treatment guidelines
- FAISS vector store for efficient retrieval
- Contextual advice based on disease and severity

## 🔧 API Usage Examples

### Disease Prediction
```javascript
const formData = new FormData();
formData.append('farm_id', 'farm123');
formData.append('image', imageFile);

fetch('/api/disease/predict/', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  console.log('Disease:', data.predicted_disease);
  console.log('Confidence:', data.confidence_score);
});
```

### Treatment Advice
```javascript
fetch('/api/disease/advice/', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    disease_name: 'Tomato Late Blight',
    severity: 'moderate'
  })
})
.then(response => response.json())
.then(data => {
  console.log('Advice:', data.advice);
});
```

## 📈 Future Enhancements

1. **Multi-language Support**: Advice in local languages
2. **Real-time Camera**: Mobile app integration
3. **Disease Progression**: Tracking disease spread over time
4. **Prevention Alerts**: Proactive disease prevention recommendations
5. **Integration with Weather**: Weather-disease correlation analysis

## 📚 References

- PlantVillage Dataset: https://www.kaggle.com/emmarex/plantdisease
- ResNet Paper: https://arxiv.org/abs/1512.03385
- FAISS: https://github.com/facebookresearch/faiss
- Hugging Face Transformers: https://huggingface.co/docs/transformers/