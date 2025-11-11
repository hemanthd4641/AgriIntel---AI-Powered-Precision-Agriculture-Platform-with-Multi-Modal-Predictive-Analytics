# Plant Disease Detection Models

This directory contains the implementation of the plant disease detection models for the Smart Agriculture Platform.

## Overview

The plant disease detection system uses deep learning models to identify common plant diseases from leaf images. This system is a critical component of precision agriculture, helping farmers identify and manage plant diseases that can significantly impact crop yields.

## Directory Structure

```
models/plant_disease/
├── __init__.py                                # Package initialization
├── demonstrate_disease_detection.py           # Demonstration script
├── demonstrate_32class_disease_detection.py   # 32-class demonstration script
├── inspect_model.py                           # Model inspection script
├── predict_disease.py                         # Prediction script
├── predict_32class_disease.py                 # 32-class prediction script
├── test_disease_model.py                      # Model testing script
└── CHANGES_SUMMARY.md                         # Summary of changes made
```

## Features

1. **Dynamic Class Handling**: Automatically adapts to models with different numbers of classes
2. **Model Information**: Retrieves detailed information about trained models
3. **Comprehensive Testing**: Includes test scripts to validate model functionality
4. **Model Inspection**: Provides detailed analysis of model structure
5. **Prediction Support**: Includes scripts for making predictions on new images
6. **32-Class Specialization**: Dedicated support for 32 common plant disease classes

## Usage

### Loading a Model

```python
from models.plant_disease.disease_detection_model import load_disease_model

# Load the model
model = load_disease_model()
```

### Making Predictions

```python
# For general models
from models.plant_disease.predict_disease import load_model, preprocess_image, predict_disease

# Load the model
model, num_classes = load_model()

# Preprocess an image
image_tensor = preprocess_image("path/to/image.jpg")

# Make a prediction
predicted_class, confidence, probabilities = predict_disease(model, image_tensor, num_classes)
```

```python
# For 32-class models
from models.plant_disease.predict_32class_disease import load_32class_model, preprocess_image, predict_32class_disease

# Load the 32-class model
model = load_32class_model()

# Preprocess an image
image_tensor = preprocess_image("path/to/image.jpg")

# Make a prediction
predictions = predict_32class_disease(model, image_tensor, top_k=3)
```

## Model Architecture

The plant disease detection system uses ResNet18 with transfer learning:
- Pre-trained weights from ImageNet
- Custom final layer for plant disease classification
- Fine-tuning of all layers for the specific task
- Dynamic adaptation to different numbers of classes

## Supported Model Types

1. **Binary Classifier (1 class)**: Detects presence/absence of disease
2. **Binary Classification (2 classes)**: Distinguishes between healthy and diseased plants
3. **Multi-class Classification (>2 classes)**: Identifies specific plant diseases
4. **32-Class Specialized Model**: Focused on 32 common plant disease classes for improved accuracy

## Integration with Smart Agriculture Platform

The plant disease detection system integrates with the broader Smart Agriculture Platform through:
1. API endpoints for disease detection
2. Shared preprocessing utilities
3. Consistent model interfaces

## Performance

With the full PlantVillage dataset (38 classes), the model typically achieves:
- Training accuracy: 90-95%
- Validation accuracy: 85-90%
- Test accuracy: 85-90%

The specialized 32-class model typically achieves:
- Training accuracy: 92-96%
- Validation accuracy: 88-92%
- Test accuracy: 87-91%

Results may vary based on:
- Training duration (more epochs = better results)
- Image resolution and quality
- Class balance in the dataset
- Hardware resources (GPU vs CPU)

## Future Improvements

Planned enhancements for the plant disease detection system include:
1. Support for object detection (bounding boxes) in addition to classification
2. Integration with drone and satellite imagery
3. Real-time detection capabilities
4. Expanded dataset support for more plant diseases
5. Additional specialized models for specific crop types

## License

This project is part of the Smart Agriculture Platform and follows the same license as the main project.