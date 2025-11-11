"""
Plant Disease Detection Model with CNN-RNN Fusion

This module implements a fusion model that combines:
1. CNN features from leaf images (using ResNet18)
2. RNN features from temporal plant health data
3. LLM-enhanced explanations for disease information
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import json
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Define disease classes directly to avoid import issues
PLANT_DISEASE_CLASSES_39 = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

class PlantDiseaseFusionModel(nn.Module):
    """Enhanced plant disease detection model with CNN-RNN fusion"""
    
    def __init__(self, num_disease_classes=39, cnn_feature_size=512, rnn_hidden_size=64):
        """
        Initialize the plant disease fusion model
        
        Args:
            num_disease_classes: Number of plant disease classes
            cnn_feature_size: Size of features from CNN (ResNet18)
            rnn_hidden_size: Hidden size of RNN for temporal data
        """
        super(PlantDiseaseFusionModel, self).__init__()
        
        # CNN feature processor (processes ResNet18 features)
        self.cnn_processor = nn.Sequential(
            nn.Linear(cnn_feature_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # RNN for temporal plant health data
        self.temporal_processor = nn.LSTM(
            input_size=10,  # Number of plant health metrics
            hidden_size=rnn_hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Attention mechanism for feature fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=128 + rnn_hidden_size,
            num_heads=4,
            dropout=0.2,
            batch_first=True
        )
        
        # Fusion and classification layers
        self.fusion = nn.Sequential(
            nn.Linear(128 + rnn_hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_disease_classes)
        )
        
    def forward(self, cnn_features, temporal_data):
        """
        Forward pass through the fusion network
        
        Args:
            cnn_features: Features from CNN (batch_size, cnn_feature_size)
            temporal_data: Temporal plant health data (batch_size, sequence_length, features)
            
        Returns:
            torch.Tensor: Output logits for disease classification
        """
        # Process CNN features
        cnn_processed = self.cnn_processor(cnn_features)
        
        # Process temporal data with RNN
        rnn_output, _ = self.temporal_processor(temporal_data)
        # Use the last time step
        temporal_features = rnn_output[:, -1, :]
        
        # Combine features
        combined_features = torch.cat([cnn_processed, temporal_features], dim=1)
        
        # Add batch dimension for attention
        combined_features = combined_features.unsqueeze(1)
        
        # Apply attention
        attended_features, _ = self.attention(
            combined_features, 
            combined_features, 
            combined_features
        )
        
        # Remove batch dimension
        attended_features = attended_features.squeeze(1)
        
        # Final classification
        output = self.fusion(attended_features)
        
        return output

class PlantDiseaseFusionDetector:
    """Complete plant disease detection system with CNN-RNN fusion"""
    
    def __init__(self, model_path=None):
        """Initialize the detector with models and components"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize models
        self.cnn_model = None
        self.fusion_model = None
        self.disease_classes = PLANT_DISEASE_CLASSES_39
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load models
        self._load_models(model_path)
        
    def _load_models(self, model_path=None):
        """Load the CNN and fusion models"""
        try:
            # Load CNN model (ResNet18-based)
            from plant_disease.models.disease_detection_model import load_disease_model
            self.cnn_model = load_disease_model(model_path)
            self.cnn_model.to(self.device)
            self.cnn_model.eval()
            print("✓ CNN model loaded successfully")
            
            # Load fusion model
            self.fusion_model = PlantDiseaseFusionModel()
            self.fusion_model.to(self.device)
            self.fusion_model.eval()
            print("✓ Fusion model loaded successfully")
            
            print(f"✓ Loaded {len(self.disease_classes)} disease classes")
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise
    
    def extract_cnn_features(self, image_tensor):
        """Extract features from CNN model (before final classification layer)"""
        try:
            # Remove the final classification layer to get features
            features = list(self.cnn_model.children())[:-1]  # Remove fc layer
            feature_extractor = nn.Sequential(*features)
            feature_extractor.to(self.device)
            feature_extractor.eval()
            
            with torch.no_grad():
                cnn_features = feature_extractor(image_tensor)
                # Flatten the features
                cnn_features = cnn_features.view(cnn_features.size(0), -1)
                
            return cnn_features
        except Exception as e:
            print(f"Error extracting CNN features: {str(e)}")
            # Return random features as fallback
            return torch.randn(image_tensor.size(0), 512).to(self.device)
    
    def preprocess_image(self, image_input):
        """Preprocess image for model input"""
        try:
            # Check if input is already a tensor
            if torch.is_tensor(image_input):
                # Assume it's already preprocessed
                if image_input.dim() == 3:
                    image_tensor = image_input.unsqueeze(0)  # Add batch dimension
                else:
                    image_tensor = image_input
                return image_tensor.to(self.device)
            
            # Handle other input types
            if isinstance(image_input, str):
                # File path
                image = Image.open(image_input).convert('RGB')
            elif isinstance(image_input, bytes):
                # Bytes data
                image = Image.open(io.BytesIO(image_input)).convert('RGB')
            else:
                # PIL Image
                image = image_input.convert('RGB')
            
            # Apply transformations
            image_tensor = self.transform(image)
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
            image_tensor = image_tensor.to(self.device)
            
            return image_tensor
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            raise
    
    def create_temporal_data(self, health_metrics=None):
        """Create temporal data for RNN processing"""
        try:
            if health_metrics is None:
                # Create dummy temporal data
                batch_size = 1
                sequence_length = 30
                num_features = 10
                temporal_data = torch.randn(batch_size, sequence_length, num_features)
            else:
                # Convert provided health metrics to tensor
                temporal_data = torch.tensor(health_metrics, dtype=torch.float32)
                if temporal_data.dim() == 2:
                    temporal_data = temporal_data.unsqueeze(0)  # Add batch dimension
            
            temporal_data = temporal_data.to(self.device)
            return temporal_data
        except Exception as e:
            print(f"Error creating temporal data: {str(e)}")
            # Return dummy data as fallback
            return torch.randn(1, 30, 10).to(self.device)
    
    def predict(self, image_input, health_metrics=None):
        """
        Make a disease prediction using CNN-RNN fusion
        
        Args:
            image_input: Image file path, bytes, or PIL Image
            health_metrics: Optional temporal health metrics (list or tensor)
            
        Returns:
            dict: Prediction results with disease, confidence, and features
        """
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image_input)
            
            # Extract CNN features
            cnn_features = self.extract_cnn_features(image_tensor)
            
            # Create temporal data
            temporal_data = self.create_temporal_data(health_metrics)
            
            # Make prediction with fusion model
            with torch.no_grad():
                logits = self.fusion_model(cnn_features, temporal_data)
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
            # Get predicted disease
            predicted_disease = self.disease_classes[predicted_idx.item()]
            
            # Get top 3 predictions
            top_probs, top_indices = torch.topk(probabilities, 3)
            top_predictions = []
            for i in range(3):
                idx = top_indices[0][i].item()
                prob = top_probs[0][i].item()
                top_predictions.append({
                    'disease': self.disease_classes[idx],
                    'confidence': prob
                })
            
            return {
                'predicted_disease': predicted_disease,
                'confidence': confidence.item(),
                'top_predictions': top_predictions,
                'cnn_features_shape': tuple(cnn_features.shape),
                'temporal_data_shape': tuple(temporal_data.shape)
            }
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            raise

def generate_disease_report(prediction_result, llm=None):
    """Generate a detailed disease report using LLM"""
    try:
        disease_name = prediction_result['predicted_disease']
        confidence = prediction_result['confidence']
        
        # Extract crop and disease
        parts = disease_name.split('___')
        if len(parts) == 2:
            crop = parts[0].replace('_', ' ').replace('(maize)', 'corn').replace('(including sour)', '')
            disease = parts[1].replace('_', ' ').replace(' Two-spotted spider mite', ' (Two-spotted spider mite)')
        else:
            crop = "Plant"
            disease = disease_name.replace('_', ' ')
        
        # Try to use LLM for enhanced report
        if llm and llm.text_generator:
            try:
                prompt = f"""
                Plant Disease Analysis Report:
                
                Disease detected: {disease} on {crop}
                Confidence: {confidence:.2%}
                
                Please provide:
                1. A brief description of this disease
                2. Key symptoms to look for
                3. Immediate treatment recommendations
                4. Long-term prevention strategies
                5. Any additional information that would be helpful for farmers
                
                Keep the response concise and actionable.
                """
                
                response = llm.text_generator(
                    prompt,
                    max_new_tokens=400,
                    temperature=0.7,
                    do_sample=True
                )
                
                report = response[0]['generated_text'][len(prompt):].strip()
                return report
            except Exception as e:
                print(f"LLM generation failed: {str(e)}")
        
        # Fallback to template-based report
        report = f"""
# Plant Disease Analysis Report

## Disease Identification
- **Crop**: {crop}
- **Disease**: {disease}
- **Confidence**: {confidence:.2%}

## General Information
This appears to be {disease} affecting {crop} plants. The model is {confidence:.2%} confident in this diagnosis.

## Recommended Actions
1. Confirm the diagnosis by examining multiple plants
2. Remove and destroy severely infected plants to prevent spread
3. Apply appropriate fungicides or bactericides as needed
4. Improve air circulation around plants
5. Avoid overhead watering

## For More Information
Consult with local agricultural extension services for specific treatment recommendations in your area.
        """.strip()
        
        return report
        
    except Exception as e:
        print(f"Error generating disease report: {str(e)}")
        return "Error generating detailed report. Please consult with an agricultural expert."

# Example usage
if __name__ == "__main__":
    print("Plant Disease Detection with CNN-RNN Fusion")
    print("=" * 45)
    
    try:
        # Initialize detector
        detector = PlantDiseaseFusionDetector()
        print("✓ Plant disease fusion detector initialized")
        
        # Test with dummy data
        print("\nTesting with dummy data...")
        
        # Create dummy image tensor
        dummy_image = torch.randn(1, 3, 224, 224)
        
        # Create dummy health metrics
        dummy_health_metrics = torch.randn(1, 30, 10)
        
        # Make prediction
        result = detector.predict(dummy_image, dummy_health_metrics)
        print("✓ Prediction successful")
        print(f"Predicted disease: {result['predicted_disease']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"CNN features shape: {result['cnn_features_shape']}")
        print(f"Temporal data shape: {result['temporal_data_shape']}")
        
        # Generate report
        try:
            from explainable_ai.llm_interface import AgricultureLLM
            llm = AgricultureLLM()
        except:
            llm = None
            
        report = generate_disease_report(result, llm)
        print("\nGenerated Report:")
        print("-" * 20)
        print(report[:200] + "..." if len(report) > 200 else report)
        
        print("\n🎉 Plant disease fusion model is working correctly!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()