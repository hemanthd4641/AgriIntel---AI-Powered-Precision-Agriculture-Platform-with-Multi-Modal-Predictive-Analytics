"""
39-Class Plant Disease Detection Prediction Script

This script loads your trained 39-class plant disease detection model and makes predictions on images.
It then uses LLM to provide detailed information about the detected disease.
"""

import torch
import torch.nn as nn
import os
from torchvision.models import resnet18
import torchvision.transforms as transforms
from PIL import Image
import json
import numpy as np

# Define the 39 plant disease classes from your model
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

def load_39class_model(model_path="plant_disease_model (1).pth"):
    """Load the trained 39-class plant disease detection model"""
    
    print("Loading 39-class plant disease detection model...")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    try:
        # Load the state dict
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Initialize model architecture (ResNet18) with 39 classes
        model = resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 39)
        
        # Load model weights
        model.load_state_dict(state_dict)
        model.eval()
        
        print("39-class model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def preprocess_image(image_path):
    """Preprocess an image for model prediction"""
    
    # Define image transformations (same as used during training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    
    return image_tensor

def predict_39class_disease(model, image_tensor, top_k=3):
    """Make a prediction using the 39-class model"""
    
    with torch.no_grad():
        output = model(image_tensor)
        
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # Get top-k predictions
        top_probabilities, top_indices = torch.topk(probabilities, top_k)
        
        # Convert to lists
        top_probs = top_probabilities.tolist()
        top_idxs = top_indices.tolist()
        
        # Get class names
        predictions = []
        for i in range(len(top_idxs)):
            class_name = PLANT_DISEASE_CLASSES_39[top_idxs[i]]
            confidence = top_probs[i]
            predictions.append({
                'class_index': top_idxs[i],
                'class_name': class_name,
                'confidence': confidence
            })
        
        return predictions

def get_disease_info(class_name):
    """Get detailed information about a specific plant disease"""
    
    # Extract crop and disease from class name
    parts = class_name.split('___')
    if len(parts) == 2:
        crop = parts[0].replace('_', ' ').replace('(maize)', 'corn').replace('(including sour)', '')
        disease = parts[1].replace('_', ' ').replace(' Two-spotted spider mite', ' (Two-spotted spider mite)')
    else:
        crop = "Plant"
        disease = class_name.replace('_', ' ')
    
    return {
        'crop': crop,
        'disease': disease,
        'full_name': class_name
    }

def generate_llm_disease_report(disease_info, confidence):
    """Generate a detailed disease report using rule-based approach"""
    
    crop = disease_info['crop']
    disease = disease_info['disease']
    
    # Disease-specific information
    disease_details = {
        "Apple scab": {
            "description": "A fungal disease that affects the leaves and fruits of apple trees, causing dark, olive-green spots.",
            "symptoms": "Dark, olive-green spots on leaves and fruits. Leaves may turn yellow and drop prematurely.",
            "treatment": "Apply fungicides containing captan or sulfur. Remove and destroy infected leaves.",
            "prevention": "Ensure good air circulation, avoid overhead watering, and apply preventive fungicides."
        },
        "Black rot": {
            "description": "A fungal disease that causes dark, sunken lesions on fruits and cankers on branches.",
            "symptoms": "Dark, sunken lesions on fruits with concentric rings. Cankers on branches and trunk.",
            "treatment": "Remove and destroy infected fruits and branches. Apply fungicides during bloom period.",
            "prevention": "Prune for good air circulation, remove mummified fruits, and apply preventive fungicides."
        },
        "Cedar apple rust": {
            "description": "A fungal disease that requires both apple and cedar trees to complete its life cycle.",
            "symptoms": "Yellow-orange spots on upper leaf surfaces. Horn-like projections on undersides of leaves.",
            "treatment": "Remove nearby juniper/cedar trees if possible. Apply fungicides in spring.",
            "prevention": "Plant resistant apple varieties. Apply preventive fungicides during spring."
        },
        "Powdery mildew": {
            "description": "A common fungal disease that appears as white, powdery spots on leaves and stems.",
            "symptoms": "White, powdery fungal growth on leaf surfaces, stems, and flowers.",
            "treatment": "Improve air circulation. Apply fungicides containing sulfur or potassium bicarbonate.",
            "prevention": "Proper spacing of plants. Avoid overhead watering."
        },
        "Early blight": {
            "description": "A fungal disease that causes dark spots with concentric rings on leaves.",
            "symptoms": "Concentric rings on leaves forming a target-like pattern. Yellowing and death of lower leaves.",
            "treatment": "Remove infected leaves. Apply fungicides containing chlorothalonil or mancozeb.",
            "prevention": "Rotate crops, space plants properly, and apply preventive fungicides."
        },
        "Late blight": {
            "description": "A serious fungal disease that can rapidly destroy plants.",
            "symptoms": "Water-soaked lesions on leaves, stems, and fruits. White fungal growth on leaf undersides.",
            "treatment": "Remove and destroy infected plants. Apply fungicides containing copper or chlorothalonil.",
            "prevention": "Avoid overhead watering, ensure good air circulation, and apply preventive fungicides."
        },
        "Leaf Mold": {
            "description": "A fungal disease that affects leaves in humid conditions.",
            "symptoms": "Pale green to yellow spots on upper leaf surfaces. Olive-green to brown mold on undersides.",
            "treatment": "Improve air circulation. Apply fungicides containing chlorothalonil.",
            "prevention": "Avoid overhead watering, space plants properly, and maintain lower humidity."
        },
        "Bacterial spot": {
            "description": "A bacterial disease that causes spots on leaves, stems, and fruits.",
            "symptoms": "Small, water-soaked spots on leaves that turn brown. Raised, blister-like spots on fruits.",
            "treatment": "Remove infected plants. Apply copper-based bactericides.",
            "prevention": "Avoid working with plants when wet, rotate crops, and use disease-free seeds."
        },
        "healthy": {
            "description": "The plant appears to be healthy with no signs of disease.",
            "symptoms": "No visible signs of disease or pest damage.",
            "treatment": "Continue with regular plant care practices.",
            "prevention": "Maintain good cultural practices to keep plants healthy."
        }
    }
    
    # Get disease details or use generic template
    disease_key = disease.lower()
    if disease_key in disease_details:
        details = disease_details[disease_key]
    elif "healthy" in disease_key:
        details = disease_details["healthy"]
    else:
        details = {
            "description": f"A condition affecting {crop} plants.",
            "symptoms": "Refer to agricultural resources for specific symptoms.",
            "treatment": "Consult with a local agricultural expert for proper treatment.",
            "prevention": "Practice good crop rotation and maintain proper plant hygiene."
        }
    
    # Generate comprehensive report
    report = f"""
# Plant Disease Analysis Report

## Disease Identification
- **Crop**: {crop}
- **Disease**: {disease}
- **Confidence**: {confidence:.2%}

## Disease Information
**Description**: {details['description']}

**Key Symptoms**:
{details['symptoms']}

## Treatment Recommendations
{details['treatment']}

## Prevention Strategies
{details['prevention']}

## Additional Information
For more detailed information about this disease and its management, consider consulting with local agricultural extension services or using our full AI-powered advisory system.
"""
    
    return report.strip()

def save_prediction_results(image_path, predictions, output_file=None):
    """Save prediction results to a JSON file"""
    
    results = {
        'image_path': image_path,
        'predictions': predictions,
        'timestamp': str(torch.utils.data.time.time()) if hasattr(torch.utils.data, 'time') else 'N/A'
    }
    
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_file = f"{base_name}_predictions.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    return output_file

def main():
    """Main function to test the 39-class plant disease detection model"""
    
    print("39-Class Plant Disease Detection System")
    print("=" * 40)
    print("This system uses your trained 39-class plant disease detection model.")
    print("Ensure you have a trained model file named 'plant_disease_model (1).pth' in this directory.")
    
    try:
        # Load the 39-class model
        model_path = "plant_disease_model (1).pth"
        if not os.path.exists(model_path):
            print(f"Model file '{model_path}' not found.")
            print("Please ensure you have a trained 39-class model in this directory.")
            return
        
        model = load_39class_model(model_path)
        
        # Get image path from user
        print("\nTo test the model, please provide an image path.")
        print("Supported formats: JPG, JPEG, PNG")
        image_path = input("Enter image path (or 'quit' to exit): ").strip()
        
        if image_path.lower() == 'quit':
            return
            
        if not os.path.exists(image_path):
            print(f"Image file not found at {image_path}")
            return
        
        print(f"\nProcessing: {image_path}")
        
        try:
            # Preprocess the image
            image_tensor = preprocess_image(image_path)
            
            # Make prediction
            predictions = predict_39class_disease(model, image_tensor, top_k=3)
            
            print("\nTop 3 Predictions:")
            print("-" * 30)
            for i, pred in enumerate(predictions, 1):
                print(f"{i}. {pred['class_name']}")
                print(f"   Confidence: {pred['confidence']:.2%}")
            
            # Get detailed information for the top prediction
            top_prediction = predictions[0]
            disease_info = get_disease_info(top_prediction['class_name'])
            
            print(f"\nDetailed Analysis for {top_prediction['class_name']}:")
            print("-" * 50)
            
            # Generate LLM-based report
            report = generate_llm_disease_report(disease_info, top_prediction['confidence'])
            print(report)
            
            # Save results
            output_file = save_prediction_results(image_path, predictions)
            print(f"\nFor more detailed analysis, check the output file: {output_file}")
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "="*50)
        print("39-class plant disease detection completed!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()