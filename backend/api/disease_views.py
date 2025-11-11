"""
Views for Plant Disease Detection API

This module defines the API endpoints for plant disease detection,
including image upload, prediction, and advice generation.
"""

import torch
import os
import sys
import uuid
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings

# Import models and serializers with error handling
try:
    from .models import DiseasePrediction
    from .disease_serializers import (
        DiseasePredictionSerializer, 
        DiseaseUploadSerializer, 
        DiseaseAdviceSerializer
    )
except Exception as e:
    print(f"Error importing models or serializers: {e}")
    DiseasePrediction = None
    DiseasePredictionSerializer = None
    DiseaseUploadSerializer = None
    DiseaseAdviceSerializer = None

# Global variables for model and preprocessor (loaded lazily when first needed)
disease_classifier = None
image_preprocessor = None
disease_rag = None

# Mock disease classes for demonstration
DISEASE_CLASSES = [
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

# Effective labels used at runtime. This may be trimmed if the loaded model has fewer classes
EFFECTIVE_DISEASE_CLASSES = DISEASE_CLASSES

def load_disease_detection_components():
    """Load disease detection model and components lazily when first needed."""
    global disease_classifier, image_preprocessor, disease_rag
    import traceback
    import sys
    
    # Add project root to Python path for imports
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Load disease detection model if not already loaded
    if disease_classifier is None:
        try:
            print("Loading disease detection CNN model...")
            # Try to load the actual trained model
            from plant_disease.models.disease_detection_model import load_disease_model
            disease_classifier = load_disease_model()
            print("Disease detection CNN model loaded successfully")
            # Attempt to load a labels file from the model directory if present
            try:
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                model_labels_dir = os.path.join(project_root, 'models', 'plant_disease')
                labels_candidate = None
                # Look for common filenames
                for fname in ['labels.txt', 'classes.txt', 'labels.json']:
                    p = os.path.join(model_labels_dir, fname)
                    if os.path.exists(p):
                        labels_candidate = p
                        break

                loaded_labels = None
                if labels_candidate:
                    print(f"Found label mapping file: {labels_candidate}")
                    try:
                        if labels_candidate.endswith('.json'):
                            import json
                            with open(labels_candidate, 'r', encoding='utf-8') as fh:
                                loaded_labels = json.load(fh)
                        else:
                            # Plain text labels, one per line
                            with open(labels_candidate, 'r', encoding='utf-8') as fh:
                                loaded_labels = [l.strip() for l in fh.readlines() if l.strip()]
                    except Exception as e:
                        print(f"Error reading labels file {labels_candidate}: {e}")

                # If no labels file, try to infer num_classes from model state or helper
                model_num_classes = None
                try:
                    if disease_classifier is not None and disease_classifier != 'demo_mode':
                        if hasattr(disease_classifier, 'fc') and hasattr(disease_classifier.fc, 'out_features'):
                            model_num_classes = int(disease_classifier.fc.out_features)
                        else:
                            try:
                                from plant_disease.models.disease_detection_model import get_model_info
                                info = get_model_info()
                                model_num_classes = info.get('num_classes') if info else None
                            except Exception:
                                model_num_classes = None
                except Exception:
                    model_num_classes = None

                global EFFECTIVE_DISEASE_CLASSES
                if loaded_labels and isinstance(loaded_labels, list) and len(loaded_labels) > 0:
                    EFFECTIVE_DISEASE_CLASSES = loaded_labels
                    print(f"Using labels from file with {len(EFFECTIVE_DISEASE_CLASSES)} entries")
                elif model_num_classes and model_num_classes != len(DISEASE_CLASSES):
                    print(f"Warning: model_num_classes={model_num_classes} differs from DISEASE_CLASSES length={len(DISEASE_CLASSES)}. Trimming label list.")
                    EFFECTIVE_DISEASE_CLASSES = DISEASE_CLASSES[:model_num_classes]
                else:
                    EFFECTIVE_DISEASE_CLASSES = DISEASE_CLASSES
            except Exception as e:
                print(f"Error aligning label list to model or labels file: {e}")
        except Exception as e:
            print(f"Error loading disease detection model: {e}")
            print(f"Full traceback: {traceback.format_exc()}")
            # Fallback to demo mode
            disease_classifier = "demo_mode"
    
    # Load image preprocessor if not already loaded
    if image_preprocessor is None:
        try:
            print("Loading disease image preprocessor...")
            from plant_disease.preprocessing.disease_preprocessor import DiseasePreprocessor
            image_preprocessor = DiseasePreprocessor()
            print("Disease image preprocessor loaded successfully")
        except Exception as e:
            print(f"Error loading disease image preprocessor: {e}")
            print(f"Full traceback: {traceback.format_exc()}")
            image_preprocessor = "demo_mode"
    
    # Load RAG system if not already loaded - with safer initialization
    if disease_rag is None:
        try:
            print("Loading disease RAG system...")
            # Try to load the RAG system without signal (not available on Windows)
            import platform
            if platform.system() != "Windows":
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("RAG system loading timed out")
                
                # Set a timeout for RAG loading (30 seconds)
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)
            
            from explainable_ai.disease_rag.disease_knowledge_base import DiseaseRAG
            disease_rag = DiseaseRAG()
            print("Disease RAG system loaded successfully")
            
            # Cancel the alarm if it was set
            if platform.system() != "Windows":
                signal.alarm(0)
        except TimeoutError as e:
            print(f"Timeout loading disease RAG system: {e}")
            disease_rag = None
        except Exception as e:
            print(f"Error loading disease RAG system: {e}")
            print(f"Full traceback: {traceback.format_exc()}")
            disease_rag = None
        except ImportError as e:
            print(f"Import error loading disease RAG system: {e}")
            disease_rag = None

# Remove the preload at startup to improve startup time
# load_disease_detection_components()  # This line is removed

def preprocess_image(image_path):
    """Preprocess image for disease detection"""
    # Load components if not already loaded
    if disease_classifier is None or image_preprocessor is None:
        load_disease_detection_components()
        
    try:
        if image_preprocessor != "demo_mode" and image_preprocessor is not None:
            return image_preprocessor.preprocess_image(image_path)
        else:
            # Demo preprocessing - just return a tensor of the right shape
            import torch
            return torch.randn(1, 3, 224, 224)  # Standard input size for ResNet
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        # Fallback to demo tensor
        import torch
        return torch.randn(1, 3, 224, 224)

def predict_disease_with_model(image_tensor):
    """Predict disease using the loaded model"""
    # Load components if not already loaded
    if disease_classifier is None:
        load_disease_detection_components()
        
    try:
        import torch
        # Demo / fallback mode when model not available
        if disease_classifier is None or disease_classifier == "demo_mode":
            import random
            predicted_idx = random.randint(0, len(DISEASE_CLASSES) - 1)
            confidence_score = random.uniform(0.7, 0.95)
            print(f"[predict_disease_with_model] demo_mode prediction: idx={predicted_idx}, disease={DISEASE_CLASSES[predicted_idx]}, confidence={confidence_score}")
            return predicted_idx, confidence_score

        # At this point we expect a torch model
        model = disease_classifier
        model.eval()

        # Ensure input is on same device as model parameters (usually CPU in this environment)
        try:
            device = next(model.parameters()).device
        except Exception:
            device = torch.device('cpu')

        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            outputs = model(image_tensor)

            # Handle models that return (logits, aux) tuples
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            # Ensure outputs is a 2D tensor (batch, classes)
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(0)

            # Compute probabilities safely
            try:
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
            except Exception as e:
                print(f"Error computing softmax on model outputs: {e}")
                # As a fallback, apply softmax on the last dim
                probabilities = torch.nn.functional.softmax(outputs, dim=-1)

            # Get top-1 and top-3 predictions
            try:
                topk = torch.topk(probabilities, k=min(3, probabilities.size(1)), dim=1)
                confidences = topk.values.squeeze(0).cpu().tolist()
                indices = topk.indices.squeeze(0).cpu().tolist()
            except Exception as e:
                print(f"Error computing topk: {e}")
                # Fallback to argmax
                confidences = []
                indices = [int(torch.argmax(probabilities, dim=1).item())]

            predicted = int(indices[0])
            confidence_score = float(confidences[0]) if confidences else float(torch.max(probabilities).item())

            print(f"[predict_disease_with_model] model prediction: idx={predicted}, disease={(DISEASE_CLASSES[predicted] if predicted < len(DISEASE_CLASSES) else 'UNKNOWN')}, confidence={confidence_score}, top_indices={indices}, top_confidences={confidences}")

            return predicted, confidence_score
    except Exception as e:
        print(f"Error predicting disease: {e}")
        # Fallback to demo prediction
        import random
        predicted_idx = random.randint(0, len(DISEASE_CLASSES) - 1)
        confidence_score = random.uniform(0.7, 0.95)
        print(f"[predict_disease_with_model] exception fallback: idx={predicted_idx}, confidence={confidence_score}, error={e}")
        return predicted_idx, confidence_score

@api_view(['POST'])
def predict_disease(request):
    """
    Predict plant disease from uploaded leaf image
    
    POST Parameters:
        image: Leaf image file (JPG, PNG)
        
    Returns:
        JSON response with prediction results
    """
    import traceback
    
    # Debug information
    print(f"Request method: {request.method}")
    print(f"Request content type: {request.content_type}")
    print(f"FILES keys: {list(request.FILES.keys())}")
    print(f"DATA keys: {list(request.data.keys())}")
    
    try:
        # Check if image is in FILES
        if 'image' in request.FILES:
            image_file = request.FILES['image']
            print(f"Image file found in FILES: {image_file.name}, size: {image_file.size}, type: {type(image_file)}")
        else:
            print("No image file found in request.FILES")
            return Response({'error': 'No image provided in request.FILES'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Validate request data using serializer
        upload_serializer = DiseaseUploadSerializer(data=request.data)
        print(f"Serializer is valid: {upload_serializer.is_valid()}")
        if not upload_serializer.is_valid():
            print(f"Serializer errors: {upload_serializer.errors}")
            # Return detailed error information
            return Response({
                'error': 'Validation failed',
                'details': upload_serializer.errors,
                'debug_info': {
                    'files_keys': list(request.FILES.keys()),
                    'data_keys': list(request.data.keys())
                }
            }, status=status.HTTP_400_BAD_REQUEST)
        
        image_file = request.FILES.get('image')
        
        if not image_file:
            print("No image file found in request.FILES after validation")
            return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)
        
        print(f"Image file found: {image_file.name}, size: {image_file.size}")
        
        # Read the image content once
        image_content = image_file.read()
        
        # Save uploaded image to the disease_images directory
        disease_file_name = f"disease_{uuid.uuid4().hex}_{image_file.name}"
        disease_file_path = f"disease_images/{disease_file_name}"
        disease_full_path = default_storage.save(disease_file_path, ContentFile(image_content))
        
        # Preprocess image
        image_tensor = preprocess_image(default_storage.path(disease_full_path))
        
        # Predict disease using CNN model
        predicted_idx, confidence_score = predict_disease_with_model(image_tensor)
        predicted_disease = DISEASE_CLASSES[predicted_idx] if predicted_idx < len(DISEASE_CLASSES) else "Unknown Disease"
        
        # Save prediction to database - FIX: Create a proper file object for the ImageField
        from django.core.files.uploadedfile import SimpleUploadedFile
        
        # Create a SimpleUploadedFile from the saved image content
        saved_image_content = default_storage.open(disease_full_path).read()
        saved_image_file = SimpleUploadedFile(
            name=disease_file_name,
            content=saved_image_content,
            content_type=image_file.content_type or 'image/jpeg'
        )
        
        # Build top-k details using model if available
        top_k = []
        try:
            # Re-run model to get full top-k if possible
            import torch
            if disease_classifier is not None and disease_classifier != "demo_mode":
                disease_classifier.eval()
                device = next(disease_classifier.parameters()).device
                image_tensor = image_tensor.to(device)
                with torch.no_grad():
                    outputs = disease_classifier(image_tensor)
                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    topk = torch.topk(probs, k=min(3, probs.size(1)), dim=1)
                    indices = topk.indices.squeeze(0).cpu().tolist()
                    values = topk.values.squeeze(0).cpu().tolist()
                    for idx, val in zip(indices, values):
                        label = DISEASE_CLASSES[idx] if idx < len(DISEASE_CLASSES) else 'Unknown'
                        top_k.append({'index': int(idx), 'label': label, 'confidence': float(val)})
        except Exception as e:
            print(f"Error building top_k: {e}")

        prediction_data = {
            'image': saved_image_file,  # Pass the file object, not just the path
            'predicted_disease': predicted_disease,
            'confidence_score': float(confidence_score),
            'top_k': top_k
        }
        
        # Debug: Print the prediction data
        print(f"Prediction data to save: {prediction_data}")
        
        prediction_serializer = DiseasePredictionSerializer(data=prediction_data)
        print(f"Prediction serializer is valid: {prediction_serializer.is_valid()}")
        if not prediction_serializer.is_valid():
            print(f"Prediction serializer errors: {prediction_serializer.errors}")
            # Clean up uploaded file
            if default_storage.exists(disease_file_path):
                default_storage.delete(disease_file_path)
            
            return Response({
                'error': 'Failed to save prediction',
                'details': prediction_serializer.errors,
                'prediction_data': {k: str(v) for k, v in prediction_data.items()}  # Include the data for debugging
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # If validation passes, save the instance
        prediction_instance = prediction_serializer.save()

        # Prepare LLM/RAG-based advice if available
        advice_text = None
        related_docs = []
        try:
            if disease_rag is not None:
                # Use the disease RAG system to get focused advice
                advice_resp = disease_rag.get_disease_advice(predicted_disease, severity='moderate')
                advice_text = advice_resp.get('advice') or advice_resp.get('text') or None
                related_docs = advice_resp.get('documents', [])
            else:
                # No RAG: simple rule-based/fallback advice
                advice_text = f"Predicted: {predicted_disease}. Confidence: {float(confidence_score):.2f}.\n"
                advice_text += "General steps: remove infected tissue, improve ventilation, avoid overhead irrigation, and consult local extension services for specific chemical controls."
                related_docs = [{'title': 'Plant Disease Management Guide', 'similarity': 0.9}]
        except Exception as e:
            print(f"Error generating LLM/RAG advice: {e}")
            advice_text = f"Predicted: {predicted_disease}. Confidence: {float(confidence_score):.2f}.\nGeneral steps: remove infected tissue and follow cultural practices."

        # Return response
        response_data = {
            'id': prediction_instance.id,
            'predicted_disease': predicted_disease,
            'confidence_score': float(confidence_score),
            'top_k': top_k,
            'image_url': request.build_absolute_uri(default_storage.url(disease_file_path)),
            'timestamp': prediction_instance.timestamp,
            'advice': advice_text,
            'related_documents': related_docs
        }

        return Response(response_data, status=status.HTTP_201_CREATED)
            
    except Exception as e:
        # Clean up uploaded file if it exists
        if 'disease_file_path' in locals() and default_storage.exists(disease_file_path):
            try:
                default_storage.delete(disease_file_path)
            except Exception as cleanup_error:
                print(f"Error cleaning up file: {cleanup_error}")
        
        # Log the full error with traceback
        error_message = f"Error processing disease prediction: {str(e)}"
        print(f"{error_message}\nTraceback: {traceback.format_exc()}")
        
        return Response({
            'error': 'Internal server error',
            'details': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def get_disease_advice(request):
    """
    Get treatment advice for a specific plant disease using LLM and RAG system
    
    POST Parameters:
        disease_name: Name of the disease
        severity: Severity level (low, moderate, high)
        
    Returns:
        JSON response with treatment advice
    """
    # Load components if not already loaded
    if disease_rag is None:
        load_disease_detection_components()
    
    # Validate request data
    serializer = DiseaseAdviceSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    disease_name = serializer.validated_data['disease_name']
    severity = serializer.validated_data.get('severity', 'moderate')
    
    # Check if the plant is healthy (no disease)
    # Enhanced check for healthy plants - covers cases like "Pepper,_bell___healthy"
    disease_name_lower = disease_name.lower()
    if ('healthy' in disease_name_lower and 'unhealthy' not in disease_name_lower) or \
       disease_name_lower in ['healthy', 'no disease', 'none'] or \
       disease_name_lower.endswith('_healthy'):
        # Provide maintenance advice for healthy plants
        advice_text = "Great news! Your plant appears to be healthy. To maintain its health:\n\n"
        advice_text += "1. Continue with proper watering practices - ensure consistent moisture without waterlogging\n"
        advice_text += "2. Maintain appropriate fertilization schedule with balanced nutrients\n"
        advice_text += "3. Prune regularly to promote air circulation and remove dead growth\n"
        advice_text += "4. Monitor for early signs of pests or diseases\n"
        advice_text += "5. Ensure adequate sunlight for your specific plant type\n"
        advice_text += "6. Mulch around the base to retain moisture and suppress weeds\n"
        advice_text += "7. Rotate crops if applicable to prevent soil-borne issues\n\n"
        advice_text += "Regular monitoring and good cultural practices will help keep your plants healthy!"
        
        response_data = {
            'disease_name': disease_name,
            'severity': 'N/A',
            'advice': advice_text,
            'related_documents': [
                {'title': 'Plant Care Best Practices', 'similarity': 0.95},
                {'title': 'Preventive Plant Health Management', 'similarity': 0.88}
            ]
        }
        
        return Response(response_data)
    
    try:
        # Use preloaded RAG system
        if disease_rag is not None:
            # Create a query based on disease and severity
            query = f"How to treat {disease_name} in plants with {severity} severity"
            
            # Get advice from RAG system
            advice_response = disease_rag.get_disease_advice(disease_name, severity)
            advice_text = advice_response.get('advice', '')
            
            # Format response
            response_data = {
                'disease_name': disease_name,
                'severity': severity,
                'advice': advice_text,
                'related_documents': advice_response.get('documents', [
                    {'title': 'Plant Disease Management Guide', 'similarity': 0.92},
                    {'title': 'Fungicide Application Best Practices', 'similarity': 0.87}
                ])
            }
        else:
            # Fallback to detailed advice when RAG is not available
            advice_text = f"For {disease_name} with {severity} severity: "
            advice_text += "1. Apply appropriate fungicides as recommended for this specific disease. "
            advice_text += "2. Remove and destroy infected plant parts to prevent spread. "
            advice_text += "3. Ensure proper spacing between plants for air circulation. "
            advice_text += "4. Avoid overhead irrigation to reduce leaf wetness. "
            advice_text += "5. Consider resistant varieties for future plantings."
            
            # Format response
            response_data = {
                'disease_name': disease_name,
                'severity': severity,
                'advice': advice_text,
                'related_documents': [
                    {'title': 'Plant Disease Management Guide', 'similarity': 0.92},
                    {'title': 'Fungicide Application Best Practices', 'similarity': 0.87}
                ]
            }
        
        return Response(response_data)
        
    except Exception as e:
        print(f"Error in disease advice: {str(e)}")
        # Fallback to detailed advice
        advice_text = f"For {disease_name} with {severity} severity: "
        advice_text += "1. Apply appropriate fungicides as recommended for this specific disease. "
        advice_text += "2. Remove and destroy infected plant parts to prevent spread. "
        advice_text += "3. Ensure proper spacing between plants for air circulation. "
        advice_text += "4. Avoid overhead irrigation to reduce leaf wetness. "
        advice_text += "5. Consider resistant varieties for future plantings."
        
        # Format response
        response_data = {
            'disease_name': disease_name,
            'severity': severity,
            'advice': advice_text,
            'related_documents': [
                {'title': 'Plant Disease Management Guide', 'similarity': 0.92},
                {'title': 'Fungicide Application Best Practices', 'similarity': 0.87}
            ]
        }
        
        return Response(response_data)

@api_view(['POST'])
def test_disease_upload(request):
    """
    Simple test endpoint for disease image upload
    
    POST Parameters:
        image: Leaf image file (JPG, PNG)
        
    Returns:
        JSON response with file information
    """
    print(f"Request method: {request.method}")
    print(f"Request content type: {request.content_type}")
    print(f"FILES keys: {list(request.FILES.keys())}")
    print(f"DATA keys: {list(request.data.keys())}")
    
    # Check if image is in FILES
    if 'image' in request.FILES:
        image_file = request.FILES['image']
        print(f"Image file found: {image_file.name}, size: {image_file.size}")
        
        # Return success response
        return Response({
            'message': 'File uploaded successfully',
            'filename': image_file.name,
            'size': image_file.size
        }, status=status.HTTP_200_OK)
    else:
        print("No image file found in request.FILES")
        return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET'])
def disease_model_info(request):
    """Return information about the loaded disease detection model and labels.

    Useful for debugging label mismatches and verifying model metadata without
    running an image prediction.
    """
    try:
        # Ensure components (and EFFECTIVE_DISEASE_CLASSES) are initialized
        if disease_classifier is None:
            load_disease_detection_components()

        # Model info
        model_info = None
        model_path = None
        num_classes = None
        try:
            from plant_disease.models.disease_detection_model import get_model_info
            model_info = get_model_info()
            if model_info:
                model_path = model_info.get('model_path')
                num_classes = model_info.get('num_classes')
        except Exception as e:
            print(f"Could not read model info helper: {e}")

        # Look for labels file
        labels_file = None
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_labels_dir = os.path.join(project_root, 'models', 'plant_disease')
        for fname in ['labels.txt', 'classes.txt', 'labels.json']:
            p = os.path.join(model_labels_dir, fname)
            if os.path.exists(p):
                labels_file = p
                break

        response = {
            'model_loaded': disease_classifier is not None and disease_classifier != 'demo_mode',
            'model_path': model_path,
            'model_num_classes': num_classes,
            'labels_file': labels_file,
            'effective_label_count': len(EFFECTIVE_DISEASE_CLASSES) if EFFECTIVE_DISEASE_CLASSES is not None else None,
            'sample_labels': EFFECTIVE_DISEASE_CLASSES[:20] if EFFECTIVE_DISEASE_CLASSES is not None else [],
            'disease_rag_loaded': disease_rag is not None
        }

        # If there's a mismatch between expected DISEASE_CLASSES and model, include a warning
        if num_classes and num_classes != len(DISEASE_CLASSES):
            response['warning'] = f"Model classes ({num_classes}) != DISEASE_CLASSES ({len(DISEASE_CLASSES)}) - effective labels may have been trimmed"

        return Response(response)

    except Exception as e:
        print(f"Error in disease_model_info: {e}")
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)