import os
import torch
import torch.nn as nn

def load_disease_model(model_path=None):
    """
    Load the trained disease detection model
    
    Args:
        model_path (str): Path to the model weights file. If None, uses default path.
        
    Returns:
        torch.nn.Module: Loaded model in evaluation mode
    """
    try:
        # Set default model path if not provided
        if model_path is None:
            # Get the project root directory
            try:
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                model_path = os.path.join(project_root, 'models', 'plant_disease', 'plant_disease_model (1).pth')
            except Exception as e:
                print(f"Error determining model path: {e}")
                return None
        
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"Disease detection model not found at {model_path}")
            return None
        
        # Load the state dict to determine the correct number of classes
        try:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        except Exception as e:
            print(f"Error loading model state dict: {e}")
            return None
        
        # Determine number of classes from the model weights
        if 'fc.weight' in state_dict:
            num_classes = state_dict['fc.weight'].shape[0]
            print(f"Model trained with {num_classes} classes")
        else:
            # Default to 38 classes for PlantVillage dataset
            num_classes = 38
            print("Using default 38 classes for PlantVillage dataset")
        
        # Initialize model architecture (ResNet18 as example)
        try:
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        except Exception as e:
            print(f"Error loading ResNet18 from torch hub: {e}")
            # Fallback to manual ResNet18 definition
            try:
                from torchvision.models import resnet18
                model = resnet18(pretrained=False)
            except Exception as e2:
                print(f"Error loading ResNet18 manually: {e2}")
                return None
            
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        # Load model weights
        try:
            model.load_state_dict(state_dict)
            model.eval()
        except Exception as e:
            print(f"Error loading model weights: {e}")
            return None
        
        print(f"Disease detection model loaded successfully from {model_path}")
        return model
        
    except Exception as e:
        print(f"Error loading disease detection model: {str(e)}")
        # Return None instead of raising exception to prevent server crash
        return None

def get_model_info(model_path=None):
    """
    Get information about the trained model
    
    Args:
        model_path (str): Path to the model weights file. If None, uses default path.
        
    Returns:
        dict: Dictionary containing model information
    """
    try:
        # Set default model path if not provided
        if model_path is None:
            # Get the project root directory
            try:
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                model_path = os.path.join(project_root, 'models', 'plant_disease', 'plant_disease_model (1).pth')
            except Exception as e:
                print(f"Error determining model path: {e}")
                return None
        
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"Disease detection model not found at {model_path}")
            return None
        
        # Load the state dict
        try:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        except Exception as e:
            print(f"Error loading model state dict: {e}")
            return None
        
        # Determine number of classes from the model weights
        if 'fc.weight' in state_dict:
            num_classes = state_dict['fc.weight'].shape[0]
        else:
            # Default to 38 classes for PlantVillage dataset
            num_classes = 38
        
        # Get model size
        try:
            model_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
        except Exception as e:
            print(f"Error getting model size: {e}")
            model_size = 0
        
        return {
            'num_classes': num_classes,
            'model_size_mb': model_size,
            'model_path': model_path
        }
        
    except Exception as e:
        print(f"Error getting model information: {str(e)}")
        return None