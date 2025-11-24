"""
Disease Image Preprocessor

This module provides preprocessing functions for plant disease detection images.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import os

class DiseasePreprocessor:
    """Preprocessor for plant disease detection images"""
    
    def __init__(self):
        """Initialize the preprocessor with standard transformations"""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Standard size for ResNet
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225]    # ImageNet standard deviations
            )
        ])
    
    def preprocess_image(self, image_path):
        """
        Preprocess an image for disease detection
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            torch.Tensor: Preprocessed image tensor with batch dimension
        """
        try:
            # Open and convert image to RGB
            image = Image.open(image_path).convert('RGB')
            
            # Apply transformations
            image_tensor = self.transform(image)
            
            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)
            
            return image_tensor
            
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    try:
        preprocessor = DiseasePreprocessor()
        print("Disease preprocessor initialized successfully!")
        
        # Test with a sample image if available
        # Note: This will fail if no sample image is available
        # image_tensor = preprocessor.preprocess_image("sample_leaf.jpg")
        # print(f"Preprocessed image tensor shape: {image_tensor.shape}")
        
    except Exception as e:
        print(f"Error initializing preprocessor: {str(e)}")