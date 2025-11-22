"""
Image preprocessing module for plant disease detection.
"""

import numpy as np
from PIL import Image, ImageEnhance
import cv2


class DiseaseImagePreprocessor:
    """Preprocessor for plant disease detection images."""
    
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
    
    def load_and_preprocess_image(self, image_path):
        """
        Load and preprocess an image for disease detection.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            np.ndarray: Preprocessed image array
        """
        # Load image
        image = Image.open(image_path)
        
        # Resize image
        image = image.resize(self.target_size)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Normalize pixel values to [0, 1]
        image_array = image_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    def preprocess_pil_image(self, pil_image):
        """
        Preprocess a PIL image for disease detection.
        
        Args:
            pil_image (Image): PIL Image object
            
        Returns:
            np.ndarray: Preprocessed image array
        """
        # Resize image
        pil_image = pil_image.resize(self.target_size)
        
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(pil_image)
        
        # Normalize pixel values to [0, 1]
        image_array = image_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    def preprocess_cv2_image(self, cv2_image):
        """
        Preprocess an OpenCV image for disease detection.
        
        Args:
            cv2_image (np.ndarray): OpenCV image (BGR format)
            
        Returns:
            np.ndarray: Preprocessed image array
        """
        # Convert BGR to RGB
        image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, self.target_size)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        
        # Convert to numpy array
        image_array = np.array(pil_image)
        
        # Normalize pixel values to [0, 1]
        image_array = image_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    def augment_image(self, image_path, augmentations=5):
        """
        Apply data augmentation to generate multiple versions of an image.
        
        Args:
            image_path (str): Path to the image file
            augmentations (int): Number of augmented images to generate
            
        Returns:
            list: List of augmented image arrays
        """
        # Load image
        image = Image.open(image_path)
        
        augmented_images = []
        
        # Original image
        original_array = self.preprocess_pil_image(image)
        augmented_images.append(original_array)
        
        # Apply augmentations
        for _ in range(augmentations):
            # Random transformations
            augmented = image.copy()
            
            # Random rotation
            angle = np.random.uniform(-15, 15)
            augmented = augmented.rotate(angle)
            
            # Random horizontal flip
            if np.random.random() > 0.5:
                augmented = augmented.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Random brightness adjustment
            brightness_factor = np.random.uniform(0.8, 1.2)
            enhancer = ImageEnhance.Brightness(augmented)
            augmented = enhancer.enhance(brightness_factor)
            
            # Random contrast adjustment
            contrast_factor = np.random.uniform(0.8, 1.2)
            enhancer = ImageEnhance.Contrast(augmented)
            augmented = enhancer.enhance(contrast_factor)
            
            # Preprocess augmented image
            augmented_array = self.preprocess_pil_image(augmented)
            augmented_images.append(augmented_array)
        
        return augmented_images

# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = DiseaseImagePreprocessor(target_size=(224, 224))
    
    print("Disease Image Preprocessor initialized")
    print(f"Target size: {preprocessor.target_size}")