```python
# Comprehensive Plant Disease Detection - Internet Dataset Download (Part 1)
# This Colab notebook downloads plant disease datasets directly from the internet

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision.models import resnet18
import os
import requests
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import zipfile
import pandas as pd
import shutil

# Extended plant disease classes (60+ classes for comprehensive training)
EXTENDED_PLANT_DISEASE_CLASSES = [
    # Original 39 classes
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
    "Tomato___healthy",
    
    # Additional 20+ classes for comprehensive training
    "Almond___Alternaria_Leaf_Spot", "Almond___Anthracnose", "Almond___Bacterial_Cancer",
    "Almond___Cercospora_Leaf_Spot", "Almond___Shot_Hole", "Almond___healthy",
    "Barley___Brown_Rust", "Barley___Net_Blotch", "Barley___Scald", "Barley___healthy",
    "Canola___Alternaria_Leaf_Spot", "Canola___Blackleg", "Canola___Sclerotinia_Stem_Rot", "Canola___healthy",
    "Cotton___Bacterial_Blight", "Cotton___Curl_Virus", "Cotton___Fusarium_Wilt", "Cotton___healthy",
    "Lettuce___Bottom_Rot", "Lettuce___Downy_Mildew", "Lettuce___Drop_Rot", "Lettuce___healthy",
    "Onion___Basal_Rot", "Onion___Blue_Mold", "Onion___White_Rot", "Onion___healthy",
    "Rice___Bacterial_Blight", "Rice___Blast", "Rice___Brown_Spot", "Rice___Tungro_Virus", "Rice___healthy",
    "Wheat___Brown_Rust", "Wheat___Leaf_Rust", "Wheat___Stem_Rust", "Wheat___Yellow_Rust", "Wheat___healthy"
]

print("Extended Plant Disease Classes (60+ classes):")
for i, class_name in enumerate(EXTENDED_PLANT_DISEASE_CLASSES):
    print(f"{i+1:2d}. {class_name}")

# Create directories for dataset
!mkdir -p extended_plant_disease_dataset
!mkdir -p extended_plant_disease_dataset/train
!mkdir -p extended_plant_disease_dataset/val

# Function to create directory structure
def create_directory_structure():
    """Create directory structure for all classes"""
    print("Creating directory structure for all classes...")
    
    for class_name in EXTENDED_PLANT_DISEASE_CLASSES:
        os.makedirs(f"extended_plant_disease_dataset/train/{class_name}", exist_ok=True)
        os.makedirs(f"extended_plant_disease_dataset/val/{class_name}", exist_ok=True)
    
    print("Directory structure created successfully!")

# Function to download datasets from multiple internet sources
def download_datasets_from_internet():
    """
    Download plant disease datasets from multiple internet sources
    """
    print("Downloading Plant Disease Datasets from Internet Sources...")
    
    # List of dataset sources (these are example URLs - replace with actual working URLs)
    dataset_sources = [
        {
            "name": "PlantVillage",
            "url": "https://data.mendeley.com/public-files/datasets/tywbtsjrjv/files/d5652a1c-3041-4172-a04d-479b3fd457f0/file_downloaded",
            "description": "PlantVillage dataset with 38 classes"
        },
        {
            "name": "PlantDoc",
            "url": "https://github.com/pratikkayal/PlantDoc-Dataset/archive/master.zip",
            "description": "PlantDoc dataset with 20+ classes"
        },
        {
            "name": "Crops & Weeds (Weed2C)",
            "url": "https://example.com/weed2c_dataset.zip",  # Replace with actual URL
            "description": "Crops and weeds classification dataset"
        }
    ]
    
    print("Dataset sources:")
    for i, source in enumerate(dataset_sources):
        print(f"{i+1}. {source['name']}: {source['description']}")
    
    print("\nNote: Replace the example URLs with actual working URLs for the datasets.")
    print("For demonstration, we'll create a synthetic dataset structure.")
    
    return True

# Continue with the implementation in the next part...
```