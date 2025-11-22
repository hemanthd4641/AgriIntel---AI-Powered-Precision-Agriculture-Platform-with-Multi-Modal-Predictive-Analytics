"""
Enhanced training script for fertilizer recommendation models with detailed insights and explanations.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import sys
import os
import json

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from crop_recommendation.preprocessing.recommendation_preprocessor import RecommendationPreprocessor
from crop_recommendation.models.crop_recommendation_models import FertilizerRecommendationModel


def create_sample_fertilizer_data():
    """Create sample fertilizer recommendation dataset for demonstration."""
    # In a real implementation, you would load data from a file
    np.random.seed(42)
    
    # Generate sample data with 6 features to match the model expectation
    n_samples = 500
    data = {
        'Nitrogen': np.random.uniform(0, 140, n_samples),
        'Phosphorus': np.random.uniform(5, 145, n_samples),
        'Potassium': np.random.uniform(5, 205, n_samples),
        'pH': np.random.uniform(3.5, 10, n_samples),
        'Crop_Type_Encoded': np.random.randint(0, 5, n_samples),  # Encoded crop type
        'Moisture': np.random.uniform(20, 80, n_samples),
    }
    
    # Create fertilizer labels based on some simple rules
    fertilizers = []
    for i in range(n_samples):
        n, p, k, ph, crop_type, moisture = (
            data['Nitrogen'][i], data['Phosphorus'][i], data['Potassium'][i],
            data['pH'][i], data['Crop_Type_Encoded'][i], data['Moisture'][i]
        )
        
        # Simple rule-based fertilizer assignment for demonstration
        if n < 50:
            fertilizer = 'Urea'  # High nitrogen
        elif p < 30:
            fertilizer = 'DAP'   # High phosphorus
        elif k < 40:
            fertilizer = 'MOP'   # High potassium
        elif moisture < 40:
            fertilizer = 'SSP'   # Soluble phosphorus
        else:
            # Balanced fertilizer
            fertilizer = 'NPK 15-15-15'
        
        fertilizers.append(fertilizer)
    
    data['Fertilizer Name'] = fertilizers
    return pd.DataFrame(data)


def generate_fertilizer_insights(model, X_train, y_train, feature_names, class_names):
    """Generate insights about the fertilizer recommendation model."""
    insights = {
        'model_type': model.model_type,
        'training_samples': len(X_train),
        'features': feature_names,
        'classes': class_names.tolist() if hasattr(class_names, 'tolist') else list(class_names),
        'feature_importance': {},
        'model_performance': {}
    }
    
    return insights


def generate_fertilizer_recommendation_explanation(fertilizer, soil_conditions, crop_type, insights):
    """
    Generate a natural language explanation for fertilizer recommendation using rule-based logic.
    In a real implementation, this would use an LLM.
    """
    explanation = f"Based on the analysis of your soil conditions and crop type, {fertilizer} is the recommended fertilizer for your farm.\n\n"
    
    explanation += "Key factors supporting this recommendation:\n"
    
    # Add soil condition analysis
    n, p, k, ph = soil_conditions['Nitrogen'], soil_conditions['Phosphorus'], soil_conditions['Potassium'], soil_conditions['pH']
    explanation += f"- Soil nutrients (N:{n:.1f}, P:{p:.1f}, K:{k:.1f}) indicate "
    if n < 50:
        explanation += "nitrogen deficiency, making nitrogen-rich fertilizers like Urea ideal.\n"
    elif p < 30:
        explanation += "phosphorus deficiency, making phosphorus-rich fertilizers like DAP ideal.\n"
    elif k < 40:
        explanation += "potassium deficiency, making potassium-rich fertilizers like MOP ideal.\n"
    else:
        explanation += "balanced nutrient levels, making balanced fertilizers like NPK 15-15-15 suitable.\n"
    
    # Add pH analysis
    explanation += f"- Soil pH ({ph:.1f}) is "
    if 6.0 <= ph <= 7.5:
        explanation += "optimal for most crops and fertilizer uptake.\n"
    elif 5.5 <= ph < 6.0 or 7.5 < ph <= 8.0:
        explanation += "slightly outside optimal range but still suitable for most fertilizers.\n"
    else:
        explanation += "outside optimal range; consider soil amendments for better fertilizer effectiveness.\n"
    
    explanation += f"\n{fertilizer} is recommended because:\n"
    fertilizer_info = {
        'Urea': "- High nitrogen content (46% N)\n- Promotes leafy growth and chlorophyll production\n- Best applied before planting or during early growth stages",
        'DAP': "- High phosphorus content (46% P2O5)\n- Promotes root development and flowering\n- Best applied at planting time",
        'MOP': "- High potassium content (60% K2O)\n- Improves disease resistance and fruit quality\n- Best applied during fruit development",
        'SSP': "- Good source of phosphorus (16% P2O5) and calcium\n- Improves soil structure and root development\n- Suitable for acidic soils",
        'NPK 15-15-15': "- Balanced nutrition for all growth stages\n- Promotes overall plant health and development\n- Suitable for general application throughout the growing season"
    }
    explanation += fertilizer_info.get(fertilizer, "- A suitable fertilizer for your soil conditions and crop type")
    
    explanation += f"\n\nFor {crop_type} cultivation:\n"
    explanation += "- Apply according to recommended rates based on soil test results\n"
    explanation += "- Consider split applications for better nutrient uptake\n"
    explanation += "- Monitor plant response and adjust application rates accordingly"
    
    return explanation


def save_model_metadata(metadata, file_path):
    """Save model metadata to a JSON file."""
    try:
        with open(file_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Model metadata saved to {file_path}")
    except Exception as e:
        print(f"Error saving model metadata: {e}")


def train_fertilizer_recommendation_model_enhanced():
    """Train the enhanced fertilizer recommendation model with detailed insights."""
    print("Creating sample fertilizer dataset...")
    df = create_sample_fertilizer_data()
    print(f"Dataset created with {len(df)} samples")
    print(f"Fertilizer distribution:\n{df['Fertilizer Name'].value_counts()}")
    
    # Preprocess data
    print("Preprocessing data...")
    preprocessor = RecommendationPreprocessor()
    # Create a copy of the dataframe with the correct column structure for preprocessing
    df_for_preprocessing = df.copy()
    df_for_preprocessing.rename(columns={'Fertilizer Name': 'Fertilizer'}, inplace=True)
    X, _, y = preprocessor.preprocess_data(df_for_preprocessing, target_columns=['Fertilizer'])
    feature_names = ['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Crop_Type_Encoded', 'Moisture']
    class_names = df['Fertilizer Name'].unique()
    
    # Check if y is None and handle it
    if y is None:
        # Create y from the original dataframe
        y = df['Fertilizer Name'].values
        # Encode the labels
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)
        class_names = le.classes_
    
    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Gradient Boosting model
    print("Training Gradient Boosting model...")
    model = FertilizerRecommendationModel(n_estimators=100, learning_rate=0.1, max_depth=3)
    model.train(X_train, y_train)
    
    # Evaluate model
    pred, prob = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    
    # Generate insights
    print("Generating model insights...")
    insights = generate_fertilizer_insights(model, X_train, y_train, feature_names, class_names)
    insights['model_accuracy'] = float(accuracy)
    
    # Create sample explanation
    sample_conditions = {
        'soil_conditions': {'Nitrogen': 40, 'Phosphorus': 35, 'Potassium': 60, 'pH': 6.5},
        'crop_type': 'Wheat'
    }
    
    # Generate sample explanation for one of the fertilizers
    sample_fertilizer = class_names[0] if len(class_names) > 0 else 'NPK 15-15-15'
    explanation = generate_fertilizer_recommendation_explanation(
        sample_fertilizer, 
        sample_conditions['soil_conditions'], 
        sample_conditions['crop_type'], 
        insights
    )
    
    # Save metadata
    metadata = {
        'model_type': model.model_type,
        'accuracy': float(accuracy),
        'features': feature_names,
        'classes': class_names.tolist() if hasattr(class_names, 'tolist') else list(class_names),
        'insights': insights,
        'sample_explanation': explanation,
        'training_date': pd.Timestamp.now().isoformat()
    }
    
    # Save model, preprocessor, and metadata
    print("Saving model, preprocessor, and metadata...")
    os.makedirs('crop_recommendation/saved_models', exist_ok=True)
    joblib.dump(model, 'crop_recommendation/saved_models/fertilizer_model_enhanced.pkl')
    joblib.dump(preprocessor, 'crop_recommendation/saved_models/fertilizer_preprocessor_enhanced.pkl')
    save_model_metadata(metadata, 'crop_recommendation/saved_models/fertilizer_model_metadata.json')
    
    print("Enhanced fertilizer recommendation model and preprocessor saved successfully!")
    
    return model, preprocessor, metadata


if __name__ == "__main__":
    train_fertilizer_recommendation_model_enhanced()