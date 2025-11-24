from rest_framework import serializers
from .models import Crop, DiseasePrediction, CropRecommendation, FertilizerRecommendation, YieldPrediction, MarketPrediction, PestPrediction

class CropSerializer(serializers.ModelSerializer):
    class Meta:
        model = Crop
        fields = '__all__'

class YieldPredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = YieldPrediction
        fields = '__all__'

class MarketPredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = MarketPrediction
        fields = '__all__'

class PestPredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = PestPrediction
        fields = '__all__'

# ========================================
# DISEASE DETECTION SERIALIZERS
# (Consolidated from disease_serializers.py)
# ========================================

"""
Serializers for Plant Disease Detection API

This module defines the serializers for converting disease detection model instances
to JSON and vice versa.
"""

from rest_framework import serializers
from .models import DiseasePrediction

class DiseasePredictionSerializer(serializers.ModelSerializer):
    """Serializer for disease prediction results"""
    
    class Meta:
        model = DiseasePrediction
        fields = '__all__'
        read_only_fields = ('timestamp',)

class DiseaseUploadSerializer(serializers.Serializer):
    """Serializer for disease image upload"""
    
    image = serializers.ImageField(required=True)
    
    def validate_image(self, value):
        """Validate uploaded image"""
        if not value:
            raise serializers.ValidationError("Image is required")
        
        # Check file size (max 5MB)
        if value.size > 5 * 1024 * 1024:
            raise serializers.ValidationError("Image size should not exceed 5MB")
        
        # Check file extension
        allowed_extensions = ['.jpg', '.jpeg', '.png']
        if not any(value.name.lower().endswith(ext) for ext in allowed_extensions):
            raise serializers.ValidationError("Only JPG, JPEG, and PNG images are allowed")
        
        return value

class DiseaseAdviceSerializer(serializers.Serializer):
    """Serializer for disease advice requests"""
    
    disease_name = serializers.CharField(required=True)
    severity = serializers.CharField(required=False, default="moderate")
    
    def validate_disease_name(self, value):
        """Validate disease name"""
        if not value:
            raise serializers.ValidationError("Disease name is required")
        return value
    
    def validate_severity(self, value):
        """Validate severity level"""
        allowed_severities = ["low", "moderate", "high"]
        if value not in allowed_severities:
            raise serializers.ValidationError(f"Severity must be one of: {allowed_severities}")
        return value


# ========================================
# RECOMMENDATION SERIALIZERS
# (Consolidated from recommendation_serializers.py)
# ========================================

"""
Serializers for crop and fertilizer recommendation data.
"""

from rest_framework import serializers
from .models import CropRecommendation, FertilizerRecommendation


class CropRecommendationSerializer(serializers.ModelSerializer):
    """Serializer for crop recommendation data."""
    
    class Meta:
        model = CropRecommendation
        fields = '__all__'
        read_only_fields = ('timestamp',)


class FertilizerRecommendationSerializer(serializers.ModelSerializer):
    """Serializer for fertilizer recommendation data."""
    
    class Meta:
        model = FertilizerRecommendation
        fields = '__all__'
        read_only_fields = ('timestamp',)


class CombinedRecommendationSerializer(serializers.Serializer):
    """Serializer for combined crop and fertilizer recommendation requests."""
    
    soil_data = serializers.DictField()
    weather_data = serializers.DictField()
    location = serializers.CharField()
    
    def validate_soil_data(self, value):
        """Validate soil data fields."""
        required_fields = ['nitrogen', 'phosphorus', 'potassium', 'ph']
        for field in required_fields:
            if field not in value:
                raise serializers.ValidationError(f"Missing required soil data field: {field}")
        return value


class RecommendationRequestSerializer(serializers.Serializer):
    """Serializer for recommendation requests."""
    
    farm_id = serializers.CharField(required=False, default='default_farm')
    soil_nitrogen = serializers.FloatField(min_value=0, max_value=500)
    soil_phosphorus = serializers.FloatField(min_value=0, max_value=500)
    soil_potassium = serializers.FloatField(min_value=0, max_value=500)
    soil_ph = serializers.FloatField(min_value=0, max_value=14)
    temperature = serializers.FloatField(min_value=-10, max_value=50)
    humidity = serializers.FloatField(min_value=0, max_value=100)
    rainfall = serializers.FloatField(min_value=0, max_value=500)
    location = serializers.CharField()
    season = serializers.CharField(required=False)
