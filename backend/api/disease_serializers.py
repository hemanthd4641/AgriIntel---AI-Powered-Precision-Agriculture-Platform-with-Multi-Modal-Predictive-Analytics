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