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