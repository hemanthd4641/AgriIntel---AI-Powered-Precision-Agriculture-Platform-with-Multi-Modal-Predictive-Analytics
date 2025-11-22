from rest_framework import serializers
from .models import Crop, DiseasePrediction, CropRecommendation, FertilizerRecommendation, YieldPrediction, MarketPrediction, PestPrediction

class CropSerializer(serializers.ModelSerializer):
    class Meta:
        model = Crop
        fields = '__all__'

class DiseasePredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = DiseasePrediction
        fields = '__all__'

class CropRecommendationSerializer(serializers.ModelSerializer):
    class Meta:
        model = CropRecommendation
        fields = '__all__'

class FertilizerRecommendationSerializer(serializers.ModelSerializer):
    class Meta:
        model = FertilizerRecommendation
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