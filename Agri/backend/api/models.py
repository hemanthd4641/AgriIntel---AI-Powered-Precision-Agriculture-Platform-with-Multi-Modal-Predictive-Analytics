"""
Models for the Smart Agriculture API.

This module defines the data models for storing crop data and prediction results.
"""

from django.db import models
from django.utils import timezone

class Crop(models.Model):
    """Model representing a crop type"""
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    
    def __str__(self):
        return self.name

class DiseasePrediction(models.Model):
    """Model representing a plant disease prediction"""
    image = models.ImageField(upload_to='disease_images/')
    predicted_disease = models.CharField(max_length=100)
    confidence_score = models.FloatField()
    timestamp = models.DateTimeField(default=timezone.now)
    
    def __str__(self):
        return f"{self.predicted_disease} ({self.confidence_score:.2f})"

class CropRecommendation(models.Model):
    """Model representing a crop recommendation"""
    recommended_crop = models.CharField(max_length=100)
    confidence_score = models.FloatField()
    soil_nitrogen = models.FloatField()
    soil_phosphorus = models.FloatField()
    soil_potassium = models.FloatField()
    soil_ph = models.FloatField()
    temperature = models.FloatField()
    humidity = models.FloatField()
    rainfall = models.FloatField()
    timestamp = models.DateTimeField(default=timezone.now)
    
    def __str__(self):
        return f"{self.recommended_crop}"

class FertilizerRecommendation(models.Model):
    """Model representing a fertilizer recommendation"""
    crop_recommendation = models.ForeignKey(CropRecommendation, on_delete=models.CASCADE)
    recommended_fertilizer = models.CharField(max_length=100)
    quantity_kg_per_ha = models.FloatField()
    timestamp = models.DateTimeField(default=timezone.now)
    
    def __str__(self):
        return f"{self.crop_recommendation.recommended_crop} - {self.recommended_fertilizer}"

class YieldPrediction(models.Model):
    """Model representing a crop yield prediction"""
    crop = models.ForeignKey(Crop, on_delete=models.CASCADE, null=True, blank=True)
    prediction_date = models.DateTimeField(default=timezone.now)
    predicted_yield_tonnes_per_ha = models.FloatField()
    confidence_score = models.FloatField(null=True, blank=True)
    
    def __str__(self):
        crop_name = self.crop.name if self.crop else "Unknown Crop"
        return f"{crop_name} - {self.predicted_yield_tonnes_per_ha} tonnes/ha"

class MarketPrediction(models.Model):
    """Model representing a crop market price prediction"""
    crop = models.ForeignKey(Crop, on_delete=models.CASCADE)
    prediction_date = models.DateTimeField(default=timezone.now)
    predicted_price_per_ton = models.FloatField()
    confidence_score = models.FloatField(null=True, blank=True)
    market_trend = models.CharField(max_length=20, choices=[
        ('bullish', 'Bullish'),
        ('bearish', 'Bearish'),
        ('neutral', 'Neutral')
    ])
    forecast_period_days = models.IntegerField(default=30)
    
    def __str__(self):
        return f"{self.crop.name} - ${self.predicted_price_per_ton}/ton"

class PestPrediction(models.Model):
    """Model representing a pest prediction based on manual data entry"""
    crop = models.ForeignKey(Crop, on_delete=models.CASCADE)
    region = models.CharField(max_length=100)
    season = models.CharField(max_length=20, choices=[
        ('Spring', 'Spring'),
        ('Summer', 'Summer'),
        ('Autumn', 'Autumn'),
        ('Winter', 'Winter')
    ])
    temperature = models.FloatField()
    humidity = models.FloatField()
    rainfall = models.FloatField()
    wind_speed = models.FloatField()
    soil_moisture = models.FloatField()
    soil_ph = models.FloatField()
    soil_type = models.CharField(max_length=50, choices=[
        ('Sandy', 'Sandy'),
        ('Clay', 'Clay'),
        ('Loam', 'Loam'),
        ('Silt', 'Silt'),
        ('Peat', 'Peat'),
        ('Chalky', 'Chalky')
    ])
    nitrogen = models.FloatField()
    phosphorus = models.FloatField()
    potassium = models.FloatField()
    weather_condition = models.CharField(max_length=20, choices=[
        ('Sunny', 'Sunny'),
        ('Cloudy', 'Cloudy'),
        ('Rainy', 'Rainy'),
        ('Humid', 'Humid'),
        ('Stormy', 'Stormy')
    ])
    irrigation_method = models.CharField(max_length=20, choices=[
        ('Drip', 'Drip'),
        ('Sprinkler', 'Sprinkler'),
        ('Furrow', 'Furrow'),
        ('Flood', 'Flood'),
        ('None', 'None')
    ])
    previous_crop = models.CharField(max_length=100)
    days_since_planting = models.IntegerField()
    plant_density = models.IntegerField()
    predicted_pest = models.CharField(max_length=100)
    pest_presence = models.BooleanField()
    severity = models.IntegerField()  # Scale of 1-10
    confidence_score = models.FloatField()
    recommended_treatment = models.TextField()
    prediction_date = models.DateTimeField(default=timezone.now)
    
    def __str__(self):
        return f"{self.crop.name} - {self.predicted_pest}"