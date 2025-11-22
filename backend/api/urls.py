from django.urls import path
from . import views, disease_views, recommendation_views
from . import recommendation_views_enhanced
from . import market_views

urlpatterns = [
    # Crop endpoints
    path('crops/', views.crop_list, name='crop-list'),
    
    # Prediction endpoints
    path('predict-yield/', views.predict_yield, name='predict-yield'),
    path('predict-pest/', views.predict_pest, name='predict-pest'),
    path('predictions/<int:crop_id>/', views.get_latest_prediction, name='get-latest-prediction'),
    
    # Market prediction endpoints
    path('predict-market-price/', market_views.predict_market_price_view, name='predict-market-price'),
    path('market-predictions/', market_views.get_market_predictions, name='get-market-predictions'),
    path('market-predictions/crop/<int:crop_id>/', market_views.get_crop_market_history, name='get-crop-market-history'),
    
    # Explanation and Q&A endpoints
    path('explain/', views.generate_explanation, name='generate-explanation'),
    path('ask/', views.ask_question, name='ask-question'),
    
    # Chatbot endpoints
    path('chatbot/', views.chatbot_query, name='chatbot-query'),
    path('chatbot/question/', views.chatbot_question, name='chatbot-question'),
    
    # Disease detection endpoints
    path('disease/predict/', disease_views.predict_disease, name='predict-disease'),
    path('disease/advice/', disease_views.get_disease_advice, name='disease-advice'),
    path('disease/test-upload/', disease_views.test_disease_upload, name='test-disease-upload'),
    path('disease/disease_model_info/', disease_views.disease_model_info, name='disease-model-info'),
    
    # Recommendation endpoints
    path('recommendations/combined/', recommendation_views.get_combined_recommendations, name='combined-recommendations'),
    path('recommendations/enhanced/', recommendation_views_enhanced.get_enhanced_recommendations, name='enhanced-recommendations'),
    path('recommendations/history/', recommendation_views.get_recommendation_history, name='recommendation-history'),
    
    # Report and chart endpoints
    path('reports/predictions/', views.get_prediction_report_data, name='prediction-report-data'),
    path('reports/diseases/', views.get_disease_report_data, name='disease-report-data'),
    path('reports/recommendations/', views.get_recommendation_report_data, name='recommendation-report-data'),
    path('reports/pests/', views.get_pest_report_data, name='pest-report-data'),
    path('reports/market/', views.get_market_report_data, name='market-report-data'),
    path('dashboard/statistics/', views.get_dashboard_statistics, name='dashboard-statistics'),
]