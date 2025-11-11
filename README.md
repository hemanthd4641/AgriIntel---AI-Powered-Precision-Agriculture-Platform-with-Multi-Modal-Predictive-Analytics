# Smart Agriculture Platform

## Overview
The Smart Agriculture Platform is an AI-powered agricultural advisory system that helps farmers make informed decisions about crop selection, fertilizer application, yield prediction, and disease management. This comprehensive solution integrates machine learning, computer vision, and natural language processing to provide personalized agricultural insights.

## Key Features
- **Crop & Fertilizer Recommendations**: Get personalized recommendations based on soil conditions, weather, and environmental factors
- **Plant Disease Detection**: Upload leaf images to identify diseases and get treatment advice
- **Yield Prediction**: Predict crop yields using environmental data and satellite imagery analysis
- **Pest Prediction**: Predict likelihood of pest infestations based on environmental and crop data
- **Market Price Prediction**: Forecast agricultural commodity prices for informed selling decisions
- **AI Chatbot**: Get agricultural advice through a natural language interface with RAG-enhanced responses
- **Reporting & Analytics**: Visualize agricultural data with charts and comprehensive reports
- **Real-time Dashboard**: Monitor system usage with live statistics from the database

## Technology Stack
### Core Technologies
- **Frontend**: HTML5, CSS3, JavaScript, Chart.js
- **Backend**: Django REST Framework, SQLite
- **AI/ML**: Scikit-learn, PyTorch, OpenCV, HuggingFace Transformers
- **NLP**: LangChain, FAISS vector database, Retrieval-Augmented Generation (RAG)
- **Data Processing**: Pandas, NumPy, Rasterio, GeoPandas

### Machine Learning Models
- **Crop Yield Prediction**: Random Forest Regressor with ~90% accuracy
- **Plant Disease Detection**: Custom CNN with >90% accuracy on 10+ plant diseases
- **Pest Prediction**: Random Forest Classifier for pest infestation prediction
- **Crop Recommendation**: Random Forest Classifier with >85% accuracy
- **Fertilizer Recommendation**: Decision Tree Classifier with >80% accuracy
- **Market Price Prediction**: Random Forest regression models

### Natural Language Processing
- **LLM Integration**: HuggingFace Transformers with Phi-3-mini-4k-instruct model
- **Vector Database**: FAISS for similarity search and knowledge retrieval
- **RAG System**: Retrieval-Augmented Generation for context-aware responses
- **Embedding Models**: sentence-transformers/all-MiniLM-L6-v2 for document encoding

## System Architecture
The platform follows a modular architecture with clear separation of concerns:

1. **Frontend Layer**: User interface for data input and result visualization
2. **API Layer**: RESTful endpoints for feature access and data exchange
3. **Business Logic Layer**: ML models and processing pipelines
4. **Data Layer**: SQLite database for persistence and FAISS for knowledge storage
5. **AI Enhancement Layer**: LLM and RAG system for natural language explanations

## How It Works

### 1. Crop Yield Prediction
- Users input environmental data (region, soil type, rainfall, temperature, etc.)
- Data is preprocessed and fed to a trained Random Forest model
- The system predicts yield in tons/hectare with confidence scores
- LLM generates natural language explanations based on retrieved agricultural knowledge

### 2. Plant Disease Detection
- Farmers upload images of affected plant leaves
- Images are preprocessed and analyzed by a CNN model
- The system identifies diseases with confidence scores
- Treatment advice is generated using retrieved knowledge and LLM enhancement

### 3. Pest Prediction
- Users input environmental and crop data (temperature, humidity, rainfall, crop type, etc.)
- Data is preprocessed and fed to a trained Random Forest model
- The system predicts the likelihood of pest infestations
- Treatment recommendations are generated using retrieved knowledge and LLM enhancement

### 4. Crop and Fertilizer Recommendation
- Users provide soil conditions and environmental data
- ML models recommend optimal crops and fertilizers
- Detailed advice is generated using agricultural best practices and LLM enhancement

### 5. Market Price Prediction
- Based on yield predictions and market conditions
- Forecasts future commodity prices to help farmers decide when to sell
- Provides market trend analysis and confidence levels

### 6. AI Chatbot
- Natural language interface for agricultural queries
- Uses RAG system to retrieve relevant knowledge
- LLM generates context-aware responses with source attribution

## API Endpoints
- `POST /api/predict-yield/` - Predict crop yields
- `POST /api/disease/predict/` - Detect plant diseases from images
- `POST /api/pest/predict/` - Predict pest infestations based on environmental data
- `POST /api/recommendations/combined/` - Get crop and fertilizer recommendations
- `POST /api/predict-market-price/` - Forecast agricultural commodity prices
- `POST /api/chatbot/` - Handle chatbot queries
- `GET /api/dashboard/statistics/` - Retrieve dashboard statistics

## Setup Instructions
1. Install Python 3.8+
2. Install required packages: `pip install -r requirements.txt`
3. Run migrations: `cd backend && python manage.py migrate`
4. Start the server: `cd backend && python manage.py runserver`
5. Access the application at `http://127.0.0.1:8000`

## Enhanced AI Systems
- **Enhanced Yield Prediction**: More detailed predictions with natural language explanations ([crop_yield_prediction/README_ENHANCED.md](crop_yield_prediction/README_ENHANCED.md))
- **Enhanced Crop & Fertilizer Recommendations**: Detailed recommendations with insights and explanations ([crop_recommendation/README_ENHANCED.md](crop_recommendation/README_ENHANCED.md))

## Hugging Face API Integration
The system supports remote LLM usage through Hugging Face API:
- Configure your API token in the `.env` file
- Uses `microsoft/Phi-3-mini-4k-instruct` model by default
- Automatically falls back to local models if API is unavailable
- See [HUGGING_FACE_API_INTEGRATION.md](docs/HUGGING_FACE_API_INTEGRATION.md) for details

## Documentation
The project includes comprehensive documentation in various README files:
- [crop_yield_prediction/README_ENHANCED.md](crop_yield_prediction/README_ENHANCED.md) - Enhanced crop yield prediction system
- [crop_recommendation/README_ENHANCED.md](crop_recommendation/README_ENHANCED.md) - Enhanced crop and fertilizer recommendation system
- [market_prediction/README.md](market_prediction/README.md) - Market prediction module
- [plant_disease/README.md](plant_disease/README.md) - Plant disease detection system

Additional documentation can be found in the respective module directories.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For questions or support, please open an issue on the repository.