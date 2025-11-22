# 🌾 AgriIntel - AI-Powered Precision Agriculture Platform

## 🌟 Overview

**AgriIntel** is a comprehensive, enterprise-grade precision agriculture platform that leverages cutting-edge artificial intelligence, machine learning, and computer vision to revolutionize modern farming practices. The platform integrates multiple predictive models with real-time AI assistance to provide farmers with actionable insights for optimal crop management, disease prevention, market timing, and resource optimization.

### 🎯 Mission Statement

To democratize access to advanced agricultural intelligence by providing farmers with AI-powered tools for data-driven decision making, ultimately increasing crop yields, reducing losses, and promoting sustainable farming practices.

### 🏆 Key Highlights

- **5 Core Prediction Modules** - Crop Yield, Disease Detection, Pest Prediction, Market Forecasting, Recommendations
- **38+ Disease Classes** - CNN-based plant disease detection with 85%+ accuracy
- **Real-time AI Chatbot** - Contextual agricultural advice powered by N8N webhook integration
- **Multi-modal ML Models** - XGBoost, CNN (ResNet18), and hybrid approaches
- **RESTful API** - Django REST Framework with comprehensive endpoints
- **Responsive UI** - Modern, tabbed interface with real-time visualizations
- **Explainable AI** - Transparent recommendations with detailed explanations

---

## 🚀 Key Features

### 1. **Crop Yield Prediction** 🌱
- Predict crop yield in tonnes per hectare based on environmental and agricultural factors
- Input parameters: Crop type, region, season, area, soil nutrients (N, P, K), rainfall, temperature, humidity
- Machine Learning: XGBoost Regressor with advanced preprocessing
- Real-time AI suggestions via N8N webhook integration
- Confidence scoring and historical comparison

### 2. **Plant Disease Detection** 🔬
- Image-based disease identification using deep CNN (ResNet18)
- Supports 38+ disease classes from PlantVillage dataset
- Upload plant leaf images for instant diagnosis
- Detailed treatment recommendations
- Confidence score for each prediction
- AI-powered contextual advice

### 3. **Crop & Fertilizer Recommendations** 💡
- Intelligent crop recommendation based on soil and environmental conditions
- Parameters: District, soil nutrients (N, P, K), pH, rainfall, temperature
- Fertilizer suggestions with quantity recommendations
- Supports 7-feature simplified preprocessing
- Districts covered: Kolhapur, Pune, Sangli, Satara, Solapur
- Enhanced metadata with crop insights

### 4. **Market Price Prediction** 📈
- Forecast crop market prices for optimal selling decisions
- Factors: Global demand, weather impact, supply/demand indices, inventory levels
- Market trend analysis (Bullish/Bearish/Neutral)
- Timing advice for selling
- Risk factor identification
- Price outlook with AI-enhanced explanations

### 5. **Pest Prediction & Management** 🦗
- Predict pest infestations based on environmental conditions
- Input: Crop, region, season, temperature, humidity, rainfall, wind speed, soil properties
- Severity assessment (Low/Medium/High/Critical)
- Integrated Pest Management (IPM) recommendations
- Monitoring advice and control strategies
- Potential crop damage estimation

### 6. **AI Agricultural Chatbot** 🤖
- Natural language query interface for farmers
- Powered by N8N webhook (https://projectu.app.n8n.cloud/webhook/agri-intel-chat)
- Context-aware responses based on prediction data
- Graceful fallback to rule-based advice
- Integrated across all feature modules

---

## 🏗️ Technical Architecture

### System Design Overview

AgriIntel follows a **3-tier architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                     PRESENTATION LAYER                       │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐           │
│  │   HTML5    │  │    CSS3    │  │ JavaScript │           │
│  │  Templates │  │   Styling  │  │  (Vanilla) │           │
│  └────────────┘  └────────────┘  └────────────┘           │
│         Responsive UI with Tabbed Navigation                │
└─────────────────────────────────────────────────────────────┘
                            ↕ HTTP/AJAX
┌─────────────────────────────────────────────────────────────┐
│                     APPLICATION LAYER                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            Django 5.2.6 REST Framework               │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │  │
│  │  │   Views     │  │ Serializers │  │   Models   │  │  │
│  │  │  (api/)     │  │  (api/)     │  │  (api/)    │  │  │
│  │  └─────────────┘  └─────────────┘  └────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  API Endpoints (RESTful):                                   │
│  • /api/predict-yield/                                      │
│  • /api/disease/predict/                                    │
│  • /api/recommendations/combined/                           │
│  • /api/predict-market-price/                               │
│  • /api/predict-pest/                                       │
└─────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────┐
│                      DATA & ML LAYER                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   XGBoost    │  │  CNN Models  │  │ Preprocessors│     │
│  │   Models     │  │  (ResNet18)  │  │   (Joblib)   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   SQLite     │  │  N8N Webhook │  │  Media Files │     │
│  │   Database   │  │   (AI Chat)  │  │   Storage    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 System Components

### Backend Components

#### 1. **Django REST API** (`backend/api/`)
- **views.py** (4992 lines): Consolidated API endpoints for all features
- **models.py**: Database models for Crop, YieldPrediction, DiseasePrediction, PestPrediction, MarketPrediction, CropRecommendation, FertilizerRecommendation
- **serializers.py**: Data serialization for API requests/responses
- **urls.py**: URL routing for all API endpoints

Key Endpoints:
```python
# Crop & Yield Endpoints
GET  /api/crops/
POST /api/predict-yield/

# Disease Detection
POST /api/disease/predict/
POST /api/disease/advice/
GET  /api/disease/disease_model_info/

# Recommendations
POST /api/recommendations/combined/
POST /api/recommendations/enhanced/
GET  /api/recommendations/history/

# Market Predictions
POST /api/predict-market-price/
GET  /api/market-predictions/
GET  /api/market-predictions/crop/<int:crop_id>/

# Pest Predictions
POST /api/predict-pest/
GET  /api/reports/pests/
```

#### 2. **Machine Learning Pipeline**

**Crop Yield Prediction** (`crop_yield/`)
- Model: `yield_model_enhanced.pkl` (XGBoost)
- Features: Crop, Region, Season, Area, Soil_N, Soil_P, Soil_K, Rainfall, Temperature, Humidity
- Preprocessing: Standard scaling, label encoding
- Output: Predicted yield (tonnes/ha), confidence score

**Disease Detection** (`plant_disease/`)
- Model: `plant_disease_model (1).pth` (ResNet18 CNN)
- Input: 224x224 RGB images
- Architecture: Transfer learning on ResNet18
- Classes: 38 disease categories
- Output: Disease class, confidence score, treatment advice

**Crop Recommendation** (`crop_fertilizer_recommendation/`)
- Model: `crop_model.pkl` (XGBoost Classifier)
- Features: District (encoded 0-4), N, P, K, pH, Rainfall, Temperature
- Simplified Preprocessing Fallback: Min-max scaling (0-1)
- Output: Recommended crop, confidence score
- Districts: Kolhapur(0), Pune(1), Sangli(2), Satara(3), Solapur(4)

**Fertilizer Recommendation** (`crop_fertilizer_recommendation/`)
- Model: `fertilizer_model.pkl` (XGBoost Classifier)
- Features: Soil nutrients, pH, crop type
- Output: Fertilizer type, quantity (kg/ha)

**Market Price Prediction** (`market_price/`)
- Model: `market_model.pkl` (XGBoost Regressor)
- Features: Crop, region, season, yield, global demand, weather impact, supply/demand indices
- Output: Predicted price, market trend, timing advice

**Pest Prediction** (`pest/`)
- Model: `pest_model.pkl` (XGBoost Classifier)
- Features: Crop, region, season, temperature, humidity, rainfall, wind speed, soil properties
- Output: Pest type, severity, IPM recommendations

#### 3. **AI Integration**

**N8N Webhook Integration**
- Endpoint: `https://projectu.app.n8n.cloud/webhook/agri-intel-chat`
- Method: POST
- Payload: JSON with prediction context and query
- Response: AI-generated suggestions and advice
- Timeout: 10 seconds with fallback


### Training Infrastructure

**Training Scripts** (`*/training/`)
- `train_crop_model_enhanced.py`
- `train_fertilizer_model_enhanced.py`
- `train_yield_model_enhanced.py`
- `train_market_model.py`
- `train_pest_model.py`
- `train_32class_disease_model.py`

**Preprocessing Modules**
- `crop_fertilizer_recommendation/preprocessing/recommendation_preprocessor.py`
- `crop_yield/preprocessing/yield_preprocessor.py`
- `market_price/preprocessing/market_preprocessor.py`
- `plant_disease/preprocessing/disease_preprocessor.py`

## 📦 Installation

### Prerequisites

- Python 3.8+
- pip
- virtualenv (recommended)
- CUDA-capable GPU (optional, for faster disease detection)

### Step 1: Clone Repository

```bash
git clone https://github.com/hemanthd4641/AgriIntel---AI-Powered-Precision-Agriculture-Platform-with-Multi-Modal-Predictive-Analytics.git
cd AgriIntel
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
cd Agri
pip install -r requirements.txt
```

**Key Dependencies**:
```
Django>=5.2.6
djangorestframework>=3.12.0
torch>=1.9.0
torchvision>=0.10.0
xgboost>=1.5.0
scikit-learn>=1.0.0
numpy>=1.21.0
pandas>=1.3.0
pillow>=8.3.0
opencv-python>=4.5.0
joblib>=1.1.0
requests>=2.25.0
```

### Step 4: Database Setup

```bash
cd backend
python manage.py makemigrations
python manage.py migrate
```
Train models from scratch
```bash
# Crop recommendation
python crop_fertilizer_recommendation/training/train_crop_model_enhanced.py

# Yield prediction
python crop_yield/training/train_yield_model_enhanced.py

# Disease detection (requires GPU)
python plant_disease/training/train_32class_disease_model.py

# Market prediction
python market_price/training/train_market_model.py

# Pest prediction
python pest/train_pest_model.py
```

### Step 8: Run Development Server

```bash
cd backend
python manage.py runserver
```

Access the application at: `http://localhost:8000`


---



## 🛠️ Technology Stack

### Backend
- **Framework**: Django 5.2.6
- **API**: Django REST Framework 3.12+
- **Database**: SQLite (development), PostgreSQL (production-ready)
- **Machine Learning**: 
  - PyTorch 1.9+ (CNN models)
  - XGBoost 1.5+ (gradient boosting)
  - scikit-learn 1.0+ (preprocessing, metrics)
- **Image Processing**: OpenCV, Pillow
- **Data Processing**: NumPy, Pandas
- **Model Serialization**: Joblib

### Frontend
- **HTML5**: Semantic markup
- **CSS3**: Flexbox, Grid, Gradients, Animations
- **JavaScript**: Vanilla ES6+ (no frameworks)
- **Charting**: Chart.js
- **AJAX**: Fetch API for asynchronous requests

### AI Integration
- **N8N Webhook**: External AI service powered by Google Gemini 2.0 Flash LLM
- **Multi-Tool Agent**: 6 specialized tools connecting to ML/DL models
- **Workflow File**: Complete N8N workflow included in `n8n workflow/` directory
- **Fallback Logic**: Rule-based advice when webhook unavailable

### DevOps & Tools
- **Version Control**: Git, GitHub
- **Package Management**: pip, requirements.txt
- **Testing**: Django TestCase, requests
- **Deployment**: Compatible with Heroku, AWS, Azure

---

## 📁 Project Structure

```
AgriIntel/
├── Agri/
│   ├── backend/                          # Django backend
│   │   ├── api/                          # REST API app
│   │   │   ├── models.py                 # Database models
│   │   │   ├── views.py                  # API endpoints (4992 lines)
│   │   │   ├── serializers.py            # Data serialization
│   │   │   ├── urls.py                   # URL routing
│   │   │   └── migrations/               # Database migrations
│   │   ├── smart_agriculture_backend/    # Django project settings
│   │   │   ├── settings.py               # Configuration
│   │   │   ├── urls.py                   # Root URL config
│   │   │   └── wsgi.py                   # WSGI entry point
│   │   ├── db.sqlite3                    # SQLite database
│   │   └── manage.py                     # Django management
│   │
│   ├── frontend/                         # Frontend assets
│   │   ├── templates/
│   │   │   ├── index.html                # Main application (949 lines)
│   │   │   └── chatbot.html              # Chatbot interface
│   │   └── static/
│   │       ├── css/
│   │       │   ├── style.css             # Main styles (1251+ lines)
│   │       │   └── chatbot.css           # Chatbot styles
│   │       └── js/
│   │           ├── main.js               # Core UI logic (1386 lines)
│   │           ├── predictions.js        # Yield prediction UI
│   │           ├── recommendations.js    # Recommendation UI
│   │           ├── market_predictions.js # Market UI
│   │           ├── pest_predictions.js   # Pest UI
│   │           └── chatbot.js            # Chatbot logic
│   │
│   ├── crop_fertilizer_recommendation/   # Crop & fertilizer module
│   │   ├── saved_models/
│   │   │   ├── crop_model.pkl            # XGBoost crop model
│   │   │   ├── crop_preprocessor.pkl     # Preprocessor
│   │   │   ├── fertilizer_model.pkl      # Fertilizer model
│   │   │   └── fertilizer_preprocessor.pkl
│   │   ├── training/
│   │   │   ├── train_crop_model_enhanced.py
│   │   │   └── train_fertilizer_model_enhanced.py
│   │   ├── preprocessing/
│   │   │   └── recommendation_preprocessor.py
│   │   └── predict_crop_enhanced.py
│   │
│   ├── crop_yield/                       # Yield prediction module
│   │   ├── yield_model_enhanced.pkl
│   │   ├── yield_preprocessor.pkl
│   │   ├── training/
│   │   │   └── train_yield_model_enhanced.py
│   │   ├── preprocessing/
│   │   │   └── yield_preprocessor.py
│   │   └── predict_yield_enhanced.py
│   │
│   ├── plant_disease/                    # Disease detection module
│   │   ├── disease_detection_model.py    # Model loader
│   │   ├── preprocessing/
│   │   │   ├── disease_preprocessor.py
│   │   │   └── image_preprocessing.py
│   │   └── training/
│   │       ├── train_32class_disease_model.py
│   │       └── complete_plant_disease_colab.py
│   │
│   ├── market_price/                     # Market prediction module
│   │   ├── predict_market.py
│   │   ├── training/
│   │   │   └── train_market_model.py
│   │   └── preprocessing/
│   │       └── market_preprocessor.py
│   │
│   ├── pest/                             # Pest prediction module
│   │   ├── train_pest_model.py
│   │   └── generate_pest_dataset.py
│   │
│   ├── models/                           # Trained models storage
│   │   ├── plant_disease/
│   │   │   ├── plant_disease_model (1).pth  # ResNet18 weights
│   │   │   └── labels.txt
│   │   ├── crop_recommendation/
│   │   ├── crop_yield_prediction/
│   │   └── market_prediction/
│   │
│   ├── datasets/                         # Training datasets
│   │   ├── crop_recommendation/
│   │   ├── crop_yield_prediction/
│   │   ├── market_prediction/
│   │   └── pest_prediction/
│   │
│   ├── media/                            # User uploads
│   │   └── temp/
│   │
│   ├── requirements.txt                  # Python dependencies
│   ├── test_ui_ai_integration.py        # Comprehensive tests
│   └── README.md                         # Module documentation
│
├── n8n workflow/                         # N8N AI workflow configuration
│   └── Agricultural AI Assistant with Multi-Model Prediction and Advisory System with webhook.json
│
└── scripts/                              # Utility scripts
    ├── ingest_documents.py
    ├── label_loader.py
    └── test_*.py                         # Various test scripts
```


#### Importing the N8N Workflow

1. **Access N8N Instance**: Log into your N8N account at https://app.n8n.cloud
2. **Import Workflow**: 
   - Click on "Workflows" → "Import from File"
   - Upload: `n8n workflow/Agricultural AI Assistant with Multi-Model Prediction and Advisory System with webhook.json`
3. **Configure Credentials**:
   - Set up Google Gemini API credentials
   - Update Django API base URL if needed (default: `http://localhost:8000`)
4. **Activate Workflow**: Enable the workflow to start receiving webhook requests
5. **Test Webhook**: Send a POST request to your webhook URL to verify integration

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Coding Standards

- Follow PEP 8 for Python code
- Use meaningful variable names
- Add docstrings to functions
- Write unit tests for new features
- Update README for major changes

### Areas for Contribution

- 🆕 **New Features**: Additional crop types, new prediction models
- 🐛 **Bug Fixes**: Report and fix bugs
- 📚 **Documentation**: Improve API docs, tutorials
- 🎨 **UI/UX**: Enhance frontend design
- ⚡ **Performance**: Optimize model loading, database queries
- 🧪 **Testing**: Add more test coverage

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025 Hemanth D

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📞 Contact & Support

### Developer
**Hemanth D**  
GitHub: [@hemanthd4641](https://github.com/hemanthd4641)  
Email: hemanthd4641@gmail.com

### Repository
🔗 [AgriIntel GitHub Repository](https://github.com/hemanthd4641/AgriIntel---AI-Powered-Precision-Agriculture-Platform-with-Multi-Modal-Predictive-Analytics)

---


---

<div align="center">

**Made with ❤️ for Farmers | Powered by AI | Built with Django & PyTorch**

⭐ **Star this repo if you find it helpful!** ⭐

</div>
