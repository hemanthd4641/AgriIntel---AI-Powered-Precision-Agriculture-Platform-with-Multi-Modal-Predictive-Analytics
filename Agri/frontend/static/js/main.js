// Smart Agriculture Platform JavaScript

// API base URL (in a real app, this would be your Django backend)
const API_BASE_URL = 'http://127.0.0.1:8000/api';

// DOM elements
const recommendationsList = document.getElementById('recommendations-list');

// Voice recognition variables

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('Smart Agriculture Platform loaded');
    
    // Set up event listeners
    setupEventListeners();
    
    // Load initial data
    loadDashboardData();
    
    // Initialize all sections that might be active
    initializeActiveSections();
});

// Set up event listeners
function setupEventListeners() {
    // Navigation links
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const target = this.getAttribute('data-target');
            showSection(target);
            
            // Update active class on navigation
            navLinks.forEach(navLink => navLink.classList.remove('active'));
            this.classList.add('active');
            
            // Scroll to chatbot section if it's the chatbot
            if (target === 'chatbot-section') {
                setTimeout(() => {
                    const chatbotSection = document.getElementById('chatbot-section');
                    if (chatbotSection) {
                        chatbotSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                        // Focus on input after scrolling
                        setTimeout(() => {
                            const messageInput = document.getElementById('messageInput');
                            if (messageInput) messageInput.focus();
                        }, 500);
                    }
                }, 100);
            }
        });
    });
    
    // Feature buttons (from dashboard)
    const featureButtons = document.querySelectorAll('.feature-btn');
    featureButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            const target = this.getAttribute('data-target');
            showSection(target);
            
            // Update active class on navigation
            document.querySelectorAll('.nav-link').forEach(navLink => navLink.classList.remove('active'));
            document.querySelector(`.nav-link[data-target="${target}"]`).classList.add('active');
        });
    });
}

// Initialize sections that might be active on page load
function initializeActiveSections() {
    // Check which sections are active and initialize them
    const activeSections = document.querySelectorAll('.feature-section.active');
    activeSections.forEach(section => {
        const sectionId = section.id;
        initializeSection(sectionId);
    });
}

// Initialize a specific section
function initializeSection(sectionId) {
    switch(sectionId) {
        case 'predictions':
            initializePredictionSection();
            break;
        case 'disease-detection':
            initializeDiseaseSection();
            break;
        case 'reports-charts':
            initializeReportsSection();
            break;
        case 'market-predictions':
            initializeMarketPredictionSection();
            break;
        case 'recommendations':
            initializeRecommendationSection();
            break;
    }
}

// Initialize recommendation section
function initializeRecommendationSection() {
    const recommendationForm = document.getElementById('recommendation-form');
    const recommendationResults = document.getElementById('recommendation-results');
    const recommendationOutput = document.getElementById('recommendation-output');
    
    if (recommendationForm && !recommendationForm.dataset.initialized) {
        recommendationForm.addEventListener('submit', function(e) {
            e.preventDefault();
            e.stopPropagation();
            // Directly call the function that should be available from recommendations.js
            // If that fails, try to initialize it directly here
            try {
                if (typeof getRecommendations === 'function') {
                    getRecommendations();
                } else {
                    // Fallback: implement the functionality directly
                    handleRecommendationSubmit(e);
                }
            } catch (error) {
                console.error('Error in recommendation submission:', error);
                handleRecommendationSubmit(e);
            }
            return false;
        });
        recommendationForm.dataset.initialized = 'true';
        console.log('Recommendation form initialized');
    }
}

// Handle recommendation form submission directly
async function handleRecommendationSubmit(e) {
    console.log('Handling recommendation submission directly');
    
    // Get DOM elements
    const recommendationResults = document.getElementById('recommendation-results');
    const recommendationOutput = document.getElementById('recommendation-output');
    
    // Get form values
    const soilNitrogen = document.getElementById('soil-nitrogen').value;
    const soilPhosphorus = document.getElementById('soil-phosphorus').value;
    const soilPotassium = document.getElementById('soil-potassium').value;
    const soilPh = document.getElementById('soil-ph').value;
    const temperature = document.getElementById('temperature').value;
    const humidity = document.getElementById('humidity').value;
    const rainfall = document.getElementById('rainfall').value;
    const location = document.getElementById('location').value;
    const season = document.getElementById('season').value;
    
    console.log('Form values:', {
        soilNitrogen, soilPhosphorus, soilPotassium, soilPh,
        temperature, humidity, rainfall, location, season
    });
    
    // Validate required fields
    if (!soilNitrogen || !soilPhosphorus || !soilPotassium || !soilPh ||
        !temperature || !humidity || !rainfall || !location || !season) {
        alert('Please fill in all fields');
        return;
    }
    
    // Show loading state
    if (recommendationOutput) {
        recommendationOutput.innerHTML = `
            <div class="loading">
                <div class="spinner"></div>
                <p>Analyzing soil conditions and environmental factors...</p>
                <p>Generating crop and fertilizer recommendations...</p>
            </div>
        `;
    }
    if (recommendationResults) {
        recommendationResults.style.display = 'block';
    }
    
    // Create data object for API request
    const requestData = {
        soil_nitrogen: parseFloat(soilNitrogen),
        soil_phosphorus: parseFloat(soilPhosphorus),
        soil_potassium: parseFloat(soilPotassium),
        soil_ph: parseFloat(soilPh),
        temperature: parseFloat(temperature),
        humidity: parseFloat(humidity),
        rainfall: parseFloat(rainfall),
        location: location,
        season: season
    };
    
    // Get CSRF token
    const meta = document.querySelector('meta[name="csrf-token"]');
    const csrfToken = meta ? meta.getAttribute('content') : '';
    
    console.log('Request data:', requestData);
    
    try {
        // Send request to backend
        const response = await fetch('http://127.0.0.1:8000/api/recommendations/combined/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken
            },
            body: JSON.stringify(requestData)
        });
        
        console.log('Response status:', response.status);
        
        const data = await response.json();
        console.log('Response data:', data);
        
        if (response.ok) {
            // Display results directly in main.js
            displayRecommendationsInMain(data);
        } else {
            if (recommendationOutput) {
                recommendationOutput.innerHTML = `
                    <div class="error-message">
                        <h4>Error</h4>
                        <p>${data.error || 'Unknown error occurred'}</p>
                    </div>
                `;
            }
        }
    } catch (error) {
        console.error('Error getting recommendations:', error);
        if (recommendationOutput) {
            recommendationOutput.innerHTML = `
                <div class="error-message">
                    <h4>Error</h4>
                    <p>Error getting recommendations: ${error.message}. Please try again.</p>
                </div>
            `;
        }
    }
}

// Display recommendation results directly in main.js
function displayRecommendationsInMain(data) {
    const recommendationOutput = document.getElementById('recommendation-output');
    const recommendationResults = document.getElementById('recommendation-results');
    
    if (!data.recommendations || data.recommendations.length === 0) {
        recommendationOutput.innerHTML = '<p>No recommendations found. Please check your input values and try again.</p>';
        return;
    }
    
    // Create enhanced results HTML with tabbed interface
    let html = `
        <div class="prediction-result">
            <h4>Crop & Fertilizer Recommendations</h4>
    `;
    
    // Add top recommendation highlight
    const topRec = data.recommendations[0];
    const confidencePercent = (topRec.confidence * 100).toFixed(1);
    
    html += `
            <div class="result-grid">
                <div class="result-item">
                    <span class="result-label">Top Recommendation:</span>
                    <span class="result-value">${topRec.crop}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Confidence:</span>
                    <span class="result-value">${confidencePercent}%</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Recommended Fertilizer:</span>
                    <span class="result-value">${topRec.fertilizer}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Quantity:</span>
                    <span class="result-value">${topRec.quantity_kg_per_ha.toFixed(1)} kg/ha</span>
                </div>
            </div>
    `;
    
    // Add enhanced insights section with tabbed interface
    html += `
            <div class="insights-section">
                <h5>AI-Powered Recommendations Insights</h5>
                <div class="insights-tabs">
                    <button class="tab-button active" onclick="switchRecommendationTab('recommendations-overview')">Overview</button>
                    <button class="tab-button" onclick="switchRecommendationTab('recommendations-details')">Detailed Recommendations</button>
                    <button class="tab-button" onclick="switchRecommendationTab('recommendations-advice')">Expert Advice</button>
                </div>
                <div class="tab-content">
                    <div id="recommendations-overview" class="tab-pane active">
                        <div class="insight-card">
                            <h6>Recommendation Summary</h6>
                            <p>Based on your soil conditions and environmental factors, ${topRec.crop} is the top recommended crop with ${confidencePercent}% confidence.</p>
                            <div class="risk-assessment">
                                <span>Recommendation Strength: ${getConfidenceLevel(confidencePercent)}</span>
                                <span>Confidence: ${confidencePercent}%</span>
                            </div>
                        </div>
                        <div class="insight-card">
                            <h6>Key Environmental Factors</h6>
                            <ul>
                                <li>Soil nutrients: ${getNutrientStatus(topRec)}</li>
                                <li>Temperature conditions: ${getTemperatureStatus()}</li>
                                <li>Moisture levels: ${getMoistureStatus()}</li>
                            </ul>
                        </div>
                    </div>
                    <div id="recommendations-details" class="tab-pane">
    `;
    
    // Add detailed recommendations for all crops with dynamic confidence bars
    data.recommendations.forEach((rec, index) => {
        const recConfidence = (rec.confidence * 100).toFixed(1);
        const confidenceWidth = Math.min(recConfidence, 100); // Cap at 100%
        
        html += `
                        <div class="insight-card">
                            <h6>${index + 1}. ${rec.crop}</h6>
                            <div class="confidence-bar">
                                <div class="confidence-level" style="width: ${confidenceWidth}%; background-color: ${getConfidenceColor(recConfidence)}"></div>
                            </div>
                            <p>Confidence: ${recConfidence}%</p>
                            <p><strong>Recommended Fertilizer:</strong> ${rec.fertilizer}</p>
                            <p><strong>Quantity:</strong> ${rec.quantity_kg_per_ha.toFixed(1)} kg/ha</p>
                            ${rec.advice ? `<p><strong>Advice:</strong> ${formatRecommendationContent(rec.advice).substring(0, 100)}${rec.advice.length > 100 ? '...' : ''}</p>` : ''}
                        </div>
        `;
    });
    
    html += `
                    </div>
                    <div id="recommendations-advice" class="tab-pane">
                        <div class="insight-card">
                            <h6>Expert Recommendations</h6>
    `;
    
    // Add advice for top recommendations
    if (topRec.advice) {
        html += `
                            <div class="recommendations-content">
                                <p>${formatRecommendationContent(topRec.advice)}</p>
                            </div>
        `;
    } else {
        html += `
                            <p>Based on the analysis of your soil conditions and environmental factors, here are some general recommendations:</p>
                            <ul>
                                <li>Maintain soil pH between 6.0-7.0 for optimal nutrient uptake</li>
                                <li>Ensure adequate drainage to prevent waterlogging</li>
                                <li>Consider crop rotation to maintain soil health</li>
                                <li>Apply fertilizers based on soil test results</li>
                            </ul>
        `;
    }
    
    html += `
                        </div>
                    </div>
                </div>
            </div>
    `;
    
    // Add input data section with dynamic analysis
    html += `
            <div class="input-data-section">
                <h5>Input Data Analysis</h5>
                <div class="data-grid">
                    ${generateDynamicDataAnalysis()}
                </div>
            </div>
        </div>
    `;
    
    recommendationOutput.innerHTML = html;
    // Ensure the results section is visible
    recommendationResults.style.display = 'block';
    
    // Scroll to results
    recommendationResults.scrollIntoView({ behavior: 'smooth' });
}

// Helper functions for display
function getConfidenceLevel(confidencePercent) {
    const confidence = parseFloat(confidencePercent);
    if (confidence >= 90) return "Very Strong";
    if (confidence >= 75) return "Strong";
    if (confidence >= 60) return "Moderate";
    if (confidence >= 40) return "Weak";
    return "Very Weak";
}

function getConfidenceColor(confidencePercent) {
    const confidence = parseFloat(confidencePercent);
    if (confidence >= 80) return "#4caf50"; // Green
    if (confidence >= 60) return "#ff9800"; // Orange
    return "#f44336"; // Red
}

function formatRecommendationContent(content) {
    if (!content) return '';
    return content
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>')
        .replace(/<br><br>/g, '</p><p>')
        .replace(/^<br>/, '')
        .replace(/^(.*)$/, '<p>$1</p>');
}

function getNutrientStatus(rec) {
    const n = rec.input_data?.Nitrogen || rec.soil_nitrogen || 0;
    const p = rec.input_data?.Phosphorus || rec.soil_phosphorus || 0;
    const k = rec.input_data?.Potassium || rec.soil_potassium || 0;
    
    let status = [];
    if (n < 50) status.push("Low N");
    else if (n > 150) status.push("High N");
    else status.push("Adequate N");
    
    if (p < 20) status.push("Low P");
    else if (p > 80) status.push("High P");
    else status.push("Adequate P");
    
    if (k < 50) status.push("Low K");
    else if (k > 200) status.push("High K");
    else status.push("Adequate K");
    
    return status.join(", ");
}

function getTemperatureStatus() {
    const temp = parseFloat(document.getElementById('temperature').value);
    if (temp < 10) return "Cool climate";
    if (temp > 30) return "Warm climate";
    return "Moderate climate";
}

function getMoistureStatus() {
    const humidity = parseFloat(document.getElementById('humidity').value);
    const rainfall = parseFloat(document.getElementById('rainfall').value);
    
    let status = [];
    if (humidity < 40) status.push("Low humidity");
    else if (humidity > 70) status.push("High humidity");
    else status.push("Moderate humidity");
    
    if (rainfall < 50) status.push("Low rainfall");
    else if (rainfall > 150) status.push("High rainfall");
    else status.push("Moderate rainfall");
    
    return status.join(", ");
}

function generateDynamicDataAnalysis() {
    const nitrogen = parseFloat(document.getElementById('soil-nitrogen').value);
    const phosphorus = parseFloat(document.getElementById('soil-phosphorus').value);
    const potassium = parseFloat(document.getElementById('soil-potassium').value);
    const ph = parseFloat(document.getElementById('soil-ph').value);
    const temperature = parseFloat(document.getElementById('temperature').value);
    const humidity = parseFloat(document.getElementById('humidity').value);
    const rainfall = parseFloat(document.getElementById('rainfall').value);
    const location = document.getElementById('location').value;
    const season = document.getElementById('season').value;
    
    let html = '';
    
    // Add soil analysis
    html += `<div class="data-item"><strong>Soil Analysis:</strong> ${getSoilAnalysis(nitrogen, phosphorus, potassium, ph)}</div>`;
    
    // Add climate analysis
    html += `<div class="data-item"><strong>Climate Suitability:</strong> ${getClimateAnalysis(temperature, humidity, rainfall)}</div>`;
    
    // Add location and season info
    html += `<div class="data-item"><strong>Location:</strong> ${location}</div>`;
    html += `<div class="data-item"><strong>Season:</strong> ${season}</div>`;
    
    return html;
}

function getSoilAnalysis(nitrogen, phosphorus, potassium, ph) {
    let analysis = [];
    
    // Nitrogen analysis
    if (nitrogen < 50) {
        analysis.push("Low N");
    } else if (nitrogen > 150) {
        analysis.push("High N");
    } else {
        analysis.push("Adequate N");
    }
    
    // Phosphorus analysis
    if (phosphorus < 20) {
        analysis.push("Low P");
    } else if (phosphorus > 80) {
        analysis.push("High P");
    } else {
        analysis.push("Adequate P");
    }
    
    // Potassium analysis
    if (potassium < 50) {
        analysis.push("Low K");
    } else if (potassium > 200) {
        analysis.push("High K");
    } else {
        analysis.push("Adequate K");
    }
    
    // pH analysis
    if (ph < 6.0) {
        analysis.push("Acidic pH");
    } else if (ph > 7.5) {
        analysis.push("Alkaline pH");
    } else {
        analysis.push("Optimal pH");
    }
    
    return analysis.join(", ");
}

function getClimateAnalysis(temperature, humidity, rainfall) {
    let analysis = [];
    
    // Temperature analysis
    if (temperature < 10) {
        analysis.push("Cool climate");
    } else if (temperature > 30) {
        analysis.push("Warm climate");
    } else {
        analysis.push("Moderate climate");
    }
    
    // Humidity analysis
    if (humidity < 40) {
        analysis.push("Low humidity");
    } else if (humidity > 70) {
        analysis.push("High humidity");
    } else {
        analysis.push("Moderate humidity");
    }
    
    // Rainfall analysis
    if (rainfall < 50) {
        analysis.push("Low rainfall");
    } else if (rainfall > 150) {
        analysis.push("High rainfall");
    } else {
        analysis.push("Moderate rainfall");
    }
    
    return analysis.join(", ");
}

// Switch between tabs in the recommendations section
window.switchRecommendationTab = function(tabName) {
    // Hide all tab panes
    const tabPanes = document.querySelectorAll('.tab-pane');
    tabPanes.forEach(pane => pane.classList.remove('active'));
    
    // Remove active class from all tab buttons
    const tabButtons = document.querySelectorAll('.tab-button');
    tabButtons.forEach(button => button.classList.remove('active'));
    
    // Show the selected tab pane
    const activePane = document.getElementById(tabName);
    if (activePane) {
        activePane.classList.add('active');
    }
    
    // Add active class to the clicked button
    event.currentTarget.classList.add('active');
};

// Initialize market prediction section
function initializeMarketPredictionSection() {
    const marketForm = document.getElementById('market-prediction-form');
    const marketResults = document.getElementById('market-prediction-results');
    const marketOutput = document.getElementById('market-prediction-output');
    
    if (marketForm && !marketForm.dataset.initialized) {
        // The form is already handled by market_predictions.js
        marketForm.dataset.initialized = 'true';
        console.log('Market prediction form initialized');
    }
}

// Show a specific section and hide others
function showSection(sectionId) {
    // Hide all sections
    document.querySelectorAll('.feature-section').forEach(section => {
        section.classList.remove('active');
    });
    
    // Show the target section
    const targetSection = document.getElementById(sectionId);
    if (targetSection) {
        targetSection.classList.add('active');
        // Initialize the section if needed
        initializeSection(sectionId);
    }
    
    // Scroll to top
    window.scrollTo(0, 0);
}

// Initialize prediction section
function initializePredictionSection() {
    const predictionForm = document.getElementById('prediction-form');
    const predictionResults = document.getElementById('prediction-results');
    const predictionOutput = document.getElementById('prediction-output');
    
    if (predictionForm && !predictionForm.dataset.initialized) {
        predictionForm.addEventListener('submit', function(e) {
            e.preventDefault();
            e.stopPropagation();
            predictYield();
            return false;
        });
        predictionForm.dataset.initialized = 'true';
        console.log('Prediction form initialized');
    }
}

// Initialize disease section
function initializeDiseaseSection() {
    const diseaseForm = document.getElementById('disease-form');
    const diseaseResults = document.getElementById('disease-results');
    const diseaseOutput = document.getElementById('disease-output');
    
    if (diseaseForm && !diseaseForm.dataset.initialized) {
        diseaseForm.addEventListener('submit', function(e) {
            e.preventDefault();
            e.stopPropagation();
            analyzeDisease();
            return false;
        });
        diseaseForm.dataset.initialized = 'true';
        console.log('Disease form initialized');
    }
}

// Initialize reports section
function initializeReportsSection() {
    const generateReportBtn = document.getElementById('generate-report');
    const downloadPdfBtn = document.getElementById('download-pdf');
    const reportOutput = document.getElementById('report-output');
    
    if (generateReportBtn && !generateReportBtn.dataset.initialized) {
        generateReportBtn.addEventListener('click', generateReport);
        generateReportBtn.dataset.initialized = 'true';
        console.log('Reports button initialized');
    }
}

// Predict crop yield from manual data input
async function predictYield() {
    console.log('Predict Yield function called');
    
    // Get DOM elements
    const predictionResults = document.getElementById('prediction-results');
    const predictionOutput = document.getElementById('prediction-output');
    
    // Get form values
    const region = document.getElementById('region').value;
    const soilType = document.getElementById('soil-type').value;
    const crop = document.getElementById('crop').value;
    const rainfallMm = document.getElementById('rainfall-mm').value;
    const temperatureCelsius = document.getElementById('temperature-celsius').value;
    const fertilizerUsed = document.getElementById('fertilizer-used').value;
    const irrigationUsed = document.getElementById('irrigation-used').value;
    const weatherCondition = document.getElementById('weather-condition').value;
    const daysToHarvest = document.getElementById('days-to-harvest').value;
    
    console.log('Form values:', {
        region, soilType, crop, rainfallMm, temperatureCelsius, 
        fertilizerUsed, irrigationUsed, weatherCondition, daysToHarvest
    });
    
    // Validate required fields
    if (!region || !soilType || !crop || !rainfallMm || !temperatureCelsius || 
        !fertilizerUsed || !irrigationUsed || !weatherCondition || !daysToHarvest) {
        alert('Please fill in all fields');
        return;
    }
    
    // Show loading state
    predictionOutput.innerHTML = '<p class="loading">Predicting yield... This may take a few moments.</p>';
    predictionResults.style.display = 'block';
    
    // Create data object for API request
    const requestData = {
        region: region,
        soil_type: soilType,
        crop: crop,
        rainfall_mm: parseFloat(rainfallMm),
        temperature_celsius: parseFloat(temperatureCelsius),
        fertilizer_used: fertilizerUsed === 'true',
        irrigation_used: irrigationUsed === 'true',
        weather_condition: weatherCondition,
        days_to_harvest: parseInt(daysToHarvest)
    };
    
    console.log('Request data:', requestData);
    
    try {
        // Send request to backend
        console.log('Sending request to:', `${API_BASE_URL}/predict-yield/`);
        const response = await fetch(`${API_BASE_URL}/predict-yield/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        console.log('Response status:', response.status);
        console.log('Response headers:', [...response.headers.entries()]);
        
        const data = await response.json();
        console.log('Response data:', data);
        
        if (response.ok) {
            // Display results
            displayYieldResults(data);
        } else {
            predictionOutput.innerHTML = `<p class="error">Error: ${data.error || 'Unknown error occurred'}</p>`;
            if (data.missing_fields) {
                predictionOutput.innerHTML += `<p class="error">Missing fields: ${data.missing_fields.join(', ')}</p>`;
            }
        }
    } catch (error) {
        console.error('Error predicting yield:', error);
        predictionOutput.innerHTML = `<p class="error">Error predicting yield: ${error.message}. Please try again.</p>`;
    }
    
    // Prevent any default form submission behavior
    return false;
}

// Display yield prediction results
function displayYieldResults(data) {
    console.log('Displaying results:', data);
    
    // Get DOM elements
    const predictionOutput = document.getElementById('prediction-output');
    const predictionResults = document.getElementById('prediction-results');
    
    // Format the explanation to be more readable
    let explanationHtml = '';
    if (data.explanation) {
        // Convert newlines to <br> tags and format as paragraphs
        explanationHtml = data.explanation
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>')
            .replace(/^(.*)$/, '<p>$1</p>');
    }
    
    predictionOutput.innerHTML = `
        <div class="yield-result">
            <h4>Predicted Yield</h4>
            <div class="yield-value">${data.predicted_yield} tons/hectare</div>
            <div class="confidence-score">Confidence: ${(data.confidence_score * 100).toFixed(2)}%</div>
            <div class="input-summary">
                <h5>Input Data:</h5>
                <ul>
                    <li>Region: ${data.input_data.Region}</li>
                    <li>Soil Type: ${data.input_data.Soil_Type}</li>
                    <li>Crop: ${data.input_data.Crop}</li>
                    <li>Rainfall: ${data.input_data.Rainfall_mm} mm</li>
                    <li>Temperature: ${data.input_data.Temperature_Celsius}°C</li>
                    <li>Fertilizer Used: ${data.input_data.Fertilizer_Used ? 'Yes' : 'No'}</li>
                    <li>Irrigation Used: ${data.input_data.Irrigation_Used ? 'Yes' : 'No'}</li>
                    <li>Weather: ${data.input_data.Weather_Condition}</li>
                    <li>Days to Harvest: ${data.input_data.Days_to_Harvest}</li>
                </ul>
            </div>
            ${explanationHtml ? `
            <div class="explanation">
                <h5>Explanation:</h5>
                ${explanationHtml}
            </div>` : ''}
            <p class="prediction-date">Predicted on: ${new Date(data.timestamp).toLocaleString()}</p>
        </div>
    `;
    // Ensure the results section is visible
    predictionResults.style.display = 'block';
    
    // Scroll to results
    predictionResults.scrollIntoView({ behavior: 'smooth' });
}

// Analyze plant disease from uploaded image
async function analyzeDisease() {
    // Get DOM elements
    const diseaseResults = document.getElementById('disease-results');
    const diseaseOutput = document.getElementById('disease-output');
    const adviceSection = document.getElementById('disease-advice');
    
    const imageFile = document.getElementById('leaf-image').files[0];
    
    if (!imageFile) {
        alert('Please upload an image');
        return;
    }
    
    // Show loading state
    diseaseOutput.innerHTML = '<p class="loading">Analyzing image for diseases... This may take a few moments.</p>';
    diseaseResults.style.display = 'block';
    
    // Hide previous advice
    if (adviceSection) {
        adviceSection.style.display = 'none';
    }
    
    // Create FormData for file upload
    const formData = new FormData();
    formData.append('image', imageFile);
    
    try {
        // Send request to backend
        const response = await fetch(`${API_BASE_URL}/disease/predict/`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            // Display results
            displayDiseaseResults(data);
            
            // Get treatment advice
            await getTreatmentAdvice(data.predicted_disease);
        } else {
            diseaseOutput.innerHTML = `<p class="error">Error: ${data.error || 'Unknown error occurred'}</p>`;
        }
    } catch (error) {
        diseaseOutput.innerHTML = `<p class="error">Error analyzing image: ${error.message}. Please try again.</p>`;
    }
}

// Display disease detection results
function displayDiseaseResults(data) {
    // Get DOM elements
    const diseaseOutput = document.getElementById('disease-output');
    const diseaseResults = document.getElementById('disease-results');
    
    let html = `
        <div class="disease-result">
            <h4>Predicted Disease</h4>
            <div class="disease-name">${data.predicted_disease}</div>
            <div class="confidence-score">Confidence: ${(data.confidence_score * 100).toFixed(2)}%</div>
            ${data.image_url ? `<img src="${data.image_url}" alt="Uploaded leaf" style="max-width: 300px; margin-top: 1rem;">` : ''}
            <p class="disease-date">Detected on: ${new Date(data.timestamp).toLocaleString()}</p>
    `;
    
    // Add AI suggestions if available
    if (data.ai_suggestions && data.ai_enabled) {
        html += `
            <div class="ai-suggestions" style="margin-top: 1.5rem;">
                <h6>🤖 AI-Powered Expert Suggestions</h6>
                <div class="recommendations-content">
                    ${data.ai_suggestions.replace(/\n/g, '<br>')}
                </div>
            </div>
        `;
    } else if (data.advice) {
        html += `
            <div class="treatment-advice" style="margin-top: 1rem;">
                <h6>Treatment Advice</h6>
                <p>${data.advice}</p>
            </div>
        `;
    }
    
    html += `</div>`;
    
    diseaseOutput.innerHTML = html;
    // Ensure the results section is visible
    diseaseResults.style.display = 'block';
}

// Get treatment advice for a disease
async function getTreatmentAdvice(diseaseName) {
    // Get DOM elements
    const adviceSection = document.getElementById('disease-advice');
    const adviceOutput = document.getElementById('advice-output');
    
    if (adviceSection && adviceOutput) {
        adviceSection.style.display = 'block';
        adviceOutput.innerHTML = '<p class="loading">Getting treatment advice... This may take a moment.</p>';
        
        try {
            const response = await fetch(`${API_BASE_URL}/disease/advice/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    disease_name: diseaseName,
                    severity: 'moderate'
                })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                adviceOutput.innerHTML = `
                    <div class="advice-content">
                        <h4>Recommended Treatment</h4>
                        <p>${data.advice}</p>
                        ${data.related_documents && data.related_documents.length > 0 ? `
                            <p><strong>Related Information:</strong></p>
                            <ul>
                                ${data.related_documents.map(doc => `<li>${doc.title}</li>`).join('')}
                            </ul>
                        ` : ''}
                    </div>
                `;
            } else {
                adviceOutput.innerHTML = `<p class="error">Error getting advice: ${data.error || 'Unknown error occurred'}</p>`;
            }
        } catch (error) {
            adviceOutput.innerHTML = `<p class="error">Error getting advice: ${error.message}. Please try again.</p>`;
        }
    }
}

// Generate report based on user selections
async function generateReport() {
    // Get DOM elements
    const reportOutput = document.getElementById('report-output');
    const reportResults = document.getElementById('report-results');
    
    // Get form values
    const reportType = document.getElementById('report-type').value;
    const timeRange = document.getElementById('time-range').value;
    
    // Show loading indicator
    reportOutput.innerHTML = '<p class="loading">Generating report... Please wait.</p>';
    if (reportResults) {
        reportResults.style.display = 'block';
    }
    
    try {
        // Determine endpoint based on report type
        let endpoint = '';
        switch (reportType) {
            case 'predictions':
                endpoint = '/reports/predictions/';
                break;
            case 'diseases':
                endpoint = '/reports/diseases/';
                break;
            case 'recommendations':
                endpoint = '/reports/recommendations/';
                break;
            case 'pests':
                endpoint = '/reports/pests/';
                break;
            case 'market':
                endpoint = '/reports/market/';
                break;
            default:
                endpoint = '/reports/predictions/';
        }
        
        // Call API to get report data
        const response = await fetch(`${API_BASE_URL}${endpoint}`);
        const data = await response.json();
        
        if (response.ok) {
            // Display report results
            displayReportResults(data, reportType);
            
            // Generate chart
            generateChart(data, reportType);
            
            // Show download button
            const downloadPdfBtn = document.getElementById('download-pdf');
            if (downloadPdfBtn) {
                downloadPdfBtn.style.display = 'inline-block';
            }
        } else {
            reportOutput.innerHTML = `<p class="error">Error generating report: ${data.error || 'Unknown error occurred'}</p>`;
        }
    } catch (error) {
        reportOutput.innerHTML = `<p class="error">Error generating report: ${error.message}</p>`;
        console.error('Report generation error:', error);
    }
}

// Display report results
function displayReportResults(data, reportType) {
    // Get DOM elements
    const reportOutput = document.getElementById('report-output');
    const reportResults = document.getElementById('report-results');
    
    let html = '';
    
    // Add summary section
    html += '<div class="report-summary">';
    html += '<h4>Report Summary</h4>';
    html += '<ul>';
    
    for (const [key, value] of Object.entries(data.summary)) {
        // Format key to be more readable
        const formattedKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        html += `<li><strong>${formattedKey}:</strong> ${value}</li>`;
    }
    
    html += '</ul>';
    html += '</div>';
    
    // Add data table
    html += '<div class="report-data">';
    html += '<h4>Detailed Data</h4>';
    
    if (data.data && data.data.length > 0) {
        html += '<div class="table-responsive">';
        html += '<table class="data-table">';
        html += '<thead>';
        
        // Create table headers based on report type
        switch (reportType) {
            case 'predictions':
                html += '<tr><th>Date</th><th>Predicted Yield</th><th>Confidence</th><th>Crop</th></tr>';
                break;
            case 'diseases':
                html += '<tr><th>Date</th><th>Disease</th><th>Confidence</th><th>Severity</th></tr>';
                break;
            case 'recommendations':
                html += '<tr><th>Date</th><th>Crop</th><th>Fertilizer</th><th>Quantity</th><th>Confidence</th></tr>';
                break;
            case 'pests':
                html += '<tr><th>Date</th><th>Pest</th><th>Confidence</th><th>Severity</th></tr>';
                break;
            case 'market':
                html += '<tr><th>Date</th><th>Crop</th><th>Predicted Price</th><th>Confidence</th><th>Trend</th></tr>';
                break;
        }
        
        html += '</thead>';
        html += '<tbody>';
        
        // Add data rows
        data.data.forEach(item => {
            html += '<tr>';
            switch (reportType) {
                case 'predictions':
                    html += `<td>${item.date || 'N/A'}</td>`;
                    html += `<td>${item.predicted_yield !== undefined ? item.predicted_yield + ' tons/ha' : 'N/A'}</td>`;
                    html += `<td>${item.confidence !== undefined ? (item.confidence * 100).toFixed(1) + '%' : 'N/A'}</td>`;
                    html += `<td>${item.crop || 'N/A'}</td>`;
                    break;
                case 'diseases':
                    html += `<td>${item.date || 'N/A'}</td>`;
                    html += `<td>${item.disease || 'N/A'}</td>`;
                    html += `<td>${item.confidence !== undefined ? (item.confidence * 100).toFixed(1) + '%' : 'N/A'}</td>`;
                    html += `<td>${item.severity || 'N/A'}</td>`;
                    break;
                case 'recommendations':
                    html += `<td>${item.date || 'N/A'}</td>`;
                    html += `<td>${item.crop || 'N/A'}</td>`;
                    html += `<td>${item.fertilizer || 'N/A'}</td>`;
                    html += `<td>${item.quantity !== undefined ? item.quantity + ' kg/ha' : 'N/A'}</td>`;
                    html += `<td>${item.confidence !== undefined ? (item.confidence * 100).toFixed(1) + '%' : 'N/A'}</td>`;
                    break;
                case 'pests':
                    html += `<td>${item.date || 'N/A'}</td>`;
                    html += `<td>${item.pest || 'N/A'}</td>`;
                    html += `<td>${item.confidence !== undefined ? (item.confidence * 100).toFixed(1) + '%' : 'N/A'}</td>`;
                    html += `<td>${item.severity !== undefined ? item.severity : 'N/A'}</td>`;
                    break;
                case 'market':
                    html += `<td>${item.date || 'N/A'}</td>`;
                    html += `<td>${item.crop || 'N/A'}</td>`;
                    html += `<td>${item.predicted_price !== undefined ? '$' + item.predicted_price.toFixed(2) + '/ton' : 'N/A'}</td>`;
                    html += `<td>${item.confidence !== undefined ? (item.confidence * 100).toFixed(1) + '%' : 'N/A'}</td>`;
                    html += `<td>${item.market_trend || 'N/A'}</td>`;
                    break;
            }
            html += '</tr>';
        });
        
        html += '</tbody>';
        html += '</table>';
        html += '</div>';
    } else {
        html += '<p>No data available for the selected period.</p>';
    }
    
    html += '</div>';
    
    // Ensure the report results section is visible
    if (reportResults) {
        reportResults.style.display = 'block';
    }
    if (reportOutput) {
        reportOutput.innerHTML = html;
    }
}

// Generate chart based on report data
function generateChart(data, reportType) {
    // Get DOM elements
    const reportChartCanvas = document.getElementById('report-chart');
    
    // Destroy existing chart if it exists
    if (window.reportChart) {
        window.reportChart.destroy();
    }
    
    // Check if canvas element exists
    if (!reportChartCanvas) {
        console.error('Chart canvas not found');
        return;
    }
    
    // Get chart context
    const ctx = reportChartCanvas.getContext('2d');
    
    // Prepare chart data based on report type
    let chartData = {};
    let chartOptions = {};
    
    // Check if we have data
    if (!data || !data.data || data.data.length === 0) {
        console.warn('No data available for chart');
        return;
    }
    
    switch (reportType) {
        case 'predictions':
            // For predictions, show yield over time
            const dates = data.data.map(item => item.date || 'Unknown');
            const predictedYields = data.data.map(item => item.predicted_yield !== undefined ? item.predicted_yield : 0);
            
            chartData = {
                labels: dates,
                datasets: [
                    {
                        label: 'Predicted Yield (tons/ha)',
                        data: predictedYields,
                        borderColor: 'rgb(75, 192, 192)',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        tension: 0.1
                    }
                ]
            };
            
            chartOptions = {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Yield Predictions Over Time'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false
                    }
                }
            };
            break;
            
        case 'diseases':
            // For diseases, show disease occurrences
            const diseaseCounts = {};
            data.data.forEach(item => {
                const disease = item.disease || 'Unknown';
                if (diseaseCounts[disease]) {
                    diseaseCounts[disease]++;
                } else {
                    diseaseCounts[disease] = 1;
                }
            });
            
            const diseaseLabels = Object.keys(diseaseCounts);
            const diseaseValues = Object.values(diseaseCounts);
            
            chartData = {
                labels: diseaseLabels,
                datasets: [{
                    label: 'Disease Occurrences',
                    data: diseaseValues,
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.2)',
                        'rgba(54, 162, 235, 0.2)',
                        'rgba(255, 205, 86, 0.2)',
                        'rgba(75, 192, 192, 0.2)',
                        'rgba(153, 102, 255, 0.2)',
                        'rgba(255, 159, 64, 0.2)'
                    ],
                    borderColor: [
                        'rgb(255, 99, 132)',
                        'rgb(54, 162, 235)',
                        'rgb(255, 205, 86)',
                        'rgb(75, 192, 192)',
                        'rgb(153, 102, 255)',
                        'rgb(255, 159, 64)'
                    ],
                    borderWidth: 1
                }]
            };
            
            chartOptions = {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Disease Occurrences'
                    }
                }
            };
            break;
            
        case 'recommendations':
            // For recommendations, show crop recommendations
            const cropCounts = {};
            data.data.forEach(item => {
                const crop = item.crop || 'Unknown';
                if (cropCounts[crop]) {
                    cropCounts[crop]++;
                } else {
                    cropCounts[crop] = 1;
                }
            });
            
            const cropLabels = Object.keys(cropCounts);
            const cropValues = Object.values(cropCounts);
            
            chartData = {
                labels: cropLabels,
                datasets: [{
                    label: 'Crop Recommendations',
                    data: cropValues,
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.2)',
                        'rgba(54, 162, 235, 0.2)',
                        'rgba(255, 205, 86, 0.2)',
                        'rgba(75, 192, 192, 0.2)',
                        'rgba(153, 102, 255, 0.2)',
                        'rgba(255, 159, 64, 0.2)'
                    ],
                    borderColor: [
                        'rgb(255, 99, 132)',
                        'rgb(54, 162, 235)',
                        'rgb(255, 205, 86)',
                        'rgb(75, 192, 192)',
                        'rgb(153, 102, 255)',
                        'rgb(255, 159, 64)'
                    ],
                    borderWidth: 1
                }]
            };
            
            chartOptions = {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Crop Recommendations Distribution'
                    }
                }
            };
            break;
            
        case 'pests':
            // For pests, show pest occurrences
            const pestCounts = {};
            data.data.forEach(item => {
                const pest = item.pest || 'Unknown';
                if (pestCounts[pest]) {
                    pestCounts[pest]++;
                } else {
                    pestCounts[pest] = 1;
                }
            });
            
            const pestLabels = Object.keys(pestCounts);
            const pestValues = Object.values(pestCounts);
            
            chartData = {
                labels: pestLabels,
                datasets: [{
                    label: 'Pest Occurrences',
                    data: pestValues,
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.2)',
                        'rgba(54, 162, 235, 0.2)',
                        'rgba(255, 205, 86, 0.2)',
                        'rgba(75, 192, 192, 0.2)',
                        'rgba(153, 102, 255, 0.2)',
                        'rgba(255, 159, 64, 0.2)'
                    ],
                    borderColor: [
                        'rgb(255, 99, 132)',
                        'rgb(54, 162, 235)',
                        'rgb(255, 205, 86)',
                        'rgb(75, 192, 192)',
                        'rgb(153, 102, 255)',
                        'rgb(255, 159, 64)'
                    ],
                    borderWidth: 1
                }]
            };
            
            chartOptions = {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Pest Occurrences'
                    }
                }
            };
            break;
            
        case 'market':
            // For market, show price trends over time
            const marketDates = data.data.map(item => item.date || 'Unknown');
            const marketPrices = data.data.map(item => item.predicted_price !== undefined ? item.predicted_price : 0);
            
            chartData = {
                labels: marketDates,
                datasets: [
                    {
                        label: 'Predicted Price ($/ton)',
                        data: marketPrices,
                        borderColor: 'rgb(153, 102, 255)',
                        backgroundColor: 'rgba(153, 102, 255, 0.2)',
                        tension: 0.1
                    }
                ]
            };
            
            chartOptions = {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Market Price Trends Over Time'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false
                    }
                }
            };
            break;
    }
    
    // Create chart
    try {
        window.reportChart = new Chart(ctx, {
            type: reportType === 'predictions' || reportType === 'market' ? 'line' : 'bar',
            data: chartData,
            options: chartOptions
        });
    } catch (error) {
        console.error('Error creating chart:', error);
    }
}

// Download report as PDF
function downloadPDF() {
    // In a real implementation, this would generate a PDF
    // For this demo, we'll just show an alert
    alert('PDF download functionality would be implemented here. In a real application, this would generate and download a PDF report.');
}

// Utility function to make API requests
async function apiRequest(endpoint, method = 'GET', data = null) {
    const url = `${API_BASE_URL}${endpoint}`;
    const options = {
        method: method,
        headers: {
            'Content-Type': 'application/json',
        }
    };
    
    if (data) {
        options.body = JSON.stringify(data);
    }
    
    try {
        const response = await fetch(url, options);
        if (!response.ok) {
            throw new Error(`API request failed: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error('API request error:', error);
        throw error;
    }
}

// Simple HTML escape helper for safe insertion of dynamic text
function escapeHtml(unsafe) {
    return unsafe
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/\"/g, '&quot;')
        .replace(/\'/g, '&#039;');
}