/**
 * Pest Prediction functionality for the Smart Agriculture Platform
 * Enhanced version with detailed insights and recommendations
 */

document.addEventListener('DOMContentLoaded', function() {
    // Get the pest prediction form and results elements
    const pestForm = document.getElementById('pest-prediction-form');
    const pestResults = document.getElementById('pest-prediction-results');
    const pestOutput = document.getElementById('pest-prediction-output');
    
    // Debug: Check if elements exist
    console.log('Pest form element:', pestForm);
    console.log('Pest results element:', pestResults);
    console.log('Pest output element:', pestOutput);
    
    // Add event listener to the form
    if (pestForm) {
        pestForm.addEventListener('submit', function(e) {
            e.preventDefault();
            console.log('Form submitted');
            predictPest();
        });
    } else {
        console.error('Pest prediction form not found');
    }
    
    /**
     * Function to handle pest prediction
     */
    function predictPest() {
        console.log('predictPest function called');
        // Debug: Log all form elements
        console.log("=== Form Elements Debug ===");
        console.log("pest-crop:", document.getElementById('pest-crop')?.value);
        console.log("pest-region:", document.getElementById('pest-region')?.value);
        console.log("pest-season:", document.getElementById('pest-season')?.value);
        console.log("pest-temperature:", document.getElementById('pest-temperature')?.value);
        console.log("pest-humidity:", document.getElementById('pest-humidity')?.value);
        console.log("pest-rainfall:", document.getElementById('pest-rainfall')?.value);
        console.log("pest-wind-speed:", document.getElementById('pest-wind-speed')?.value);
        console.log("pest-soil-moisture:", document.getElementById('pest-soil-moisture')?.value);
        console.log("pest-soil-ph:", document.getElementById('pest-soil-ph')?.value);
        console.log("pest-soil-type:", document.getElementById('pest-soil-type')?.value);
        console.log("pest-nitrogen:", document.getElementById('pest-nitrogen')?.value);
        console.log("pest-phosphorus:", document.getElementById('pest-phosphorus')?.value);
        console.log("pest-potassium:", document.getElementById('pest-potassium')?.value);
        console.log("pest-weather-condition:", document.getElementById('pest-weather-condition')?.value);
        console.log("pest-irrigation-method:", document.getElementById('pest-irrigation-method')?.value);
        console.log("pest-previous-crop:", document.getElementById('pest-previous-crop')?.value);
        console.log("pest-days-since-planting:", document.getElementById('pest-days-since-planting')?.value);
        console.log("pest-plant-density:", document.getElementById('pest-plant-density')?.value);

        // Get form data
        const formData = {
            crop: document.getElementById('pest-crop').value,
            region: document.getElementById('pest-region').value,
            season: document.getElementById('pest-season').value,
            temperature: document.getElementById('pest-temperature').value !== '' ? parseFloat(document.getElementById('pest-temperature').value) : null,
            humidity: document.getElementById('pest-humidity').value !== '' ? parseFloat(document.getElementById('pest-humidity').value) : null,
            rainfall: document.getElementById('pest-rainfall').value !== '' ? parseFloat(document.getElementById('pest-rainfall').value) : null,
            wind_speed: document.getElementById('pest-wind-speed').value !== '' ? parseFloat(document.getElementById('pest-wind-speed').value) : null,
            soil_moisture: document.getElementById('pest-soil-moisture').value !== '' ? parseFloat(document.getElementById('pest-soil-moisture').value) : null,
            soil_ph: document.getElementById('pest-soil-ph').value !== '' ? parseFloat(document.getElementById('pest-soil-ph').value) : null,
            soil_type: document.getElementById('pest-soil-type').value,
            nitrogen: document.getElementById('pest-nitrogen').value !== '' ? parseFloat(document.getElementById('pest-nitrogen').value) : null,
            phosphorus: document.getElementById('pest-phosphorus').value !== '' ? parseFloat(document.getElementById('pest-phosphorus').value) : null,
            potassium: document.getElementById('pest-potassium').value !== '' ? parseFloat(document.getElementById('pest-potassium').value) : null,
            weather_condition: document.getElementById('pest-weather-condition').value,
            irrigation_method: document.getElementById('pest-irrigation-method').value,
            previous_crop: document.getElementById('pest-previous-crop').value,
            days_since_planting: document.getElementById('pest-days-since-planting').value !== '' ? parseInt(document.getElementById('pest-days-since-planting').value) : null,
            plant_density: document.getElementById('pest-plant-density').value !== '' ? parseInt(document.getElementById('pest-plant-density').value) : null
        };
        
        console.log("Form Data Collected:", formData);
        
        // Validate form data
        if (!validatePestFormData(formData)) {
            console.log('Form validation failed');
            return;
        }
        
        console.log('Form validation passed');
        // Show loading state
        showPestLoadingState();
        
        // Send request to backend
        fetch('http://127.0.0.1:8000/api/predict-pest/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCSRFToken()
            },
            body: JSON.stringify(formData)
        })
        .then(response => {
            console.log('Response status:', response.status);
            console.log('Response headers:', response.headers);
            return response.json();
        })
        .then(data => {
            console.log('Response data:', data);
            if (data.error) {
                showPestError(data.error);
            } else {
                displayPestResults(data);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showPestError('An error occurred while processing your request. Please try again.');
        });
    }
    
    /**
     * Function to validate form data
     */
    function validatePestFormData(data) {
        console.log("Validating form data:", data);
        
        // Function to format field names for display
        function formatFieldName(fieldName) {
            return fieldName
                .replace(/_/g, ' ')
                .replace(/\b\w/g, l => l.toUpperCase());
        }
        
        // Check if all required fields are filled
        const requiredFields = [
            'crop', 'region', 'season', 'temperature', 'humidity', 'rainfall', 
            'wind_speed', 'soil_moisture', 'soil_ph', 'soil_type', 'nitrogen', 
            'phosphorus', 'potassium', 'weather_condition', 'irrigation_method', 
            'previous_crop', 'days_since_planting', 'plant_density'
        ];
        
        for (const field of requiredFields) {
            // Special handling for numeric fields that can be 0
            const numericFields = ['temperature', 'humidity', 'rainfall', 'wind_speed', 
                                 'soil_moisture', 'soil_ph', 'nitrogen', 'phosphorus', 
                                 'potassium', 'days_since_planting', 'plant_density'];
            
            if (numericFields.includes(field)) {
                // For numeric fields, check if they are null or undefined (but allow 0)
                // Using Number.isNaN to properly check for invalid numbers
                if (data[field] === null || data[field] === undefined || Number.isNaN(data[field])) {
                    showPestError(`Please fill in all required fields. Missing: ${formatFieldName(field)}`);
                    return false;
                }
            } else {
                // For non-numeric fields, check if they are empty strings or null/undefined
                if (data[field] === null || data[field] === undefined || data[field] === '') {
                    showPestError(`Please fill in all required fields. Missing: ${formatFieldName(field)}`);
                    return false;
                }
            }
        }
        
        // Additional validation
        if (data.temperature !== null && (data.temperature < -10 || data.temperature > 50)) {
            showPestError('Temperature must be between -10°C and 50°C');
            return false;
        }
        
        if (data.humidity !== null && (data.humidity < 0 || data.humidity > 100)) {
            showPestError('Humidity must be between 0% and 100%');
            return false;
        }
        
        if (data.rainfall !== null && (data.rainfall < 0 || data.rainfall > 500)) {
            showPestError('Rainfall must be between 0mm and 500mm');
            return false;
        }
        
        if (data.wind_speed !== null && (data.wind_speed < 0 || data.wind_speed > 100)) {
            showPestError('Wind speed must be between 0km/h and 100km/h');
            return false;
        }
        
        if (data.soil_moisture !== null && (data.soil_moisture < 0 || data.soil_moisture > 100)) {
            showPestError('Soil moisture must be between 0% and 100%');
            return false;
        }
        
        if (data.soil_ph !== null && (data.soil_ph < 0 || data.soil_ph > 14)) {
            showPestError('Soil pH must be between 0 and 14');
            return false;
        }
        
        if (data.nitrogen !== null && (data.nitrogen < 0 || data.nitrogen > 500)) {
            showPestError('Nitrogen must be between 0kg/ha and 500kg/ha');
            return false;
        }
        
        if (data.phosphorus !== null && (data.phosphorus < 0 || data.phosphorus > 500)) {
            showPestError('Phosphorus must be between 0kg/ha and 500kg/ha');
            return false;
        }
        
        if (data.potassium !== null && (data.potassium < 0 || data.potassium > 500)) {
            showPestError('Potassium must be between 0kg/ha and 500kg/ha');
            return false;
        }
        
        if (data.days_since_planting !== null && (data.days_since_planting < 1 || data.days_since_planting > 365)) {
            showPestError('Days since planting must be between 1 and 365');
            return false;
        }
        
        if (data.plant_density !== null && (data.plant_density < 1000 || data.plant_density > 100000)) {
            showPestError('Plant density must be between 1000 and 100000 plants/ha');
            return false;
        }
        
        return true;
    }
    
    /**
     * Function to show loading state
     */
    function showPestLoadingState() {
        console.log('Showing loading state');
        const submitButton = pestForm.querySelector('button[type="submit"]');
        if (submitButton) {
            submitButton.disabled = true;
            submitButton.textContent = 'Predicting...';
        }
        
        if (pestOutput) {
            pestOutput.innerHTML = `
                <div class="loading">
                    <div class="spinner"></div>
                    <p>Analyzing farm conditions for pest prediction...</p>
                    <p>Consulting knowledge base for enhanced insights...</p>
                </div>
            `;
        }
        if (pestResults) {
            pestResults.style.display = 'block';
        }
        console.log('Loading state displayed');
    }
    
    /**
     * Function to display pest prediction results with enhanced insights
     */
    function displayPestResults(data) {
        console.log('Displaying pest results:', data);
        // Hide loading state
        const submitButton = pestForm.querySelector('button[type="submit"]');
        if (submitButton) {
            submitButton.disabled = false;
            submitButton.textContent = 'Predict Pest Occurrence';
        }
        
        // Format the results with enhanced presentation
        const pestPresence = data.pest_presence ? 'Yes' : 'No';
        const severityLevel = getSeverityLevel(data.severity);
        const severityDescription = getSeverityDescription(data.severity);
        
        // Create the results HTML with enhanced formatting
        let html = `
            <div class="prediction-result">
                <h4>Pest Prediction for ${data.input_data.crop}</h4>
                <div class="result-grid">
                    <div class="result-item">
                        <span class="result-label">Predicted Pest:</span>
                        <span class="result-value">${data.predicted_pest}</span>
                    </div>
                    <div class="result-item">
                        <span class="result-label">Pest Presence:</span>
                        <span class="result-value ${data.pest_presence ? 'high-risk' : 'low-risk'}">${pestPresence}</span>
                    </div>
                    <div class="result-item">
                        <span class="result-label">Severity Level:</span>
                        <span class="result-value severity-${data.severity}">${data.severity}/10 (${severityLevel})</span>
                    </div>
                    <div class="result-item">
                        <span class="result-label">Confidence:</span>
                        <span class="result-value">${(data.confidence_score * 100).toFixed(1)}%</span>
                    </div>
                </div>
        `;
        
        // Add explanation if available
        if (data.recommended_treatment) {
            html += `
                <div class="explanation-section">
                    <h5>Pest Analysis</h5>
                    <p>${formatExplanation(data.recommended_treatment)}</p>
                </div>
            `;
        }
        
        // Add enhanced insights if available (mock data for now)
        html += `
                <div class="insights-section">
                    <h5>AI-Powered Pest Insights</h5>
                    <div class="insights-tabs">
                        <button class="tab-button active" onclick="switchPestTab('overview')">Overview</button>
                        <button class="tab-button" onclick="switchPestTab('analysis')">Analysis</button>
                        <button class="tab-button" onclick="switchPestTab('recommendations')">Recommendations</button>
                        <button class="tab-button" onclick="switchPestTab('damage')">Damage Assessment</button>
                    </div>
                    <div class="tab-content">
        `;
        
        // Overview tab (default)
        html += `
                        <div id="overview" class="tab-pane active">
                            <div class="insight-card">
                                <h6>Prediction Summary</h6>
                                <p>The ${data.predicted_pest} is predicted to be present with ${severityDescription.toLowerCase()} severity. This prediction is based on the environmental conditions and crop data you provided.</p>
                                <div class="risk-assessment">
                                    <span>Risk Level: ${severityDescription}</span>
                                    <span>Confidence: ${(data.confidence_score * 100).toFixed(1)}%</span>
                                </div>
                            </div>
        `;
        
        // Add pest analysis information if available
        if (data.pest_analysis) {
            const analysis = data.pest_analysis;
            html += `
                            <div class="insight-card">
                                <h6>Pest Analysis</h6>
                                <div class="analysis-grid">
                                    <div class="analysis-item">
                                        <span class="analysis-label">Severity:</span>
                                        <span class="analysis-value">${analysis.severity_description}</span>
                                    </div>
                                    <div class="analysis-item">
                                        <span class="analysis-label">Confidence:</span>
                                        <span class="analysis-value">${analysis.confidence_description}</span>
                                    </div>
                                </div>
                            </div>
            `;
            
            // Add environmental factors if available
            if (analysis.environmental_factors && analysis.environmental_factors.length > 0) {
                html += `
                            <div class="insight-card">
                                <h6>Environmental Impact Factors</h6>
                                <ul>
                `;
                analysis.environmental_factors.forEach(factor => {
                    html += `<li>${factor}</li>`;
                });
                html += `
                                </ul>
                            </div>
                `;
            }
        }
        
        // Key Environmental Factors (existing code)
        html += `
                            <div class="insight-card">
                                <h6>Key Environmental Factors</h6>
                                <ul>
                                    <li>Temperature: ${data.input_data.temperature}°C ${getTemperatureImpact(data.input_data.temperature)}</li>
                                    <li>Humidity: ${data.input_data.humidity}% ${getHumidityImpact(data.input_data.humidity)}</li>
                                    <li>Rainfall: ${data.input_data.rainfall}mm ${getRainfallImpact(data.input_data.rainfall)}</li>
                                    <li>Soil Moisture: ${data.input_data.soil_moisture}% ${getSoilMoistureImpact(data.input_data.soil_moisture)}</li>
                                </ul>
                            </div>
                        </div>
        `;
        
        // Analysis tab
        html += `
                        <div id="analysis" class="tab-pane">
                            <div class="insight-card">
                                <h6>Comprehensive Pest Analysis</h6>
                                <div class="analysis-content">
                                    <p>The ${data.predicted_pest} is known to thrive under the current environmental conditions:</p>
                                    <ul>
                                        <li>Temperature range of ${data.input_data.temperature - 2}°C to ${data.input_data.temperature + 2}°C is optimal for this pest</li>
                                        <li>Humidity level of ${data.input_data.humidity}% supports pest development</li>
                                        <li>Soil conditions with pH ${data.input_data.soil_ph} and moisture ${data.input_data.soil_moisture}% are favorable</li>
                                        <li>The ${data.input_data.previous_crop} may have contributed to pest buildup</li>
                                    </ul>
                                    <p>This pest typically causes damage by ${getPestDamageType(data.predicted_pest)} and can reduce yield by ${getPestYieldImpact(data.predicted_pest)} if left untreated.</p>
                                </div>
                            </div>
                        </div>
        `;
        
        // Recommendations tab
        html += `
                        <div id="recommendations" class="tab-pane">
        `;
        
        // Add AI suggestions from n8n webhook if available
        if (data.ai_suggestions && data.ai_enabled) {
            html += `
                            <div class="ai-suggestions">
                                <h6>🤖 AI-Powered Expert Suggestions</h6>
                                <div class="recommendations-content">
                                    ${formatRecommendationsContent(data.ai_suggestions)}
                                </div>
                            </div>
            `;
        } else if (data.recommended_treatment) {
            html += `
                            <div class="insight-card">
                                <h6>AI-Powered Recommendations</h6>
                                <div class="recommendations-content">
                                    ${formatRecommendationsContent(data.recommended_treatment)}
                                </div>
                            </div>
            `;
        }
        
        html += `
                            <div class="insight-card">
                                <h6>Prevention Strategies</h6>
                                <ul>
                                    <li>Implement crop rotation to break pest life cycles</li>
                                    <li>Use resistant crop varieties when available</li>
                                    <li>Maintain proper field sanitation to remove pest habitats</li>
                                    <li>Encourage beneficial insects that prey on this pest</li>
                                </ul>
                            </div>
                        </div>
        `;
        
        // Damage Assessment tab
        html += `
                        <div id="damage" class="tab-pane">
        `;
        
        if (data.pest_analysis && data.pest_analysis.potential_damage) {
            html += `
                            <div class="insight-card">
                                <h6>Potential Crop Damage</h6>
                                <p>${data.pest_analysis.potential_damage}</p>
                            </div>
            `;
        }
        
        if (data.pest_analysis && data.pest_analysis.monitoring_advice) {
            html += `
                            <div class="insight-card">
                                <h6>Monitoring Advice</h6>
                                <p>${data.pest_analysis.monitoring_advice}</p>
                            </div>
            `;
        }
        
        html += `
                        </div>
                    </div>
                </div>
        `;
        
        // Add input data section
        html += `
            <div class="input-data-section">
                <h5>Input Data Used</h5>
                <div class="data-grid">
                    <div class="data-item"><strong>Crop:</strong> ${data.input_data.crop}</div>
                    <div class="data-item"><strong>Region:</strong> ${data.input_data.region}</div>
                    <div class="data-item"><strong>Season:</strong> ${data.input_data.season}</div>
                    <div class="data-item"><strong>Temperature:</strong> ${data.input_data.temperature}°C</div>
                    <div class="data-item"><strong>Humidity:</strong> ${data.input_data.humidity}%</div>
                    <div class="data-item"><strong>Rainfall:</strong> ${data.input_data.rainfall}mm</div>
                    <div class="data-item"><strong>Wind Speed:</strong> ${data.input_data.wind_speed}km/h</div>
                    <div class="data-item"><strong>Soil Moisture:</strong> ${data.input_data.soil_moisture}%</div>
                    <div class="data-item"><strong>Soil pH:</strong> ${data.input_data.soil_ph}</div>
                    <div class="data-item"><strong>Soil Type:</strong> ${data.input_data.soil_type}</div>
                    <div class="data-item"><strong>Nitrogen:</strong> ${data.input_data.nitrogen}kg/ha</div>
                    <div class="data-item"><strong>Phosphorus:</strong> ${data.input_data.phosphorus}kg/ha</div>
                    <div class="data-item"><strong>Potassium:</strong> ${data.input_data.potassium}kg/ha</div>
                    <div class="data-item"><strong>Weather:</strong> ${data.input_data.weather_condition}</div>
                    <div class="data-item"><strong>Irrigation:</strong> ${data.input_data.irrigation_method}</div>
                    <div class="data-item"><strong>Previous Crop:</strong> ${data.input_data.previous_crop}</div>
                    <div class="data-item"><strong>Days Since Planting:</strong> ${data.input_data.days_since_planting}</div>
                    <div class="data-item"><strong>Plant Density:</strong> ${data.input_data.plant_density} plants/ha</div>
                </div>
            </div>
        `;
        
        html += `
            </div>
        `;
        
        if (pestOutput) {
            pestOutput.innerHTML = html;
        }
        if (pestResults) {
            pestResults.style.display = 'block';
        }
        console.log('Results displayed');
    }
    
    /**
     * Switch between tabs in the insights section
     */
    window.switchPestTab = function(tabName) {
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
    
    /**
     * Function to format explanation text
     */
    function formatExplanation(text) {
        if (!text) return '';
        return text
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>')
            .replace(/<br><br>/g, '</p><p>')
            .replace(/^<br>/, '')
            .replace(/<br>$/, '');
    }
    
    /**
     * Function to format recommendations content for better display
     */
    function formatRecommendationsContent(content) {
        if (!content) return '';
        // Convert numbered lists and bullet points
        content = content.replace(/(\d+\.)\s/g, '<strong>$1</strong> ');
        content = content.replace(/(\*|\•)\s/g, '• ');
        return formatExplanation(content);
    }
    
    /**
     * Function to get severity level description
     */
    function getSeverityLevel(severity) {
        if (severity >= 8) return 'Very High';
        if (severity >= 6) return 'High';
        if (severity >= 4) return 'Moderate';
        if (severity >= 2) return 'Low';
        return 'Very Low';
    }
    
    /**
     * Function to get severity description
     */
    function getSeverityDescription(severity) {
        if (severity >= 8) return 'Critical Risk';
        if (severity >= 6) return 'High Risk';
        if (severity >= 4) return 'Moderate Risk';
        if (severity >= 2) return 'Low Risk';
        return 'Minimal Risk';
    }
    
    /**
     * Function to get temperature impact description
     */
    function getTemperatureImpact(temp) {
        if (temp < 5) return '(extremely cold conditions - limited pest activity)';
        if (temp < 10) return '(very cold conditions - reduced pest activity)';
        if (temp < 15) return '(cool conditions - limited pest development)';
        if (temp < 20) return '(mild conditions - moderate pest activity)';
        if (temp < 25) return '(optimal conditions - active pest development)';
        if (temp < 30) return '(warm conditions - high pest activity)';
        if (temp < 35) return '(hot conditions - very high pest activity)';
        return '(extremely hot conditions - stress on both crops and pests)';
    }

    /**
     * Function to get humidity impact description
     */
    function getHumidityImpact(humidity) {
        if (humidity < 20) return '(very dry conditions - limited pest survival)';
        if (humidity < 40) return '(dry conditions - reduced pest development)';
        if (humidity < 60) return '(moderate conditions - suitable for many pests)';
        if (humidity < 80) return '(humid conditions - favorable for pest growth)';
        return '(very humid conditions - optimal for many pests)';
    }

    /**
     * Function to get rainfall impact description
     */
    function getRainfallImpact(rainfall) {
        if (rainfall < 10) return '(very low rainfall - drought stress on crops)';
        if (rainfall < 30) return '(low rainfall - limited water availability)';
        if (rainfall < 60) return '(moderate rainfall - adequate moisture)';
        if (rainfall < 100) return '(high rainfall - excess moisture)';
        return '(very high rainfall - potential waterlogging)';
    }

    /**
     * Function to get soil moisture impact description
     */
    function getSoilMoistureImpact(moisture) {
        if (moisture < 10) return '(very dry soil - water stress)';
        if (moisture < 30) return '(dry soil - limited moisture)';
        if (moisture < 60) return '(moderate moisture - optimal conditions)';
        if (moisture < 80) return '(wet soil - excess moisture)';
        return '(very wet soil - potential waterlogging)';
    }
    
    /**
     * Function to get pest damage type (mock data)
     */
    function getPestDamageType(pest) {
        const damageTypes = {
            'Aphids': 'sucking plant juices and transmitting viruses',
            'Armyworms': 'consuming leaves and stems',
            'Boll Weevils': 'damaging cotton bolls',
            'Corn Borers': 'boring into corn stalks',
            'Cutworms': 'cutting young plants at the base',
            'Flea Beetles': 'creating small holes in leaves',
            'Grasshoppers': 'consuming foliage',
            'Hessian Fly': 'causing stunted growth',
            'Japanese Beetles': 'skeletonizing leaves',
            'Leafhoppers': 'sucking plant sap and spreading disease',
            'Root Knot Nematodes': 'forming galls on roots',
            'Spider Mites': 'sucking plant juices and causing stippling',
            'Stem Borers': 'damaging plant vascular system',
            'Thrips': 'scarring fruits and transmitting viruses',
            'White Grubs': 'damaging roots',
            'Whiteflies': 'sucking plant juices and excreting honeydew',
            'Wireworms': 'damaging roots and tubers'
        };
        return damageTypes[pest] || 'damaging plant tissues';
    }
    
    /**
     * Function to get pest yield impact (mock data)
     */
    function getPestYieldImpact(pest) {
        const yieldImpacts = {
            'Aphids': '10-30%',
            'Armyworms': '20-50%',
            'Boll Weevils': '30-60%',
            'Corn Borers': '15-40%',
            'Cutworms': '10-25%',
            'Flea Beetles': '5-15%',
            'Grasshoppers': '25-50%',
            'Hessian Fly': '10-20%',
            'Japanese Beetles': '15-30%',
            'Leafhoppers': '10-25%',
            'Root Knot Nematodes': '20-40%',
            'Spider Mites': '10-20%',
            'Stem Borers': '15-35%',
            'Thrips': '5-15%',
            'White Grubs': '10-25%',
            'Whiteflies': '10-20%',
            'Wireworms': '15-30%'
        };
        return yieldImpacts[pest] || '10-30%';
    }
    
    /**
     * Function to show error messages
     */
    function showPestError(message) {
        console.log('Showing error message:', message);
        // Hide loading state
        const submitButton = pestForm.querySelector('button[type="submit"]');
        if (submitButton) {
            submitButton.disabled = false;
            submitButton.textContent = 'Predict Pest Occurrence';
        }
        
        if (pestOutput) {
            pestOutput.innerHTML = `
                <div class="error-message">
                    <h4>Error</h4>
                    <p>${message}</p>
                    <p>Please check your input data and try again.</p>
                </div>
            `;
        }
        if (pestResults) {
            pestResults.style.display = 'block';
        }
        console.log('Error message displayed');
    }
    
    /**
     * Function to get CSRF token
     */
    function getCSRFToken() {
        // Try to get the CSRF token from the hidden input field in the form
        const csrfInput = document.querySelector('#pest-prediction-form input[name=csrfmiddlewaretoken]');
        if (csrfInput) {
            return csrfInput.value;
        }
        // Fallback to looking for any csrfmiddlewaretoken element
        const csrfElement = document.querySelector('[name=csrfmiddlewaretoken]');
        if (csrfElement) {
            return csrfElement.value;
        }
        // Last resort - try the meta tag in the head
        const metaTag = document.querySelector('meta[name=csrf-token]');
        if (metaTag) {
            return metaTag.getAttribute('content');
        }
        // Return empty string if nothing found
        console.error('CSRF token not found');
        return '';
    }
});