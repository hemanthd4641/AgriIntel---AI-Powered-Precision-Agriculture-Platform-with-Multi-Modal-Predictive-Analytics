/**
 * Market Predictions JavaScript Module
 * Handles market price prediction functionality
 */

document.addEventListener('DOMContentLoaded', function() {
    // Get form and result elements
    const marketForm = document.getElementById('market-prediction-form');
    const marketResults = document.getElementById('market-prediction-results');
    const marketOutput = document.getElementById('market-prediction-output');
    
    // Handle form submission
    if (marketForm) {
        marketForm.addEventListener('submit', function(e) {
            e.preventDefault();
            predictMarketPrice();
        });
    }
    
    /**
     * Predict market price based on form inputs
     */
    function predictMarketPrice() {
        // Get form data
        const crop = document.getElementById('market-crop').value;
        const region = document.getElementById('market-region').value;
        const season = document.getElementById('market-season').value;
        const yieldPrediction = document.getElementById('expected-yield').value;
        const globalDemand = document.getElementById('global-demand').value;
        const weatherImpact = document.getElementById('weather-impact').value;
        const economicCondition = document.getElementById('economic-condition').value;
        const supplyIndex = document.getElementById('supply-index').value;
        const demandIndex = document.getElementById('demand-index').value;
        const inventoryLevel = document.getElementById('inventory-level').value;
        const exportDemand = document.getElementById('export-demand').value;
        const productionCost = document.getElementById('production-cost').value;
        const daysToHarvest = document.getElementById('days-to-harvest').value;
        const fertilizerUsage = document.getElementById('fertilizer-usage').value;
        const irrigationUsage = document.getElementById('irrigation-usage').value;
        
        // Validate required fields
        if (!crop || !region || !season || !yieldPrediction) {
            alert('Please fill in all required fields');
            return;
        }
        
        // Show loading state
        showMarketLoading();
        
        // Safely convert values to appropriate types with defaults
        function safeFloat(value, defaultValue = 0.0) {
            if (value === '' || value === null || value === undefined) {
                return defaultValue;
            }
            const parsed = parseFloat(value);
            return isNaN(parsed) ? defaultValue : parsed;
        }
        
        function safeInt(value, defaultValue = 0) {
            if (value === '' || value === null || value === undefined) {
                return defaultValue;
            }
            const parsed = parseInt(value, 10);
            return isNaN(parsed) ? defaultValue : parsed;
        }
        
        // Prepare data for API call
        const requestData = {
            crop: crop,
            region: region,
            season: season,
            yield_prediction: safeFloat(yieldPrediction),
            global_demand: globalDemand || 'medium',
            weather_impact: weatherImpact || 'normal',
            economic_condition: economicCondition || 'stable',
            supply_index: safeFloat(supplyIndex, 60.0),
            demand_index: safeFloat(demandIndex, 60.0),
            inventory_level: safeFloat(inventoryLevel, 50.0),
            export_demand: safeFloat(exportDemand, 60.0),
            production_cost: safeFloat(productionCost, 200.0),
            days_to_harvest: safeInt(daysToHarvest, 90),
            fertilizer_usage: fertilizerUsage || 'medium',
            irrigation_usage: irrigationUsage || 'medium'
        };
        
        console.log("Sending request data:", requestData);
        
        // Make API call
        fetch('http://127.0.0.1:8000/api/predict-market-price/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCSRFToken()
            },
            body: JSON.stringify(requestData)
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showMarketError(data.error);
            } else {
                displayMarketResults(data);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showMarketError('An error occurred while predicting market price. Please try again.');
        });
    }
    
    /**
     * Display market prediction results
     */
    function displayMarketResults(data) {
        // Hide loading state
        hideMarketLoading();
        
        // Show results section
        marketResults.style.display = 'block';
        
        // Generate HTML for results
        let html = `
            <div class="prediction-result">
                <h4>Market Price Prediction for ${data.crop.name}</h4>
                <div class="result-grid">
                    <div class="result-item">
                        <span class="result-label">Predicted Price:</span>
                        <span class="result-value">$${data.predicted_price_per_ton.toFixed(2)}/ton</span>
                    </div>
                    <div class="result-item">
                        <span class="result-label">Market Trend:</span>
                        <span class="result-value trend-${data.market_trend}">${data.market_trend.charAt(0).toUpperCase() + data.market_trend.slice(1)}</span>
                    </div>
                    <div class="result-item">
                        <span class="result-label">Confidence:</span>
                        <span class="result-value">${(data.confidence_score * 100).toFixed(1)}%</span>
                    </div>
                    <div class="result-item">
                        <span class="result-label">Forecast Period:</span>
                        <span class="result-value">${data.forecast_period_days} days</span>
                    </div>
                </div>
        `;
        
        // Add explanation if available
        if (data.explanation) {
            html += `
                <div class="explanation-section">
                    <h5>Market Analysis</h5>
                    <p>${data.explanation}</p>
                </div>
            `;
        }
        
        // Add enhanced insights if available
        if (data.enhanced_insights) {
            const insights = data.enhanced_insights;
            html += `
                <div class="insights-section">
                    <h5>AI-Powered Market Insights</h5>
                    <div class="insights-tabs">
                        <button class="tab-button active" onclick="switchMarketTab('overview')">Overview</button>
                        <button class="tab-button" onclick="switchMarketTab('analysis')">Analysis</button>
                        <button class="tab-button" onclick="switchMarketTab('recommendations')">Recommendations</button>
                        <button class="tab-button" onclick="switchMarketTab('intelligence')">Market Intelligence</button>
                    </div>
                    <div class="tab-content">
            `;
            
            // Overview tab (default)
            html += `
                        <div id="overview" class="tab-pane active">
                            <div class="insight-card">
                                <h6>Prediction Summary</h6>
                                <p>${insights.prediction_summary || 'No summary available'}</p>
                                <div class="risk-assessment">
                                    <span>Risk Level: ${insights.risk_assessment || 'N/A'}</span>
                                    <span>Confidence: ${insights.confidence_level || 'N/A'}</span>
                                </div>
                            </div>
            `;
            
            // Add key factors if available
            if (insights.key_factors && insights.key_factors.length > 0) {
                html += `
                            <div class="insight-card">
                                <h6>Key Market Factors</h6>
                                <ul>
                `;
                insights.key_factors.forEach(factor => {
                    html += `<li>${factor}</li>`;
                });
                html += `
                                </ul>
                            </div>
                `;
            }
            
            html += `
                        </div>
            `;
            
            // Analysis tab
            html += `
                        <div id="analysis" class="tab-pane">
            `;
            
            if (insights.market_analysis) {
                html += `
                            <div class="insight-card">
                                <h6>Comprehensive Market Analysis</h6>
                                <div class="analysis-content">${formatAnalysisContent(insights.market_analysis)}</div>
                            </div>
                `;
            } else if (insights.llm_explanation) {
                html += `
                            <div class="insight-card">
                                <h6>AI-Powered Analysis</h6>
                                <div class="analysis-content">${formatAnalysisContent(insights.llm_explanation)}</div>
                            </div>
                `;
            } else if (insights.rule_based_explanation) {
                $('#analysis-section').append(`
                    <div class="analysis-card">
                        <h4><i class="fas fa-chart-line"></i> Market Analysis</h4>
                        <div class="analysis-content">${formatAnalysisContent(insights.rule_based_explanation)}</div>
                    </div>
                `);
            }

            html += `
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
                                <div class="recommendations-content">${formatRecommendationsContent(data.ai_suggestions)}</div>
                            </div>
                `;
            } else if (insights.llm_recommendations) {
                html += `
                            <div class="insight-card">
                                <h6>AI-Powered Recommendations</h6>
                                <div class="recommendations-content">${formatRecommendationsContent(insights.llm_recommendations)}</div>
                            </div>
                `;
            } else if (insights.recommendations && insights.recommendations.length > 0) {
                html += `
                            <div class="insight-card">
                                <h6>Market Recommendations</h6>
                                <ul>
                `;
                insights.recommendations.forEach(rec => {
                    html += `<li>${rec}</li>`;
                });
                html += `
                                </ul>
                            </div>
                `;
            } else if (insights.rule_based_recommendations) {
                $('#recommendations-section').append(`
                    <div class="recommendations-card">
                        <h4><i class="fas fa-lightbulb"></i> Recommendations</h4>
                        <div class="recommendations-content">${formatRecommendationsContent(insights.rule_based_recommendations)}</div>
                    </div>
                `);
            } else {
                html += `
                            <div class="insight-card">
                                <p>No specific recommendations available at this time.</p>
                            </div>
                `;
            }
            
            html += `
                        </div>
            `;
            
            // Market Intelligence tab
            html += `
                        <div id="intelligence" class="tab-pane">
            `;
            
            // Add market intelligence if available
            if (data.market_intelligence) {
                const intel = data.market_intelligence;
                html += `
                            <div class="insight-card">
                                <h6>Price Outlook</h6>
                                <p>${intel.price_outlook || 'No outlook available'}</p>
                            </div>
                `;
                
                if (intel.timing_advice) {
                    html += `
                            <div class="insight-card">
                                <h6>Timing Advice</h6>
                                <p>${intel.timing_advice}</p>
                            </div>
                `;
                }
                
                if (intel.risk_factors && intel.risk_factors.length > 0) {
                    html += `
                            <div class="insight-card">
                                <h6>Risk Factors</h6>
                                <ul>
                    `;
                    intel.risk_factors.forEach(factor => {
                        html += `<li>${factor}</li>`;
                    });
                    html += `
                                </ul>
                            </div>
                `;
                }
                
                if (intel.historical_context) {
                    html += `
                            <div class="insight-card">
                                <h6>Historical Context</h6>
                                <p>${intel.historical_context}</p>
                            </div>
                `;
                }
            } else {
                html += `
                            <div class="insight-card">
                                <p>No market intelligence available at this time.</p>
                            </div>
                `;
            }
            
            html += `
                        </div>
            `;
            
            html += `
                </div>
            `;
        }
        
        html += `
            </div>
        `;
        
        // Display results
        marketOutput.innerHTML = html;
        
        // Scroll to results
        marketResults.scrollIntoView({ behavior: 'smooth' });
    }
    
    /**
     * Switch between tabs in the insights section
     */
    window.switchMarketTab = function(tabName) {
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
     * Format analysis content for better display
     */
    function formatAnalysisContent(content) {
        if (!content) return '';
        return content
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>')
            .replace(/<br><br>/g, '</p><p>')
            .replace(/^<br>/, '')
            .replace(/<br>$/, '');
    }
    
    /**
     * Format recommendations content for better display
     */
    function formatRecommendationsContent(content) {
        if (!content) return '';
        // Convert numbered lists
        content = content.replace(/(\d+\.)\s/g, '<strong>$1</strong> ');
        return formatAnalysisContent(content);
    }
    
    /**
     * Show loading state
     */
    function showMarketLoading() {
        marketOutput.innerHTML = `
            <div class="loading">
                <div class="spinner"></div>
                <p>Analyzing market conditions and predicting prices with AI...</p>
                <p>Consulting knowledge base for enhanced insights...</p>
            </div>
        `;
        marketResults.style.display = 'block';
    }
    
    /**
     * Hide loading state
     */
    function hideMarketLoading() {
        // Loading state is automatically replaced when results are displayed
    }
    
    /**
     * Show error message
     */
    function showMarketError(message) {
        marketResults.style.display = 'block';
        marketOutput.innerHTML = `
            <div class="error-message">
                <h4>Error</h4>
                <p>${message}</p>
                <p>Please make sure you have trained the market prediction model first.</p>
            </div>
        `;
    }
    
    /**
     * Get CSRF token for Django
     */
    function getCSRFToken() {
        const meta = document.querySelector('meta[name="csrf-token"]');
        return meta ? meta.getAttribute('content') : '';
    }
});

// Initialize when the page loads
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initMarketPredictions);
} else {
    initMarketPredictions();
}

function initMarketPredictions() {
    console.log('Market predictions module initialized');
}