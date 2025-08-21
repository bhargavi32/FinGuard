# Test FinGuard Enhanced API with Explainable AI

Write-Host "============================================" -ForegroundColor Green
Write-Host "  TESTING EXPLAINABLE AI FEATURES" -ForegroundColor Green  
Write-Host "============================================" -ForegroundColor Green

Start-Sleep -Seconds 3

# Test 1: Health Check
Write-Host "`nTEST 1: Health Check (Enhanced Version)" -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method GET
    Write-Host "SUCCESS! Enhanced API is running!" -ForegroundColor Green
    Write-Host "Status: $($health.status)" -ForegroundColor White
    Write-Host "Version: $($health.version)" -ForegroundColor White
    Write-Host "Features: $($health.features)" -ForegroundColor Cyan
} catch {
    Write-Host "FAILED: Enhanced API not responding!" -ForegroundColor Red
    exit 1
}

# Test 2: Normal Transaction Explanation
Write-Host "`nTEST 2: Explainable AI - Normal Transaction" -ForegroundColor Yellow
Write-Host "Testing explainable AI for a normal 25.50 purchase..." -ForegroundColor Gray

$normalTransaction = @{
    Time = 43200
    Amount = 25.50
    V1 = 0.1; V2 = -0.1; V3 = 0.2; V4 = -0.2; V5 = 0.1
    V6 = -0.1; V7 = 0.1; V8 = -0.1; V9 = 0.1; V10 = -0.1
}

try {
    $body = $normalTransaction | ConvertTo-Json
    $explanation = Invoke-RestMethod -Uri "http://localhost:8000/fraud/explain" -Method POST -Body $body -ContentType "application/json"
    
    Write-Host "EXPLAINABLE AI RESULT:" -ForegroundColor Green
    Write-Host "Decision: $($explanation.prediction.is_fraud)" -ForegroundColor $(if($explanation.prediction.is_fraud) { "Red" } else { "Green" })
    Write-Host "Probability: $([math]::Round($explanation.prediction.fraud_probability * 100, 1))%" -ForegroundColor White
    Write-Host "Risk Level: $($explanation.prediction.risk_level)" -ForegroundColor Cyan
    Write-Host "`nAI EXPLANATION:" -ForegroundColor Cyan
    Write-Host $explanation.explanation -ForegroundColor White
    Write-Host "`nRECOMMENDATION:" -ForegroundColor Yellow
    Write-Host $explanation.recommendation -ForegroundColor White
    
    Write-Host "`nTOP FEATURE CONTRIBUTIONS:" -ForegroundColor Magenta
    foreach ($feature in $explanation.feature_importance[0..2]) {
        Write-Host "  - $($feature.feature): $($feature.importance) ($($feature.impact) risk)" -ForegroundColor Gray
    }
    
} catch {
    Write-Host "FAILED: Could not get explanation!" -ForegroundColor Red
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 3: Suspicious Transaction Explanation  
Write-Host "`n============================================" -ForegroundColor Green
Write-Host "TEST 3: Explainable AI - Suspicious Transaction" -ForegroundColor Yellow
Write-Host "Testing explainable AI for a suspicious 9999 purchase at 2 AM..." -ForegroundColor Gray

$suspiciousTransaction = @{
    Time = 7200
    Amount = 9999
    V1 = 4.0; V2 = 3.5; V3 = 3.0; V4 = -3.0; V5 = 4.0
    V6 = -4.0; V7 = 3.5; V8 = -3.5; V9 = 4.5; V10 = -4.0
}

try {
    $body = $suspiciousTransaction | ConvertTo-Json
    $explanation = Invoke-RestMethod -Uri "http://localhost:8000/fraud/explain" -Method POST -Body $body -ContentType "application/json"
    
    Write-Host "EXPLAINABLE AI RESULT:" -ForegroundColor Green
    Write-Host "Decision: $($explanation.prediction.is_fraud)" -ForegroundColor $(if($explanation.prediction.is_fraud) { "Red" } else { "Green" })
    Write-Host "Probability: $([math]::Round($explanation.prediction.fraud_probability * 100, 1))%" -ForegroundColor White
    Write-Host "Risk Level: $($explanation.prediction.risk_level)" -ForegroundColor $(if($explanation.prediction.risk_level -eq "high") { "Red" } else { "Yellow" })
    Write-Host "`nAI EXPLANATION:" -ForegroundColor Cyan
    Write-Host $explanation.explanation -ForegroundColor White
    Write-Host "`nRECOMMENDATION:" -ForegroundColor Yellow
    Write-Host $explanation.recommendation -ForegroundColor White
    
    Write-Host "`nTOP FEATURE CONTRIBUTIONS:" -ForegroundColor Magenta
    foreach ($feature in $explanation.feature_importance[0..2]) {
        Write-Host "  - $($feature.feature): $($feature.importance) ($($feature.impact) risk)" -ForegroundColor Gray
    }
    
} catch {
    Write-Host "FAILED: Could not get explanation!" -ForegroundColor Red
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 4: Model Performance
Write-Host "`n============================================" -ForegroundColor Green
Write-Host "TEST 4: Model Performance Metrics" -ForegroundColor Yellow

try {
    $performance = Invoke-RestMethod -Uri "http://localhost:8000/model/performance" -Method GET
    
    Write-Host "MODEL PERFORMANCE:" -ForegroundColor Green
    Write-Host "Accuracy: $($performance.accuracy)" -ForegroundColor White
    Write-Host "Precision: $($performance.precision)" -ForegroundColor White
    Write-Host "Recall: $($performance.recall)" -ForegroundColor White
    Write-Host "F1-Score: $($performance.f1_score)" -ForegroundColor White
    Write-Host "AUC-ROC: $($performance.auc_roc)" -ForegroundColor Cyan
    Write-Host "Training Data: $($performance.training_data)" -ForegroundColor Gray
    
} catch {
    Write-Host "FAILED: Could not get performance metrics!" -ForegroundColor Red
}

# Summary
Write-Host "`n============================================" -ForegroundColor Green
Write-Host "  EXPLAINABLE AI TESTING COMPLETE!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green

Write-Host "`nNEW FEATURES DEMONSTRATED:" -ForegroundColor Cyan
Write-Host "- Natural language explanations" -ForegroundColor White
Write-Host "- Feature importance analysis" -ForegroundColor White  
Write-Host "- SHAP-like value attribution" -ForegroundColor White
Write-Host "- Actionable recommendations" -ForegroundColor White
Write-Host "- Detailed model performance metrics" -ForegroundColor White

Write-Host "`nWHY THIS MATTERS FOR YOUR RESUME:" -ForegroundColor Yellow
Write-Host "- Shows understanding of AI ethics and transparency" -ForegroundColor White
Write-Host "- Demonstrates regulatory compliance awareness" -ForegroundColor White
Write-Host "- Proves ability to make AI systems explainable" -ForegroundColor White
Write-Host "- Ready for enterprise and banking applications" -ForegroundColor White

Write-Host "`nYOUR ENHANCED API: http://localhost:8000/docs" -ForegroundColor Green
Write-Host "New endpoint: /fraud/explain - The game changer!" -ForegroundColor Magenta

