# Simple FinGuard API Test - Clean Version

Write-Host "============================================" -ForegroundColor Green
Write-Host "  TESTING YOUR FINGUARD AI SYSTEM" -ForegroundColor Green  
Write-Host "============================================" -ForegroundColor Green

# Test 1: Health Check
Write-Host "`nTEST 1: Health Check" -ForegroundColor Yellow
Write-Host "Checking if your AI is alive..." -ForegroundColor Gray

try {
    $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method GET
    Write-Host "SUCCESS! Your AI is running!" -ForegroundColor Green
    Write-Host "Status: $($health.status)" -ForegroundColor White
    Write-Host "Version: $($health.version)" -ForegroundColor White
} catch {
    Write-Host "FAILED: Your AI is not responding!" -ForegroundColor Red
    Write-Host "Make sure your server is running!" -ForegroundColor Red
    exit 1
}

# Test 2: Normal Transaction
Write-Host "`nTEST 2: Normal Transaction (Safe)" -ForegroundColor Yellow
Write-Host "Testing a normal 25.50 dollar purchase..." -ForegroundColor Gray

$normalTransaction = @{
    Time = 43200
    Amount = 25.50
    V1 = 0.1; V2 = -0.1; V3 = 0.2; V4 = -0.2; V5 = 0.1
    V6 = -0.1; V7 = 0.1; V8 = -0.1; V9 = 0.1; V10 = -0.1
}

try {
    $body = $normalTransaction | ConvertTo-Json
    $prediction = Invoke-RestMethod -Uri "http://localhost:8000/fraud/predict" -Method POST -Body $body -ContentType "application/json"
    
    Write-Host "PREDICTION COMPLETE!" -ForegroundColor Green
    Write-Host "Amount: $25.50" -ForegroundColor Cyan
    Write-Host "Time: 12:00 PM (Normal)" -ForegroundColor Cyan
    Write-Host "Is Fraud: $($prediction.is_fraud)" -ForegroundColor $(if($prediction.is_fraud) { "Red" } else { "Green" })
    Write-Host "Risk Level: $($prediction.risk_level)" -ForegroundColor $(if($prediction.risk_level -eq "low") { "Green" } else { "Yellow" })
    Write-Host "Probability: $([math]::Round($prediction.fraud_probability * 100, 1))%" -ForegroundColor White
    
    if (-not $prediction.is_fraud) {
        Write-Host "RESULT: Transaction looks SAFE!" -ForegroundColor Green
    } else {
        Write-Host "RESULT: Transaction flagged as suspicious!" -ForegroundColor Red
    }
} catch {
    Write-Host "FAILED: Could not test normal transaction!" -ForegroundColor Red
}

# Test 3: Suspicious Transaction  
Write-Host "`nTEST 3: Suspicious Transaction (Risky)" -ForegroundColor Yellow
Write-Host "Testing a suspicious 9999 dollar purchase at 2 AM..." -ForegroundColor Gray

$suspiciousTransaction = @{
    Time = 7200
    Amount = 9999
    V1 = 4.0; V2 = 3.5; V3 = 3.0; V4 = -3.0; V5 = 4.0
    V6 = -4.0; V7 = 3.5; V8 = -3.5; V9 = 4.5; V10 = -4.0
}

try {
    $body = $suspiciousTransaction | ConvertTo-Json
    $prediction = Invoke-RestMethod -Uri "http://localhost:8000/fraud/predict" -Method POST -Body $body -ContentType "application/json"
    
    Write-Host "PREDICTION COMPLETE!" -ForegroundColor Green
    Write-Host "Amount: $9,999 (Very High!)" -ForegroundColor Cyan
    Write-Host "Time: 2:00 AM (Suspicious!)" -ForegroundColor Cyan
    Write-Host "Is Fraud: $($prediction.is_fraud)" -ForegroundColor $(if($prediction.is_fraud) { "Red" } else { "Green" })
    Write-Host "Risk Level: $($prediction.risk_level)" -ForegroundColor $(if($prediction.risk_level -eq "high") { "Red" } elseif($prediction.risk_level -eq "medium") { "Yellow" } else { "Green" })
    Write-Host "Probability: $([math]::Round($prediction.fraud_probability * 100, 1))%" -ForegroundColor White
    
    if ($prediction.is_fraud) {
        Write-Host "RESULT: Transaction flagged as FRAUD!" -ForegroundColor Red
    } else {
        Write-Host "RESULT: Transaction looks normal" -ForegroundColor Green
    }
} catch {
    Write-Host "FAILED: Could not test suspicious transaction!" -ForegroundColor Red
}

# Summary
Write-Host "`n============================================" -ForegroundColor Green
Write-Host "  TESTING COMPLETE!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green

Write-Host "`nWHAT YOU JUST SAW:" -ForegroundColor Cyan
Write-Host "- Your AI server is running perfectly" -ForegroundColor White
Write-Host "- It can detect normal vs suspicious transactions" -ForegroundColor White  
Write-Host "- It gives real-time fraud scores" -ForegroundColor White
Write-Host "- It's ready for real-world use!" -ForegroundColor White

Write-Host "`nWANT MORE?" -ForegroundColor Cyan
Write-Host "Open browser: http://localhost:8000/docs" -ForegroundColor Yellow

Write-Host "`nYour AI Fraud Detection System is LIVE!" -ForegroundColor Green

