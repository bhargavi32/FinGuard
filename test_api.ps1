# PowerShell script to test FinGuard API

Write-Host "🚀 Testing FinGuard API" -ForegroundColor Green

# Test 1: Health Check
Write-Host "`n1. Testing Health Endpoint..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method GET
    Write-Host "✅ Health Status: $($health.status)" -ForegroundColor Green
    Write-Host "   Version: $($health.version)" -ForegroundColor White
} catch {
    Write-Host "❌ Health check failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 2: Root Endpoint
Write-Host "`n2. Testing Root Endpoint..." -ForegroundColor Yellow
try {
    $root = Invoke-RestMethod -Uri "http://localhost:8000/" -Method GET
    Write-Host "✅ Root Response: $($root.message)" -ForegroundColor Green
} catch {
    Write-Host "❌ Root endpoint failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 3: Fraud Prediction
Write-Host "`n3. Testing Fraud Prediction..." -ForegroundColor Yellow
try {
    $transaction = @{
        Time = 1000.5
        Amount = 150.75
        V1 = 0.144
        V2 = -0.359
        V3 = 1.123
        V4 = -0.267
        V5 = 0.567
        V6 = -1.234
        V7 = 0.890
        V8 = -0.456
        V9 = 1.789
        V10 = -0.123
    }
    
    $body = $transaction | ConvertTo-Json
    $prediction = Invoke-RestMethod -Uri "http://localhost:8000/fraud/predict" -Method POST -Body $body -ContentType "application/json"
    
    Write-Host "✅ Fraud Prediction Result:" -ForegroundColor Green
    Write-Host "   Is Fraud: $($prediction.is_fraud)" -ForegroundColor White
    Write-Host "   Probability: $($prediction.fraud_probability)" -ForegroundColor White
    Write-Host "   Risk Level: $($prediction.risk_level)" -ForegroundColor White
    
} catch {
    Write-Host "❌ Fraud prediction failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 4: High-Risk Transaction
Write-Host "`n4. Testing High-Risk Transaction..." -ForegroundColor Yellow
try {
    $highRiskTransaction = @{
        Time = 3600  # 1 AM
        Amount = 2500  # High amount
        V1 = 3.0
        V2 = 3.5
        V3 = 2.8
        V4 = -0.267
        V5 = 0.567
        V6 = -1.234
        V7 = 0.890
        V8 = -0.456
        V9 = 1.789
        V10 = -0.123
    }
    
    $body = $highRiskTransaction | ConvertTo-Json
    $prediction = Invoke-RestMethod -Uri "http://localhost:8000/fraud/predict" -Method POST -Body $body -ContentType "application/json"
    
    Write-Host "✅ High-Risk Prediction Result:" -ForegroundColor Green
    Write-Host "   Is Fraud: $($prediction.is_fraud)" -ForegroundColor White
    Write-Host "   Probability: $($prediction.fraud_probability)" -ForegroundColor White
    Write-Host "   Risk Level: $($prediction.risk_level)" -ForegroundColor White
    
} catch {
    Write-Host "❌ High-risk prediction failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 5: Available Products
Write-Host "`n5. Testing Products Endpoint..." -ForegroundColor Yellow
try {
    $products = Invoke-RestMethod -Uri "http://localhost:8000/recommendations/products" -Method GET
    Write-Host "✅ Products Available: $($products.total_products)" -ForegroundColor Green
    Write-Host "   Categories: $($products.categories -join ', ')" -ForegroundColor White
} catch {
    Write-Host "❌ Products endpoint failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n🎉 API Testing Complete!" -ForegroundColor Green
Write-Host "`n📖 API Documentation: http://localhost:8000/docs" -ForegroundColor Blue
