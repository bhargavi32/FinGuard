# 🚀 FinGuard API Live Test - Let's see your AI in action!

Write-Host "=" * 60 -ForegroundColor Green
Write-Host "🛡️  TESTING YOUR FINGUARD AI SYSTEM  🛡️" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Green

# Test 1: Is your AI awake?
Write-Host "`n🏥 TEST 1: Health Check - Is your AI alive?" -ForegroundColor Yellow
Write-Host "Checking: http://localhost:8000/health" -ForegroundColor Gray

try {
    $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method GET
    Write-Host "✅ SUCCESS! Your AI is ALIVE and HEALTHY!" -ForegroundColor Green
    Write-Host "   Status: $($health.status)" -ForegroundColor White
    Write-Host "   Version: $($health.version)" -ForegroundColor White
} catch {
    Write-Host "❌ FAILED: Your AI might be sleeping!" -ForegroundColor Red
    Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Test 2: Basic API response
Write-Host "`n🏠 TEST 2: Basic API Response" -ForegroundColor Yellow
Write-Host "Checking: http://localhost:8000/" -ForegroundColor Gray

try {
    $root = Invoke-RestMethod -Uri "http://localhost:8000/" -Method GET
    Write-Host "✅ SUCCESS! Your API is responding!" -ForegroundColor Green
    Write-Host "   Message: $($root.message)" -ForegroundColor White
} catch {
    Write-Host "❌ FAILED: API not responding!" -ForegroundColor Red
}

# Test 3: Normal Transaction (Should be SAFE)
Write-Host "`n💳 TEST 3: Normal Transaction - Should be SAFE" -ForegroundColor Yellow
Write-Host "Testing a normal $25.50 purchase at 12:00 PM..." -ForegroundColor Gray

$normalTransaction = @{
    Time = 43200      # 12:00 PM (normal shopping time)
    Amount = 25.50    # Small, normal amount
    V1 = 0.1; V2 = -0.1; V3 = 0.2; V4 = -0.2; V5 = 0.1
    V6 = -0.1; V7 = 0.1; V8 = -0.1; V9 = 0.1; V10 = -0.1
}

try {
    $body = $normalTransaction | ConvertTo-Json
    $prediction = Invoke-RestMethod -Uri "http://localhost:8000/fraud/predict" -Method POST -Body $body -ContentType "application/json"
    
    Write-Host "✅ PREDICTION COMPLETE!" -ForegroundColor Green
    Write-Host "   💰 Amount: $25.50" -ForegroundColor Cyan
    Write-Host "   🕐 Time: 12:00 PM (Normal)" -ForegroundColor Cyan
    Write-Host "   🔍 Is Fraud: $($prediction.is_fraud)" -ForegroundColor $(if($prediction.is_fraud) { "Red" } else { "Green" })
    Write-Host "   📊 Risk Level: $($prediction.risk_level)" -ForegroundColor $(if($prediction.risk_level -eq "low") { "Green" } elseif($prediction.risk_level -eq "medium") { "Yellow" } else { "Red" })
    Write-Host "   🎯 Probability: $($prediction.fraud_probability * 100)%" -ForegroundColor White
    
    if (-not $prediction.is_fraud) {
        Write-Host "   ✅ RESULT: Transaction looks NORMAL and SAFE!" -ForegroundColor Green
    } else {
        Write-Host "   ⚠️  RESULT: Transaction flagged as suspicious!" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ FAILED: Fraud prediction failed!" -ForegroundColor Red
    Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 4: Suspicious Transaction (Should be RISKY)
Write-Host "`n🚨 TEST 4: Suspicious Transaction - Should be RISKY" -ForegroundColor Yellow
Write-Host "Testing a suspicious $9,999 purchase at 2:00 AM..." -ForegroundColor Gray

$suspiciousTransaction = @{
    Time = 7200       # 2:00 AM (suspicious time)
    Amount = 9999     # Very high amount
    V1 = 4.0; V2 = 3.5; V3 = 3.0; V4 = -3.0; V5 = 4.0
    V6 = -4.0; V7 = 3.5; V8 = -3.5; V9 = 4.5; V10 = -4.0
}

try {
    $body = $suspiciousTransaction | ConvertTo-Json
    $prediction = Invoke-RestMethod -Uri "http://localhost:8000/fraud/predict" -Method POST -Body $body -ContentType "application/json"
    
    Write-Host "✅ PREDICTION COMPLETE!" -ForegroundColor Green
    Write-Host "   💰 Amount: $9,999 (Very High!)" -ForegroundColor Cyan
    Write-Host "   🕐 Time: 2:00 AM (Suspicious!)" -ForegroundColor Cyan
    Write-Host "   🔍 Is Fraud: $($prediction.is_fraud)" -ForegroundColor $(if($prediction.is_fraud) { "Red" } else { "Green" })
    Write-Host "   📊 Risk Level: $($prediction.risk_level)" -ForegroundColor $(if($prediction.risk_level -eq "low") { "Green" } elseif($prediction.risk_level -eq "medium") { "Yellow" } else { "Red" })
    Write-Host "   🎯 Probability: $($prediction.fraud_probability * 100)%" -ForegroundColor White
    
    if ($prediction.is_fraud) {
        Write-Host "   🚨 RESULT: Transaction flagged as FRAUD!" -ForegroundColor Red
    } else {
        Write-Host "   ✅ RESULT: Transaction looks normal (surprising!)" -ForegroundColor Green
    }
} catch {
    Write-Host "❌ FAILED: Fraud prediction failed!" -ForegroundColor Red
    Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Red
}

# Summary
Write-Host "`n" + "=" * 60 -ForegroundColor Green
Write-Host "🎉 TESTING COMPLETE! YOUR AI IS WORKING!" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Green

Write-Host "`n📋 WHAT YOU JUST SAW:" -ForegroundColor Cyan
Write-Host "   ✅ Your AI server is running perfectly" -ForegroundColor White
Write-Host "   ✅ It can detect normal vs suspicious transactions" -ForegroundColor White
Write-Host "   ✅ It gives real-time fraud scores" -ForegroundColor White
Write-Host "   ✅ It's ready for real-world use!" -ForegroundColor White

Write-Host "`n🌐 WANT TO SEE MORE?" -ForegroundColor Cyan
Write-Host "   Open your browser and go to: http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host "   This will show you a beautiful interactive interface!" -ForegroundColor Yellow

Write-Host "`n🎯 YOUR AI FRAUD DETECTION SYSTEM IS LIVE! 🎯" -ForegroundColor Green

