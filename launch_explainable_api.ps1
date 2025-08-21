# FinGuard Explainable AI API Launcher
Write-Host "🛡️ FinGuard Explainable AI API" -ForegroundColor Green
Write-Host "===================================" -ForegroundColor Green

Write-Host "`n🧠 Starting Advanced Explainable AI API..." -ForegroundColor Yellow
Write-Host "💡 This API provides detailed fraud explanations" -ForegroundColor White
Write-Host "🔍 Features: Natural language reasoning, feature analysis" -ForegroundColor White

Write-Host "`n🚀 Launching Explainable API..." -ForegroundColor Cyan
Write-Host "📊 API will be available at: http://localhost:8000" -ForegroundColor White
Write-Host "📚 Documentation: http://localhost:8000/docs" -ForegroundColor White
Write-Host "💡 Press Ctrl+C to stop the API" -ForegroundColor Gray

# Stop current simple API if running (optional)
Write-Host "`n⚠️ Note: Stop your current simple API first (Ctrl+C)" -ForegroundColor Yellow
Write-Host "Then run: python explainable_fraud_api.py" -ForegroundColor Cyan

# Launch explainable API
python explainable_fraud_api.py
