# FinGuard Dashboard Launcher - PowerShell Version
Write-Host "🛡️ FinGuard AI Platform" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green

Write-Host "`n📦 Checking requirements..." -ForegroundColor Yellow

# Check if Streamlit is available
try {
    python -c "import streamlit; print('✅ Streamlit - OK')"
    python -c "import plotly; print('✅ Plotly - OK')"
    python -c "import requests; print('✅ Requests - OK')"
} catch {
    Write-Host "⚠️ Missing packages detected. Installing..." -ForegroundColor Yellow
    pip install streamlit plotly requests
}

Write-Host "`n🚀 Launching Dashboard..." -ForegroundColor Cyan
Write-Host "📊 Opening in browser: http://localhost:8501" -ForegroundColor White
Write-Host "💡 Press Ctrl+C to stop the dashboard" -ForegroundColor Gray

# Launch Streamlit dashboard
streamlit run dashboard_app.py --server.headless false --server.address localhost --server.port 8501
