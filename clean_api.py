#!/usr/bin/env python3
"""
Clean FinGuard API - Simple, Fast, Accurate
Following the architecture diagram with proper fraud detection
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
from typing import Dict, Any
from datetime import datetime
import os

# Load the trained fraud detection models
try:
    models = {}
    models['logistic_regression'] = joblib.load('models/logistic_regression.joblib')
    models['random_forest'] = joblib.load('models/random_forest.joblib')
    scaler = joblib.load('models/scaler.joblib')
    feature_names = joblib.load('models/feature_names.joblib')
    print("✅ Fraud detection models loaded successfully!")
except FileNotFoundError:
    print("❌ Models not found! Run clean_fraud_detector.py first")
    models = None

# Initialize FastAPI
app = FastAPI(
    title="🛡️ FinGuard Clean API",
    version="1.0.0",
    description="Clean, fast, accurate fraud detection API following enterprise architecture"
)

# Pydantic models
class TransactionInput(BaseModel):
    """Input model for fraud detection"""
    Time: float = Field(..., description="Transaction time in seconds")
    Amount: float = Field(..., ge=0, description="Transaction amount")
    V1: float = Field(..., description="Anonymized feature V1")
    V2: float = Field(..., description="Anonymized feature V2") 
    V3: float = Field(..., description="Anonymized feature V3")
    V4: float = Field(..., description="Anonymized feature V4")
    V5: float = Field(..., description="Anonymized feature V5")
    V6: float = Field(..., description="Anonymized feature V6")
    V7: float = Field(..., description="Anonymized feature V7")
    V8: float = Field(..., description="Anonymized feature V8")
    V9: float = Field(..., description="Anonymized feature V9")
    V10: float = Field(..., description="Anonymized feature V10")

class FraudPrediction(BaseModel):
    """Output model for fraud prediction"""
    is_fraud: bool = Field(..., description="Whether transaction is fraud")
    fraud_probability: float = Field(..., description="Fraud probability (0-1)")
    risk_level: str = Field(..., description="Risk level: minimal, low, medium, high, critical")
    confidence_score: float = Field(..., description="Model confidence (0-1)")
    explanation: str = Field(..., description="Human-readable explanation")
    model_details: Dict[str, Any] = Field(..., description="Individual model predictions")
    prediction_time_ms: float = Field(..., description="Prediction time in milliseconds")

def calculate_rule_bonus(transaction: dict) -> float:
    """Enhanced rule-based fraud detection for obvious patterns"""
    bonus = 0.0
    
    amount = float(transaction.get('Amount', 0))
    time_val = float(transaction.get('Time', 0))
    
    # Amount-based risk (aggressive for clear fraud patterns)
    if amount > 5000:
        bonus += 0.35  # Very high amounts
    elif amount > 3000:
        bonus += 0.25
    elif amount > 2000:
        bonus += 0.15
    elif amount > 1000:
        bonus += 0.08
    
    # Time-based risk (suspicious hours)
    hour = (time_val % 86400) // 3600
    if hour < 6 or hour > 22:  # Late night/early morning
        bonus += 0.3
    elif hour < 8 or hour > 20:  # Extended hours
        bonus += 0.1
    
    # Pattern-based risk (V features anomalies)
    high_v_count = 0
    extreme_v_count = 0
    
    for i in range(1, 11):
        v_key = f'V{i}'
        if v_key in transaction:
            v_val = abs(float(transaction[v_key]))
            if v_val > 3:
                extreme_v_count += 1
                high_v_count += 1
                bonus += 0.08
            elif v_val > 2:
                high_v_count += 1  
                bonus += 0.04
            elif v_val > 1.5:
                bonus += 0.02
    
    # Bonus for multiple anomalous features
    if extreme_v_count >= 4:
        bonus += 0.25
    elif extreme_v_count >= 2:
        bonus += 0.15
    elif high_v_count >= 6:
        bonus += 0.2
    elif high_v_count >= 4:
        bonus += 0.12
    elif high_v_count >= 2:
        bonus += 0.08
    
    return bonus

def generate_explanation(transaction: dict, fraud_prob: float) -> str:
    """Generate human-readable explanation for fraud decision"""
    amount = float(transaction.get('Amount', 0))
    time_val = float(transaction.get('Time', 0))
    hour = int((time_val % 86400) // 3600)
    
    risk_factors = []
    
    # Amount analysis
    if amount > 5000:
        risk_factors.append(f"Very high amount: ${amount:,.2f}")
    elif amount > 3000:
        risk_factors.append(f"High amount: ${amount:,.2f}")
    elif amount > 1000:
        risk_factors.append(f"Large amount: ${amount:,.2f}")
    
    # Time analysis
    if hour < 6 or hour > 22:
        risk_factors.append(f"Suspicious timing: {hour:02d}:00 (high-risk hours)")
    elif hour < 8 or hour > 20:
        risk_factors.append(f"Off-hours transaction: {hour:02d}:00")
    
    # Pattern analysis
    high_v_features = sum(1 for i in range(1, 11) 
                         if abs(float(transaction.get(f'V{i}', 0))) > 2)
    extreme_v_features = sum(1 for i in range(1, 11) 
                            if abs(float(transaction.get(f'V{i}', 0))) > 3)
    
    if extreme_v_features >= 3:
        risk_factors.append(f"Multiple extreme anomalies ({extreme_v_features} features)")
    elif high_v_features >= 4:
        risk_factors.append(f"Several unusual patterns ({high_v_features} features)")
    elif high_v_features >= 2:
        risk_factors.append(f"Some unusual patterns ({high_v_features} features)")
    
    # Generate explanation based on fraud probability
    if fraud_prob >= 0.8:
        base_msg = "🚨 CRITICAL FRAUD RISK detected"
    elif fraud_prob >= 0.6:
        base_msg = "🔴 HIGH fraud risk detected"
    elif fraud_prob >= 0.4:
        base_msg = "🟡 MEDIUM fraud risk detected"
    elif fraud_prob >= 0.2:
        base_msg = "🟠 LOW fraud risk detected"
    else:
        base_msg = "✅ Transaction appears legitimate"
    
    if risk_factors:
        return f"{base_msg} due to: {'; '.join(risk_factors)}"
    else:
        return f"{base_msg} - all indicators within normal ranges"

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "🛡️ FinGuard Clean API - Enterprise Fraud Detection",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "models_loaded": len(models) if models else 0,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/fraud/predict", response_model=FraudPrediction)
async def predict_fraud(transaction: TransactionInput):
    """
    Predict fraud for a transaction with accurate results and explanations
    
    This endpoint uses ensemble ML models + enhanced rule-based detection
    to provide accurate fraud detection with human-readable explanations.
    """
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded. Run clean_fraud_detector.py first")
    
    start_time = datetime.now()
    
    try:
        # Convert to dict for processing
        transaction_dict = transaction.model_dump()
        
        # Prepare features for ML models
        feature_data = np.array([[transaction_dict[fname] for fname in feature_names]])
        
        # Create DataFrame with proper feature names to avoid warnings
        import pandas as pd
        feature_df = pd.DataFrame(feature_data, columns=feature_names)
        feature_data_scaled = scaler.transform(feature_df)
        
        # Get predictions from each model
        model_predictions = {}
        probabilities = []
        
        for name, model in models.items():
            try:
                prob = model.predict_proba(feature_data_scaled)[0, 1]
                model_predictions[name] = {
                    'fraud_probability': float(prob),
                    'is_fraud': bool(prob > 0.5)
                }
                probabilities.append(prob)
            except Exception as e:
                print(f"Error with model {name}: {e}")
                # Fallback probability
                prob = 0.5
                model_predictions[name] = {
                    'fraud_probability': 0.5,
                    'is_fraud': False,
                    'error': str(e)
                }
                probabilities.append(prob)
        
        # Ensemble prediction (average)
        if probabilities:
            ensemble_prob = np.mean(probabilities)
        else:
            ensemble_prob = 0.5
        
        # Apply rule-based enhancement for obvious fraud patterns
        rule_bonus = calculate_rule_bonus(transaction_dict)
        final_prob = min(ensemble_prob + rule_bonus, 0.99)
        
        # Determine risk level
        if final_prob >= 0.8:
            risk_level = "critical"
        elif final_prob >= 0.6:
            risk_level = "high"
        elif final_prob >= 0.4:
            risk_level = "medium"
        elif final_prob >= 0.2:
            risk_level = "low"
        else:
            risk_level = "minimal"
        
        # Calculate confidence (distance from decision boundary)
        confidence = abs(final_prob - 0.5) * 2
        
        # Generate explanation
        explanation = generate_explanation(transaction_dict, final_prob)
        
        # Calculate prediction time
        prediction_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Add rule adjustment info to model details
        model_predictions['ensemble'] = {
            'ml_probability': float(ensemble_prob),
            'rule_bonus': float(rule_bonus),
            'final_probability': float(final_prob)
        }
        
        return FraudPrediction(
            is_fraud=bool(final_prob > 0.5),
            fraud_probability=float(final_prob),
            risk_level=risk_level,
            confidence_score=float(confidence),
            explanation=explanation,
            model_details=model_predictions,
            prediction_time_ms=float(prediction_time)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🛡️ Starting FinGuard Clean API...")
    print("🎯 Enterprise fraud detection with accurate results")
    print("📊 API Documentation: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
