#!/usr/bin/env python3
"""
Test the exact JSON format that works in Swagger UI
"""

import json

# Test Case 1: Normal Transaction (for Swagger UI)
normal_transaction = {
    "Time": 28800.0,
    "Amount": 127.50,
    "V1": -0.8,
    "V2": 0.6,
    "V3": 1.1,
    "V4": 0.3,
    "V5": -0.5,
    "V6": 0.4,
    "V7": 0.2,
    "V8": 0.1,
    "V9": 0.3,
    "V10": 0.2
}

# Test Case 2: Fraud Transaction (for Swagger UI)
fraud_transaction = {
    "Time": 14400.0,
    "Amount": 4850.00,
    "V1": 3.8,
    "V2": 4.2,
    "V3": 3.9,
    "V4": 3.5,
    "V5": -3.7,
    "V6": 3.6,
    "V7": 3.1,
    "V8": 2.8,
    "V9": 3.4,
    "V10": 3.2
}

print("🛡️ FinGuard API - Swagger UI Test Cases")
print("=" * 60)

print("\n📋 **Copy these JSON examples for Swagger UI testing:**")

print(f"\n✅ **Test Case 1: Normal Transaction**")
print("```json")
print(json.dumps(normal_transaction, indent=2))
print("```")

print(f"\n🚨 **Test Case 2: Fraud Transaction**")  
print("```json")
print(json.dumps(fraud_transaction, indent=2))
print("```")

print(f"\n🌐 **How to use in Swagger UI:**")
print(f"1. Go to: http://localhost:8000/docs")
print(f"2. Click on 'POST /fraud/predict'")
print(f"3. Click 'Try it out'")
print(f"4. Copy and paste one of the JSON examples above")
print(f"5. Click 'Execute'")

print(f"\n🎯 **Expected Results:**")
print(f"• Normal Transaction: ~0.2% fraud probability")
print(f"• Fraud Transaction: ~99% fraud probability")
